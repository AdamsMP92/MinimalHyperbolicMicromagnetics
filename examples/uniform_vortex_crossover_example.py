"""Radius sweep through the uniform-to-vortex crossover.

For every radius, the script calculates a complete major hysteresis loop for
the two hyperbolic models H' and H''. Existing complete calculations are
treated as a cache and are never overwritten.

To fill the corrected H' data to a 0.1 nm grid with four workers, use::

    pixi run uniform-vortex-crossover --models Hp --radius-step-nm 0.1 --workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minimal_hyperbolic_micromagnetics import (
    HysteresisResult,
    HysteresisSettings,
    ModelParameters,
    ProfileComputation,
    analyze_hysteresis,
    compute_and_store_hysteresis,
    compute_profiles,
    vortex_nucleation_field,
    vortex_to_uniform_field_from_hysteresis,
)


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "uniform_vortex_crossover_example_output"
)
KU, MS, A = 4.8e4, 1.7e6, 1.0e-11
MODEL_FACTORS = {"Hpp": 0.0, "Hp": 1.0}
BMAX, LOW_STEP, MID_STEP, OUTER_STEP = 1.0, 1e-4, 1e-3, 5e-3

# Initialized once per process so that every worker reuses the same profile
# table instead of recomputing it for every radius.
_WORKER_PROFILES = None
_WORKER_FIELDS = None
_WORKER_DESCENDING_LENGTH = None
_WORKER_OUTPUT = None


def segment(start: float, stop: float, step: float) -> np.ndarray:
    """Return an endpoint-inclusive field segment."""

    return np.linspace(start, stop, round(abs(stop - start) / step) + 1)


def radius_grid(start_nm: float, stop_nm: float, step_nm: float) -> np.ndarray:
    """Build an inclusive decimal grid without cumulative floating-point drift."""

    if step_nm <= 0.0 or stop_nm < start_nm:
        raise ValueError("radius grid requires step > 0 and stop >= start")
    intervals = round((stop_nm - start_nm) / step_nm)
    if not np.isclose(start_nm + intervals * step_nm, stop_nm, atol=1e-10):
        raise ValueError("radius range must contain an integer number of steps")
    return np.round(start_nm + step_nm * np.arange(intervals + 1), 10)


def build_field_protocol(radii_nm: np.ndarray) -> tuple[np.ndarray, int]:
    """Construct the common major-loop field path for the radius sweep."""

    bk = 2.0 * KU / MS
    bnuc = max(0.0, np.max(vortex_nucleation_field(KU, MS, A, radii_nm * 1e-9)))
    b_fine = np.ceil(1.25 * bk / MID_STEP) * MID_STEP
    b_vortex = np.ceil((bnuc + 1.5 * bk) / OUTER_STEP) * OUTER_STEP
    descending = np.concatenate([
        segment(BMAX, b_vortex, OUTER_STEP),
        segment(b_vortex, b_fine, MID_STEP)[1:],
        segment(b_fine, -b_fine, LOW_STEP)[1:],
        segment(-b_fine, -b_vortex, MID_STEP)[1:],
        segment(-b_vortex, -BMAX, OUTER_STEP)[1:],
    ])
    return np.concatenate([descending, descending[::-1]]), len(descending)


def output_paths(output: Path, model_name: str, radius_nm: float) -> tuple[Path, Path]:
    loop = output / model_name / f"hysteresis_r{radius_nm:.1f}.csv"
    return loop, loop.with_name(f"{loop.stem}_metadata.csv")


def output_state(output: Path, model_name: str, radius_nm: float) -> str:
    """Return ``complete``, ``missing``, or ``partial`` for one radius."""

    loop, metadata = output_paths(output, model_name, radius_nm)
    present = loop.exists(), metadata.exists()
    if all(present):
        return "complete"
    if any(present):
        return "partial"
    return "missing"


def observables_row(result, model_name: str, radius_nm: float, descending_length: int):
    """Extract the compact radius-dependent quantities used by the figures."""

    analysis = analyze_hysteresis(result)
    zero = np.flatnonzero(
        np.isclose(result.B_T[:descending_length], 0.0, rtol=0.0, atol=1e-14)
    )[0]
    remanent_nu = result.nu_min[zero]
    # Panel (b) compares the same path-dependent event for every method: the
    # positive-field endpoint of the vortex on the ascending leg. This is not
    # generally identical to the uniform-state nucleation spinodal B_nuc.
    return_field = vortex_to_uniform_field_from_hysteresis(result)
    vortex_uniform_field = (
        return_field
        if remanent_nu > 1.0e-12 and return_field >= 0.0
        else float("nan")
    )
    return {
        "model": model_name,
        "radius_nm": radius_nm,
        "remanence": analysis.descending.remanence,
        "coercive_field_T": abs(analysis.descending.coercive_field_T),
        "vortex_uniform_field_T": vortex_uniform_field,
        "remanent_nu": remanent_nu,
        "remanent_tau_rad": result.tau_rad[zero],
    }


def load_result(path: Path) -> HysteresisResult:
    """Load a stored loop if its compact summary row is absent."""

    table = pd.read_csv(path)
    required = {
        "B_T": "B_T",
        "mz_avg": "mz_avg",
        "nu_min": "nu_min",
        "tau_rad": "tau_min_rad",
        "energy": "energy",
    }
    optional = [
        "stability_nu_curvature",
        "stability_mixed_curvature",
        "stability_tau_curvature",
        "stability_eigenvalue_min",
        "stability_eigenvalue_max",
        "uniform_tau_rad",
        "uniform_vortex_curvature",
        "uniform_orientation_curvature",
        "uniform_stability_eigenvalue_min",
        "uniform_stability_eigenvalue_max",
    ]
    values = {
        target: table[source].to_numpy(dtype=float)
        for target, source in required.items()
    }
    values.update({
        name: table[name].to_numpy(dtype=float)
        for name in optional
        if name in table
    })
    return HysteresisResult(**values)


def initialize_worker(profiles, fields, descending_length, output):
    global _WORKER_PROFILES, _WORKER_FIELDS, _WORKER_DESCENDING_LENGTH, _WORKER_OUTPUT
    _WORKER_PROFILES = profiles
    _WORKER_FIELDS = fields
    _WORKER_DESCENDING_LENGTH = descending_length
    _WORKER_OUTPUT = Path(output)


def calculate_radius(task):
    """Compute and publish one previously missing radius."""

    model_name, gux_factor, radius_nm = task
    target, target_metadata = output_paths(_WORKER_OUTPUT, model_name, radius_nm)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(target.suffix + ".lock")
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(
            f"radius is already claimed: {model_name} R={radius_nm:.1f}"
        ) from error

    staging = (
        target.parent
        / ".staging"
        / f"{target.stem}-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging_target = staging / target.name
    try:
        state = output_state(_WORKER_OUTPUT, model_name, radius_nm)
        if state != "missing":
            raise RuntimeError(
                f"refusing to overwrite {state} output for "
                f"{model_name} R={radius_nm:.1f}"
            )
        model = ModelParameters(KU, MS, A, radius_nm * 1e-9, gux_factor=gux_factor)
        result = compute_and_store_hysteresis(
            staging_target,
            model,
            _WORKER_PROFILES,
            settings=HysteresisSettings(fields=_WORKER_FIELDS),
        )
        staging_metadata = staging_target.with_name(
            f"{staging_target.stem}_metadata.csv"
        )
        # Both files remain invisible until the calculation completed.
        staging_target.rename(target)
        staging_metadata.rename(target_metadata)
        return observables_row(
            result, model_name, radius_nm, _WORKER_DESCENDING_LENGTH
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        lock.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_FACTORS),
        default=tuple(MODEL_FACTORS),
    )
    parser.add_argument(
        "--radii-nm",
        nargs="+",
        type=float,
        default=None,
        help="explicit radii in nm; overrides the regular grid",
    )
    parser.add_argument("--radius-min-nm", type=float, default=6.0)
    parser.add_argument("--radius-max-nm", type=float, default=20.0)
    parser.add_argument("--radius-step-nm", type=float, default=0.5)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent radius processes (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report cached and missing cases without computing",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def update_summary(output, rows, models, radii_nm, descending_length):
    """Merge new rows while retaining every untouched cached calculation."""

    summary_path = output / "radius_observables.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    new_keys = {
        (row["model"], round(float(row["radius_nm"]), 10)) for row in rows
    }
    for model_name in models:
        for radius_nm in radii_nm:
            key = model_name, round(float(radius_nm), 10)
            if key in new_keys:
                continue
            loop, _ = output_paths(output, model_name, radius_nm)
            if loop.exists():
                rows.append(
                    observables_row(
                        load_result(loop), model_name, radius_nm, descending_length
                    )
                )

    additions = pd.DataFrame(rows)
    if not additions.empty and not summary.empty:
        replacement = pd.MultiIndex.from_frame(additions[["model", "radius_nm"]])
        old = pd.MultiIndex.from_frame(summary[["model", "radius_nm"]])
        summary = summary.loc[~old.isin(replacement)]
    summary = pd.concat([summary, additions], ignore_index=True)
    summary = summary.sort_values(["model", "radius_nm"]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    columns = [
        "remanence",
        "vortex_uniform_field_T",
        "remanent_tau_rad",
        "coercive_field_T",
    ]
    labels = [r"$m_r$", r"$B_{v\to u}$ (T)", r"$\tau_{rem}$ (rad)", r"$B_c$ (T)"]
    figure, axes = plt.subplots(
        2, 2, figsize=(8, 6), sharex=True, constrained_layout=True
    )
    for axis, column, label in zip(axes.flat, columns, labels):
        summary.pivot(index="radius_nm", columns="model", values=column).plot(ax=axis)
        axis.set(xlabel=r"$R$ (nm)", ylabel=label)
        axis.grid(alpha=0.2)
    figure.savefig(output / "uniform_vortex_crossover.png", dpi=300)
    plt.close(figure)


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    radii_nm = (
        np.asarray(args.radii_nm, dtype=float)
        if args.radii_nm is not None
        else radius_grid(args.radius_min_nm, args.radius_max_nm, args.radius_step_nm)
    )
    radii_nm = np.unique(np.round(radii_nm, 10))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    partial = [
        (name, radius_nm)
        for name in args.models
        for radius_nm in radii_nm
        if output_state(output, name, radius_nm) == "partial"
    ]
    if partial:
        details = ", ".join(f"{name} R={radius:.1f}" for name, radius in partial)
        raise RuntimeError(f"incomplete cached output; inspect before resuming: {details}")

    tasks = [
        (name, MODEL_FACTORS[name], float(radius_nm))
        for name in args.models
        for radius_nm in radii_nm
        if output_state(output, name, radius_nm) == "missing"
    ]
    total = len(args.models) * len(radii_nm)
    print(
        f"Radius sweep: target={total}, cached={total - len(tasks)}, "
        f"missing={len(tasks)}, workers={args.workers}",
        flush=True,
    )
    if args.dry_run:
        return

    fields, descending_length = build_field_protocol(radii_nm)
    rows = []
    if tasks:
        profiles = compute_profiles(
            ProfileComputation(n_nu=2000, n_quad=360, l_max_demag=161)
        )
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=(profiles, fields, descending_length, output),
        ) as executor:
            futures = {executor.submit(calculate_radius, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                name, _, radius_nm = futures[future]
                rows.append(future.result())
                print(
                    f"[{index:03d}/{len(tasks):03d}] completed "
                    f"{name} R={radius_nm:.1f} nm",
                    flush=True,
                )
    else:
        print("Nothing to calculate; all requested radii are already cached.")

    update_summary(output, rows, args.models, radii_nm, descending_length)
    print(f"Summary: {output / 'radius_observables.csv'}", flush=True)


if __name__ == "__main__":
    main()
