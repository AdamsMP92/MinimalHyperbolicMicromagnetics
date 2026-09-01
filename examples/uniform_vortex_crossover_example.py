"""Resolve the uniform-to-vortex crossover with the hyperbolic models.

The calculation covers both reduced Hamiltonians on a configurable radius
grid. It evaluates descending remanence, the positive-field vortex-to-uniform
return, the remanent texture-axis angle, and the coercive-field magnitude. The
complete field-resolved histories are retained so that additional observables,
in particular ``nu(B, R)`` and ``tau(B, R)``, can be analyzed without rerunning
the hysteresis calculations.

The field path is intentionally nonuniform. A 0.1 mT default spacing is used
between -70 mT and +70 mT, where coercivity is extracted, while larger steps
are sufficient away from that interval. This resolves the low-field curve ten
times more finely than the original 1 mT data without introducing a new
adaptive-refinement layer into the package.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minimal_hyperbolic_micromagnetics import (
    HysteresisSettings,
    ModelParameters,
    ProfileComputation,
    analyze_hysteresis,
    compute_profiles,
    run_hysteresis,
    split_field_branches,
    vortex_nucleation_field,
    vortex_nucleation_radius,
)


DEFAULT_OUTPUT_DIR = HERE / "uniform_vortex_crossover_example_output"

KU = 4.8e4
MS = 1.7e6
A = 1.0e-11
BMAX = 1.0
BETA_DEG = 0.0

RADIUS_MIN_NM = 6.0
RADIUS_MAX_NM = 20.0
RADIUS_STEP_NM = 0.5

PROFILE_SETTINGS = ProfileComputation(
    nu_min=0.0,
    nu_max=20.0,
    n_nu=2000,
    n_quad=360,
    l_max_demag=161,
)

MODEL_VARIANTS = {
    "Hpp": 0.0,
    "Hp": 1.0,
}

MODEL_LABELS = {
    "Hpp": r"$\mathcal{H}^{\prime\prime}$",
    "Hp": r"$\mathcal{H}^\prime$",
}

MODEL_COLORS = {
    "Hpp": "black",
    "Hp": "#CC79A7",
}

VORTEX_THRESHOLD = 1.0e-6


def radius_tag(radius_nm: float) -> str:
    """Return the stable filename representation used by the paper archive."""

    return f"{radius_nm:.1f}".replace(".", "p")


def field_segment(start: float, stop: float, step: float) -> np.ndarray:
    """Return an endpoint-inclusive monotonic segment with a validated step."""

    interval = abs(stop - start)
    count = int(round(interval / step))
    if count < 1 or not np.isclose(count * step, interval, atol=1.0e-14):
        raise ValueError(
            f"step {step:g} T does not divide [{start:g}, {stop:g}] T"
        )
    return np.linspace(start, stop, count + 1)


def join_segments(*segments: np.ndarray) -> np.ndarray:
    """Join endpoint-inclusive segments without duplicating shared endpoints."""

    joined = [np.asarray(segments[0], dtype=float)]
    joined.extend(np.asarray(segment[1:], dtype=float) for segment in segments[1:])
    return np.concatenate(joined)


def make_field_protocol(low_field_step_mT: float = 0.1) -> np.ndarray:
    """Build the complete major loop used for the radius sweep."""

    low_step = float(low_field_step_mT) * 1.0e-3
    if low_step <= 0.0:
        raise ValueError("low_field_step_mT must be positive")

    descending = join_segments(
        field_segment(BMAX, 0.60, 5.0e-3),
        field_segment(0.60, 0.07, 1.0e-3),
        field_segment(0.07, -0.07, low_step),
        field_segment(-0.07, -0.60, 1.0e-3),
        field_segment(-0.60, -BMAX, 5.0e-3),
    )
    # Retain the turning point on both branches.  This makes each branch a
    # self-contained endpoint-inclusive field path in the archived tables.
    return np.concatenate([descending, descending[::-1]])


def interpolate_zero(x: np.ndarray, y: np.ndarray, index: int) -> float:
    """Interpolate x at y=0 between index and index+1."""

    x0, x1 = float(x[index]), float(x[index + 1])
    y0, y1 = float(y[index]), float(y[index + 1])
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def positive_uniform_restabilization_field(result) -> float:
    """Return the positive-field stability recovery of the uniform state."""

    ascending = next(
        branch
        for branch in split_field_branches(result)
        if branch.name == "ascending"
    )
    fields = np.asarray(ascending.B_T, dtype=float)
    curvature = np.asarray(ascending.uniform_vortex_curvature, dtype=float)
    finite = np.isfinite(curvature)
    crossings = np.flatnonzero(
        (fields[:-1] >= 0.0)
        & finite[:-1]
        & finite[1:]
        & (curvature[:-1] <= 0.0)
        & (curvature[1:] > 0.0)
    )
    if len(crossings) == 0:
        return float("nan")
    return interpolate_zero(fields, curvature, int(crossings[0]))


def selected_vortex_uniform_return_field(result) -> float:
    """Return the resolved selected-state vortex-to-uniform transition."""

    ascending = next(
        branch
        for branch in split_field_branches(result)
        if branch.name == "ascending"
    )
    fields = np.asarray(ascending.B_T, dtype=float)
    nu = np.asarray(ascending.nu_min, dtype=float)
    crossings = np.flatnonzero(
        (fields[:-1] >= 0.0)
        & (nu[:-1] > VORTEX_THRESHOLD)
        & (nu[1:] <= VORTEX_THRESHOLD)
    )
    if len(crossings) == 0:
        return float("nan")
    return interpolate_zero(
        fields,
        nu - VORTEX_THRESHOLD,
        int(crossings[0]),
    )


def remanent_state(result) -> tuple[float, float]:
    """Return nu and tau at the exact descending B=0 field point."""

    descending = next(
        branch
        for branch in split_field_branches(result)
        if branch.name == "descending"
    )
    zero = np.flatnonzero(np.isclose(descending.B_T, 0.0, atol=1.0e-14))
    if len(zero) != 1:
        raise RuntimeError("field protocol must contain one descending B=0 point")
    index = int(zero[0])
    return float(descending.nu_min[index]), float(descending.tau_rad[index])


def texture_axis_angle_deg(nu: float, tau: float) -> float:
    """Map the unoriented vortex axis onto the interval from 0 to 90 degrees."""

    if nu <= VORTEX_THRESHOLD:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(abs(np.cos(tau)), 0.0, 1.0))))


def branch_labels(fields: np.ndarray) -> np.ndarray:
    """Return one storage label for each point of the complete loop."""

    turning = int(np.argmin(fields))
    labels = np.full(len(fields), "ascending", dtype=object)
    labels[: turning + 1] = "descending"
    return labels


def store_loop(
    path: Path,
    result,
    *,
    radius_nm: float,
    variant: str,
    gux_factor: float,
) -> None:
    """Store the complete state and stability history for one radius."""

    table = pd.DataFrame(result.as_dict())
    table = table.rename(
        columns={
            "nu_min": "nu",
            "tau_min_rad": "tau_rad",
        }
    )
    branches = branch_labels(np.asarray(result.B_T, dtype=float))
    branch_index = np.zeros(len(table), dtype=int)
    for branch in ("descending", "ascending"):
        selected = branches == branch
        branch_index[selected] = np.arange(np.count_nonzero(selected))

    table.insert(0, "field_index", np.arange(len(table)))
    table.insert(1, "branch", branches)
    table.insert(2, "branch_index", branch_index)
    table.insert(3, "radius_nm", float(radius_nm))
    table.insert(4, "model_variant", variant)
    table.insert(5, "gux_factor", float(gux_factor))
    tau_column = int(table.columns.get_loc("tau_rad"))
    table.insert(
        tau_column + 1,
        "vortex_present",
        (table["nu"].to_numpy() > VORTEX_THRESHOLD).astype(int),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def store_run_metadata(
    path: Path,
    *,
    radii_nm: np.ndarray,
    variants: tuple[str, ...],
    fields: np.ndarray,
    profile_settings: ProfileComputation,
    save_full_loops: bool,
) -> None:
    """Store the physical, numerical, and column-definition provenance."""

    metadata = {
        "schema_version": 1,
        "example": "uniform_vortex_crossover_example.py",
        "material_parameters": {
            "Ku_J_per_m3": KU,
            "Ms_A_per_m": MS,
            "A_J_per_m": A,
            "beta_deg": BETA_DEG,
        },
        "model_variants": {
            variant: {"gux_factor": MODEL_VARIANTS[variant]}
            for variant in variants
        },
        "radii_nm": [float(value) for value in radii_nm],
        "field_protocol": {
            "maximum_field_T": BMAX,
            "number_of_points": int(len(fields)),
            "turning_point_duplicated_between_branches": True,
            "segments_descending": [
                {"start_T": 1.0, "stop_T": 0.60, "step_T": 5.0e-3},
                {"start_T": 0.60, "stop_T": 0.07, "step_T": 1.0e-3},
                {
                    "start_T": 0.07,
                    "stop_T": -0.07,
                    "step_T": float(
                        np.min(np.abs(np.diff(fields))[np.diff(fields) != 0.0])
                    ),
                },
                {"start_T": -0.07, "stop_T": -0.60, "step_T": 1.0e-3},
                {"start_T": -0.60, "stop_T": -1.0, "step_T": 5.0e-3},
            ],
        },
        "profile_computation": {
            "nu_min": profile_settings.nu_min,
            "nu_max": profile_settings.nu_max,
            "n_nu": profile_settings.n_nu,
            "n_quad": profile_settings.n_quad,
            "l_max_demag": profile_settings.l_max_demag,
        },
        "storage": {
            "full_loops_saved": bool(save_full_loops),
            "loop_pattern": "<model>/hysteresis_r<RADIUS>.csv",
            "radius_summary": "<model>/radius_observables.csv",
            "profiles": "profiles.csv",
            "field_protocol": "field_protocol.csv",
        },
        "column_definitions": {
            "nu": "dimensionless hyperbolic-vortex profile parameter",
            "tau_rad": "polar rotation coordinate used internally by the model",
            "vortex_present": f"1 where nu > {VORTEX_THRESHOLD:g}, else 0",
        },
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def run_radius_series(
    *,
    radii_nm: np.ndarray,
    variants: tuple[str, ...],
    fields: np.ndarray,
    profiles: dict[str, np.ndarray],
    output_dir: Path,
    save_full_loops: bool,
) -> dict[str, pd.DataFrame]:
    """Calculate and store all selected model and radius combinations."""

    summaries: dict[str, pd.DataFrame] = {}
    field_steps = np.abs(np.diff(fields))
    minimum_field_step = float(np.min(field_steps[field_steps > 0.0]))
    critical_radius_nm = float(vortex_nucleation_radius(KU, MS, A) * 1.0e9)

    for variant in variants:
        gux_factor = MODEL_VARIANTS[variant]
        variant_dir = output_dir / variant
        rows = []

        for radius_nm in radii_nm:
            start = time.perf_counter()
            model = ModelParameters(
                Ku=KU,
                Ms=MS,
                A=A,
                R=float(radius_nm) * 1.0e-9,
                beta_deg=BETA_DEG,
                gux_factor=gux_factor,
            )
            result = run_hysteresis(
                model,
                profiles,
                settings=HysteresisSettings(
                    fields=fields,
                    stoner_wohlfarth=False,
                ),
            )
            analysis = analyze_hysteresis(result, nucleation_method="stability")
            nu_rem, tau_rem = remanent_state(result)
            analytic_return = float(
                vortex_nucleation_field(KU, MS, A, model.R)
            )
            if radius_nm < critical_radius_nm:
                analytic_return = float("nan")

            row = {
                "radius_nm": float(radius_nm),
                "remanence": float(analysis.descending.remanence),
                "remanent_tau_rad": tau_rem,
                "remanent_nu": nu_rem,
                "remanent_axis_angle_deg": texture_axis_angle_deg(
                    nu_rem,
                    tau_rem,
                ),
                "vortex_present": int(nu_rem > VORTEX_THRESHOLD),
                "coercive_field_T": abs(
                    float(analysis.descending.coercive_field_T)
                ),
                "reverse_vortex_uniform_field_T": (
                    positive_uniform_restabilization_field(result)
                ),
                "selected_reverse_vortex_uniform_field_T": (
                    selected_vortex_uniform_return_field(result)
                ),
                "descending_vortex_nucleation_field_T": float(
                    analysis.descending.vortex_nucleation_field_T
                ),
                "analytic_vortex_nucleation_field_T": analytic_return,
                "field_points": len(fields),
                "minimum_field_step_T": minimum_field_step,
                "profile_points": len(profiles["nu"]),
                "profile_nu_step": float(profiles["nu"][1] - profiles["nu"][0]),
                "runtime_s": time.perf_counter() - start,
            }
            rows.append(row)

            if save_full_loops:
                store_loop(
                    variant_dir
                    / f"hysteresis_r{radius_tag(float(radius_nm))}.csv",
                    result,
                    radius_nm=float(radius_nm),
                    variant=variant,
                    gux_factor=gux_factor,
                )

            print(
                f"{variant}: R={radius_nm:4.1f} nm, "
                f"m_rem={row['remanence']:.8f}, "
                f"B_c={1.0e3 * row['coercive_field_T']:.5f} mT, "
                f"B_vu={row['reverse_vortex_uniform_field_T']:.8f} T, "
                f"{row['runtime_s']:.2f} s",
                flush=True,
            )

        summary = pd.DataFrame(rows)
        variant_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(variant_dir / "radius_observables.csv", index=False)
        summaries[variant] = summary

    return summaries


def plot_summary(
    summaries: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Plot four radius-dependent diagnostics of the crossover."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(8.0, 6.2),
        sharex=True,
        constrained_layout=True,
    )
    panels = (
        ("remanence", r"$\langle m_z\rangle_{\rm rem}$", 1.0),
        (
            "reverse_vortex_uniform_field_T",
            r"$B_{\mathrm{v}\to\mathrm{u}}^\uparrow$ (T)",
            1.0,
        ),
        (
            "remanent_axis_angle_deg",
            "remanent axis angle (deg)",
            1.0,
        ),
        ("coercive_field_T", r"$\mu_0H_c$ (mT)", 1.0e3),
    )

    for axis, (column, ylabel, scale), panel_label in zip(
        axes.flat,
        panels,
        ("(a)", "(b)", "(c)", "(d)"),
    ):
        for variant, table in summaries.items():
            axis.plot(
                table["radius_nm"],
                scale * table[column],
                color=MODEL_COLORS[variant],
                lw=1.5,
                label=MODEL_LABELS[variant],
            )
        axis.axvline(
            vortex_nucleation_radius(KU, MS, A) * 1.0e9,
            color="0.5",
            lw=0.8,
            ls=":",
        )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.text(0.02, 0.96, panel_label, transform=axis.transAxes, va="top")

    axes[1, 0].set_xlabel(r"$R$ (nm)")
    axes[1, 1].set_xlabel(r"$R$ (nm)")
    axes[0, 0].legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Return command-line overrides while preserving manuscript defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-min-nm", type=float, default=RADIUS_MIN_NM)
    parser.add_argument("--radius-max-nm", type=float, default=RADIUS_MAX_NM)
    parser.add_argument("--radius-step-nm", type=float, default=RADIUS_STEP_NM)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_VARIANTS),
        default=tuple(MODEL_VARIANTS),
    )
    parser.add_argument("--low-field-step-mT", type=float, default=0.1)
    parser.add_argument(
        "--profile-points",
        type=int,
        default=PROFILE_SETTINGS.n_nu,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--save-full-loops",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete reproducible uniform-to-vortex crossover sweep."""

    args = parse_args()
    if args.radius_step_nm <= 0.0:
        raise ValueError("radius-step-nm must be positive")
    if args.radius_max_nm < args.radius_min_nm:
        raise ValueError("radius-max-nm must not be smaller than radius-min-nm")
    if args.profile_points < 3:
        raise ValueError("profile-points must be at least three")

    radii_nm = np.arange(
        args.radius_min_nm,
        args.radius_max_nm + 0.5 * args.radius_step_nm,
        args.radius_step_nm,
    )
    fields = make_field_protocol(args.low_field_step_mT)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_settings = ProfileComputation(
        nu_min=PROFILE_SETTINGS.nu_min,
        nu_max=PROFILE_SETTINGS.nu_max,
        n_nu=args.profile_points,
        n_quad=PROFILE_SETTINGS.n_quad,
        chunk_size=PROFILE_SETTINGS.chunk_size,
        l_max_demag=PROFILE_SETTINGS.l_max_demag,
        n_mu_demag=PROFILE_SETTINGS.n_mu_demag,
    )
    print(
        f"Computing {profile_settings.n_nu} profile points and "
        f"{len(fields)} field points per loop.",
        flush=True,
    )
    profiles = compute_profiles(profile_settings, verbose=True)
    pd.DataFrame(profiles).to_csv(output_dir / "profiles.csv", index=False)
    field_table = pd.DataFrame(
        {
            "field_index": np.arange(len(fields)),
            "branch": branch_labels(fields),
            "B_T": fields,
        }
    )
    field_table["branch_index"] = field_table.groupby("branch").cumcount()
    field_table.to_csv(output_dir / "field_protocol.csv", index=False)
    store_run_metadata(
        output_dir / "run_metadata.json",
        radii_nm=radii_nm,
        variants=tuple(args.models),
        fields=fields,
        profile_settings=profile_settings,
        save_full_loops=args.save_full_loops,
    )

    summaries = run_radius_series(
        radii_nm=radii_nm,
        variants=tuple(args.models),
        fields=fields,
        profiles=profiles,
        output_dir=output_dir,
        save_full_loops=args.save_full_loops,
    )
    if args.plot:
        plot_summary(
            summaries,
            output_dir / "uniform_vortex_crossover.png",
        )

    print(f"Saved uniform-vortex crossover data to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
