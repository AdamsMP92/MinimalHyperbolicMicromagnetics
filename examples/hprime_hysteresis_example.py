"""Compute five H' major loops with the protocol of Supplemental Sec. I E.

Run ``pixi run hprime-hysteresis`` to compute all radii sequentially and save
the loop CSVs, metadata, common profile table, provenance, and paperfig PNG.
Use ``pixi run hprime-hysteresis --plot-only`` to redraw verified stored data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

# Keep the small example sequential, including the numerical libraries.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
for variable in THREAD_VARIABLES:
    os.environ.setdefault(variable, "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paperfig as pf

import minimal_hyperbolic_micromagnetics as mhm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "hprime_hysteresis_example_output"
RADII_NM = (8.0, 9.3, 10.0, 12.0, 20.0)
KU, MS, A = 4.8e4, 1.7e6, 1.0e-11
COLORS = ("#0072B2", "#56B4E9", "#D55E00", "#009E73", "#CC79A7")
PROFILE_SETTINGS = mhm.ProfileComputation(
    nu_min=0.0, nu_max=20.0, n_nu=2000,
    n_quad=360, l_max_demag=161, n_mu_demag=648,
)


def field_protocol() -> np.ndarray:
    """Build the SM's common +1 T -> -1 T -> +1 T sequence, including B=0."""
    segments = (
        (1.0, 0.595, 0.005), (0.595, 0.071, 0.001),
        (0.071, -0.071, 0.0001), (-0.071, -0.595, 0.001),
        (-0.595, -1.0, 0.005),
    )
    parts = [
        np.linspace(start, stop, round(abs(stop - start) / step) + 1)
        for start, stop, step in segments
    ]
    descending = np.concatenate([parts[0], *(part[1:] for part in parts[1:])])
    return np.concatenate([descending, descending[::-1]])


def loop_path(radius_nm: float) -> Path:
    return OUTPUT_DIR / f"hprime_r{radius_nm:.1f}.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration() -> dict:
    """Record the complete physical, profile, and field-grid configuration."""
    return {
        "radii_nm": list(RADII_NM),
        "material": {"Ku_J_per_m3": KU, "Ms_A_per_m": MS, "A_J_per_m": A},
        "beta_deg": 0.0, "gux_factor": 1.0, "stoner_wohlfarth": False,
        "profiles": asdict(PROFILE_SETTINGS),
        "fields_T": field_protocol().tolist(),
    }


def compute_loops() -> None:
    """Compute the common profile table once and five complete loops in order."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_snapshot = OUTPUT_DIR / "calculation_example_source.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    print("Computing the common H' profile table.", flush=True)
    profiles = mhm.compute_and_store_profiles(
        OUTPUT_DIR / "profiles.csv", PROFILE_SETTINGS,
        make_plot=False, verbose=True,
    ).to_dict(orient="list")
    fields = field_protocol()
    settings = mhm.HysteresisSettings(fields=fields, stoner_wohlfarth=False)
    for index, radius_nm in enumerate(RADII_NM, 1):
        print(f"[{index}/{len(RADII_NM)}] H': R={radius_nm:g} nm, {len(fields)} field points", flush=True)
        model = mhm.ModelParameters(
            Ku=KU, Ms=MS, A=A, R=radius_nm * 1.0e-9,
            beta_deg=0.0, gux_factor=1.0,
        )
        result = mhm.compute_and_store_hysteresis(
            loop_path(radius_nm), model, profiles, settings=settings,
            make_plot=False, print_runtime=True,
        )
        print(f"Completed R={radius_nm:g} nm in {result.elapsed_s:.2f} s.", flush=True)

    package_root = Path(mhm.__file__).resolve().parent
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    files = [OUTPUT_DIR / "profiles.csv", source_snapshot]
    for radius in RADII_NM:
        path = loop_path(radius)
        files.extend([path, path.with_name(f"{path.stem}_metadata.csv")])
    manifest = {
        "configuration": configuration(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit, "git_status": git_status,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [Path(__file__).resolve(), *sorted(package_root.glob("*.py"))]
        },
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "threads": {key: os.environ.get(key) for key in THREAD_VARIABLES},
        "output_sha256": {path.name: sha256(path) for path in files},
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def load_loops() -> dict[float, pd.DataFrame]:
    """Validate the saved configuration and checksums before reading the loops."""
    manifest = json.loads((OUTPUT_DIR / "run_manifest.json").read_text())
    if manifest["configuration"] != configuration():
        raise ValueError("Stored H' configuration differs; recompute before plotting.")
    for name, expected in manifest["output_sha256"].items():
        if sha256(OUTPUT_DIR / name) != expected:
            raise ValueError(f"Stored data checksum differs: {name}")
    loops = {}
    for radius in RADII_NM:
        data = pd.read_csv(loop_path(radius))
        np.testing.assert_allclose(data.B_T, field_protocol(), rtol=0, atol=1e-14)
        if not np.all(np.isfinite(data[["B_T", "mz_avg", "nu_min", "tau_min_rad", "energy"]])):
            raise ValueError(f"Nonfinite loop values at R={radius:g} nm.")
        loops[radius] = data
    return loops


def plot_loops(loops: dict[float, pd.DataFrame]) -> Path:
    """Show the complete major loops and a central-field enlargement."""
    fig = pf.create_paper_figure(width_cm=17.8, height_cm=6.9, use_latex=True)
    for index, (scale, limit, ticks, unit) in enumerate((
        (1.0, 1.03, [-1.0, -0.5, 0.0, 0.5, 1.0], "T"),
        (1.0e3, 80.0, [-80.0, -40.0, 0.0, 40.0, 80.0], "mT"),
    )):
        curves = [
            {"x": loops[radius].B_T.to_numpy() * scale,
             "y": loops[radius].mz_avg.to_numpy(),
             "color": color, "linewidth": 1.1}
            for radius, color in zip(RADII_NM, COLORS)
        ]
        left = 1.6 + 8.45 * index
        pf.plotLinLin_panel_core(
            fig, curves, pos_cm=(left, 1.1), size_cm=(6.5, 4.7),
            xlabel=rf"$B_0$ [$\mathrm{{{unit}}}$]",
            ylabel=r"$\left\langle m_z\right\rangle_{V_R}$",
            xlim=(-limit, limit), ylim=(-1.05, 1.05),
            xticks=ticks, yticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        pf.add_label_cm(fig, f"({chr(ord('a') + index)})", x_cm=left + 0.2, y_cm=5.45, va="top")
    for index, (radius, color) in enumerate(zip(RADII_NM, COLORS)):
        left = 1.25 + 3.25 * index
        pf.add_line_cm(fig, left, 6.35, left + 0.45, 6.35, color=color, linewidth=1.1)
        pf.add_label_cm(
            fig, rf"$R={radius:g}\,\mathrm{{nm}}$",
            x_cm=left + 0.6, y_cm=6.35, va="center",
        )
    output = OUTPUT_DIR / "hprime_hysteresis.png"
    fig.savefig(output, dpi=600)
    plt.close(fig)
    plot_manifest = {
        "script_sha256": sha256(Path(__file__)),
        "run_manifest_sha256": sha256(OUTPUT_DIR / "run_manifest.json"),
        "png_sha256": sha256(output), "use_latex": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_DIR / "plot_manifest.json").write_text(
        json.dumps(plot_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="redraw stored, verified loops")
    args = parser.parse_args()
    if not args.plot_only:
        compute_loops()
    output = plot_loops(load_loops())
    print(f"Saved H' hysteresis plot to {output}", flush=True)


if __name__ == "__main__":
    main()
