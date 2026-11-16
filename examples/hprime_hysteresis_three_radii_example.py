"""Compute H' loops at R = 8, 10, 14 nm and plot one square field window.

Run ``pixi run hprime-hysteresis-three-radii`` to compute the three loops
sequentially and save CSV data, provenance, and a LaTeX/paperfig PNG.
``--plot-only`` redraws verified stored data without rerunning the solver.
The full field cycle is +1 T -> -1 T -> +1 T; the plotted window is
-0.5 T <= B_0 <= 0.5 T, with both the figure and its axes square.
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

THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paperfig as pf

import minimal_hyperbolic_micromagnetics as mhm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "hprime_hysteresis_three_radii_example_output"
RADII_NM = (8.0, 10.0, 14.0)
KU, MS, A = 4.8e4, 1.7e6, 1.0e-11
COLORS = ("#0072B2", "#D55E00", "#009E73")
PROFILE_SETTINGS = mhm.ProfileComputation(
    nu_min=0.0, nu_max=20.0, n_nu=2000,
    n_quad=360, l_max_demag=161, n_mu_demag=648,
)


def field_protocol() -> np.ndarray:
    """Use the 5262-point major-loop protocol of Supplemental Sec. I E."""
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
    return {
        "radii_nm": list(RADII_NM),
        "material": {"Ku_J_per_m3": KU, "Ms_A_per_m": MS, "A_J_per_m": A},
        "beta_deg": 0.0, "gux_factor": 1.0, "stoner_wohlfarth": False,
        "profiles": asdict(PROFILE_SETTINGS),
        "fields_T": field_protocol().tolist(),
    }


def compute_loops() -> None:
    """Compute one common profile table and all three radii sequentially."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = OUTPUT_DIR / "calculation_example_source.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    print("Computing the common H' profile table.", flush=True)
    profiles = mhm.compute_and_store_profiles(
        OUTPUT_DIR / "profiles.csv", PROFILE_SETTINGS,
        make_plot=False, verbose=True,
    ).to_dict(orient="list")
    settings = mhm.HysteresisSettings(fields=field_protocol(), stoner_wohlfarth=False)
    summaries = []
    for index, radius_nm in enumerate(RADII_NM, 1):
        print(f"[{index}/{len(RADII_NM)}] H': R={radius_nm:g} nm", flush=True)
        model = mhm.ModelParameters(
            Ku=KU, Ms=MS, A=A, R=radius_nm * 1.0e-9,
            beta_deg=0.0, gux_factor=1.0,
        )
        result = mhm.compute_and_store_hysteresis(
            loop_path(radius_nm), model, profiles, settings=settings,
            make_plot=False, print_runtime=True,
        )
        analysis = mhm.analyze_hysteresis(result)
        summaries.append({
            "R_nm": radius_nm, "field_points": len(result.B_T),
            "elapsed_s": result.elapsed_s,
            "descending": asdict(analysis.descending),
            "ascending": asdict(analysis.ascending),
            "max_abs_mz": float(np.max(np.abs(result.mz_avg))),
            "closure_error_mz": float(abs(result.mz_avg[-1] - result.mz_avg[0])),
        })
        print(f"Completed R={radius_nm:g} nm in {result.elapsed_s:.2f} s.", flush=True)

    summary_path = OUTPUT_DIR / "loop_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    files = [OUTPUT_DIR / "profiles.csv", snapshot, summary_path]
    for radius in RADII_NM:
        path = loop_path(radius)
        files.extend([path, path.with_name(f"{path.stem}_metadata.csv")])
    package_root = Path(mhm.__file__).resolve().parent
    manifest = {
        "configuration": configuration(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [Path(__file__).resolve(), *sorted(package_root.glob("*.py"))]
        },
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "threads": {key: os.environ[key] for key in THREAD_VARIABLES},
        "output_sha256": {path.name: sha256(path) for path in files},
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def load_loops() -> dict[float, pd.DataFrame]:
    """Check provenance, field ordering, finite values, and major-loop closure."""
    manifest = json.loads((OUTPUT_DIR / "run_manifest.json").read_text())
    if manifest["configuration"] != configuration():
        raise ValueError("Stored configuration differs; recompute before plotting.")
    for name, expected in manifest["output_sha256"].items():
        if sha256(OUTPUT_DIR / name) != expected:
            raise ValueError(f"Stored data checksum differs: {name}")
    loops = {}
    fields = field_protocol()
    half = len(fields) // 2
    for radius in RADII_NM:
        data = pd.read_csv(loop_path(radius))
        np.testing.assert_allclose(data.B_T, fields, rtol=0, atol=1.0e-14)
        if not (np.all(np.diff(data.B_T[:half]) < 0.0)
                and np.all(np.diff(data.B_T[half:]) > 0.0)):
            raise ValueError(f"Nonmonotonic field branch at R={radius:g} nm.")
        if not np.all(np.isfinite(data[["B_T", "mz_avg", "nu_min", "tau_min_rad", "energy"]])):
            raise ValueError(f"Nonfinite loop values at R={radius:g} nm.")
        if np.max(np.abs(data.mz_avg)) > 1.0 + 1.0e-12:
            raise ValueError(f"Magnetization exceeds saturation at R={radius:g} nm.")
        np.testing.assert_allclose(data.mz_avg.iloc[[0, half - 1, -1]], [1.0, -1.0, 1.0], atol=1.0e-12)
        np.testing.assert_allclose(data.nu_min.iloc[[0, half - 1, -1]], 0.0, atol=1.0e-12)
        loops[radius] = data
    return loops


def plot_loops(loops: dict[float, pd.DataFrame]) -> Path:
    """Make a square single-column paperfig figure with square plotting axes."""
    fig = pf.create_paper_figure(width_cm=8.6, height_cm=8.6, use_latex=True)
    curves = [
        {"x": loops[radius].B_T.to_numpy(), "y": loops[radius].mz_avg.to_numpy(),
         "color": color, "linewidth": 1.2}
        for radius, color in zip(RADII_NM, COLORS)
    ]
    pf.plotLinLin_panel_core(
        fig, curves, pos_cm=(1.55, 1.1), size_cm=(6.6, 6.6),
        xlabel=r"$B_0$ [$\mathrm{T}$]",
        ylabel=r"$\left\langle m_z\right\rangle_{V_R}$",
        xlim=(-0.5, 0.5), ylim=(-1.05, 1.05),
        xticks=[-0.5, -0.25, 0.0, 0.25, 0.5],
        yticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    for index, (radius, color) in enumerate(zip(RADII_NM, COLORS)):
        left = 1.05 + 2.55 * index
        pf.add_line_cm(fig, left, 8.15, left + 0.4, 8.15, color=color, linewidth=1.2)
        pf.add_label_cm(
            fig, rf"$R={radius:g}\,\mathrm{{nm}}$",
            x_cm=left + 0.55, y_cm=8.15, va="center",
        )
    output = OUTPUT_DIR / "hprime_hysteresis_three_radii.png"
    # Preserve the square canvas; tight bounding-box cropping would change it.
    fig.savefig(output, dpi=600)
    plt.close(fig)
    manifest = {
        "script_sha256": sha256(Path(__file__)),
        "run_manifest_sha256": sha256(OUTPUT_DIR / "run_manifest.json"),
        "png_sha256": sha256(output), "use_latex": True,
        "figure_size_cm": [8.6, 8.6], "axes_size_cm": [6.6, 6.6],
        "xlim_T": [-0.5, 0.5],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_DIR / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="redraw verified stored loops")
    args = parser.parse_args()
    if not args.plot_only:
        compute_loops()
    output = plot_loops(load_loops())
    print(f"Saved H' hysteresis plot to {output}", flush=True)


if __name__ == "__main__":
    main()
