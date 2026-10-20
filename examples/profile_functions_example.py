"""Plot the dimensionless profile functions in Supplemental Eqs. (8)--(12).

The paperfig/LaTeX figure contains one row of four panels: exchange, both
anisotropy profiles, Zeeman moment, and the full magnetostatic profile.
Values retain the scaling of the SM definitions. The geometric profiles
depend on nu alone; no material parameters or field sweep are required.

Run ``pixi run profile-functions`` to save the PNG, CSV, and metadata under
``profile_functions_example_output/`` in the repository root.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paperfig as pf

import minimal_hyperbolic_micromagnetics as mhm


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "profile_functions_example_output"
SETTINGS = mhm.ProfileComputation(
    nu_min=0.0, nu_max=20.0, n_nu=2000,
    n_quad=360, l_max_demag=161, n_mu_demag=648,
)
PROFILE_COLUMNS = ("nu", "g_ex", "g_u_z", "g_u_x", "g_z_z", "g_dem")


def evaluate_profiles() -> pd.DataFrame:
    """Evaluate the SM profiles with the quadratures used by the paper."""
    profiles = mhm.compute_profiles(SETTINGS, verbose=True)
    return pd.DataFrame({name: profiles[name] for name in PROFILE_COLUMNS})


def plot_profiles(data: pd.DataFrame, output: Path) -> None:
    """Create the four panels using paperfig's native linear-panel helper."""
    plt.rcParams.update({"font.size": 8})
    fig = pf.create_paper_figure(width_cm=17.8, height_cm=5.8, use_latex=True)
    exchange_top = float(np.ceil(data.g_ex.max()))
    panels = (
        (
            (("g_ex", "#D55E00", None),), r"$g_{\mathrm{ex}}(\nu)$",
            (-0.03 * exchange_top, exchange_top),
            np.arange(0.0, exchange_top + 0.1, 5.0), None,
        ),
        (
            (
                ("g_u_z", "#009E73", r"$g_{\mathrm{u}}^z$"),
                ("g_u_x", "#0072B2", r"$g_{\mathrm{u}}^x$"),
            ),
            r"$g_{\mathrm{u}}^\alpha(\nu)$", (-0.03, 1.03),
            [0.0, 0.5, 1.0], None,
        ),
        (
            (("g_z_z", "#CC79A7", None),), r"$g_z^z(\nu)$",
            (-0.03, 1.03), [0.0, 0.5, 1.0], None,
        ),
        (
            (("g_dem", "black", None),), r"$g_{\mathrm{dem}}(\nu)$",
            (0.32, 0.505), [1.0 / 3.0, 0.4, 0.5],
            [r"$1/3$", r"$0.4$", r"$0.5$"],
        ),
    )
    for index, (profiles, ylabel, ylim, yticks, yticklabels) in enumerate(panels):
        curves = []
        for column, color, label in profiles:
            curve = {
                "x": data.nu.to_numpy(), "y": data[column].to_numpy(),
                "color": color, "linewidth": 1.1,
            }
            if label is not None:
                curve["label"] = label
            curves.append(curve)
        left = 1.2 + 4.4 * index
        pf.plotLinLin_panel_core(
            fig, curves, pos_cm=(left, 1.1), size_cm=(2.9, 3.7),
            xlabel=r"$\nu$", ylabel=ylabel, xlim=(0.0, 20.0), ylim=ylim,
            xticks=[0.0, 10.0, 20.0], yticks=yticks, yticklabels=yticklabels,
        )
        pf.add_label_cm(
            fig, f"({chr(ord('a') + index)})", x_cm=left, y_cm=5.05, va="bottom",
        )
    fig.savefig(output, dpi=600)
    plt.close(fig)


def main() -> None:
    data = evaluate_profiles()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "profile_functions.csv"
    png_path = OUTPUT_DIR / "profile_functions.png"
    data.to_csv(csv_path, index=False)
    plot_profiles(data, png_path)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    profile_source = Path(mhm.__file__).resolve().parent / "profiles.py"
    metadata = {
        "settings": asdict(SETTINGS),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(Path(__file__)),
        "profile_source_sha256": digest(profile_source),
        "csv_sha256": digest(csv_path), "png_sha256": digest(png_path),
        "numpy": np.__version__, "pandas": pd.__version__, "use_latex": True,
    }
    (OUTPUT_DIR / "profile_functions_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"Saved profile data to {csv_path}")
    print(f"Saved four-panel plot to {png_path}")


if __name__ == "__main__":
    main()
