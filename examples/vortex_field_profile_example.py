"""Analytical counterpart of Fig. 1 in Adams and Michels, PRB 113, 224406.

Panel (a) samples the hyperbolic vortex on a 2 nm Cartesian grid inside a
sphere of radius 20 nm. Panel (b) contains only the analytical tanh and sech
curves at nu=4.443; no micromagnetic simulation data are used or required.
These are local magnetization profiles, distinct from the integrated energy
profiles plotted by ``profile_functions_example.py``.

Run ``pixi run vortex-field-profile``. The combined paperfig/LaTeX PNG, field
and profile CSVs, and metadata are saved in
``vortex_field_profile_example_output/``. Rendering is off screen.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paperfig as pf
from vtkmodules.vtkCommonCore import vtkMultiThreader, vtkSMPTools


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "vortex_field_profile_example_output"
RADIUS_NM = 20.0
GRID_SPACING_NM = 2.0
NU = 4.443
CHIRALITY = 1
PROFILE_SAMPLES = 1001
FIGURE_DPI = 600
RENDER_DPI = 400


def hyperbolic_vortex(points_nm: np.ndarray) -> np.ndarray:
    """Return body-frame unit vectors with vortex axis z' and positive core polarity.

    The axis is evaluated by its regular limit, avoiding the undefined
    azimuthal unit vector at rho=0.
    """
    x, y, _ = np.asarray(points_nm, dtype=float).T
    rho = np.hypot(x, y)
    psi = NU * rho / RADIUS_NM
    azimuthal_over_rho = np.divide(
        CHIRALITY * np.tanh(psi), rho,
        out=np.zeros_like(rho), where=rho > 0.0,
    )
    return np.column_stack((
        -y * azimuthal_over_rho,
        x * azimuthal_over_rho,
        1.0 / np.cosh(psi),
    ))


def sample_field() -> pd.DataFrame:
    """Evaluate the ansatz at body-frame Cartesian cell centers in the sphere."""
    cells = int(round(2.0 * RADIUS_NM / GRID_SPACING_NM))
    axis = -RADIUS_NM + (np.arange(cells) + 0.5) * GRID_SPACING_NM
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = points.reshape(-1, 3)
    points = points[np.sum(points**2, axis=1) <= RADIUS_NM**2]
    return pd.DataFrame(
        np.column_stack((points, hyperbolic_vortex(points))),
        columns=("x_nm", "y_nm", "z_nm", "mx", "my", "mz"),
    )


def sample_profiles() -> pd.DataFrame:
    """Evaluate the body-frame cylindrical components on 0 <= rho'/R <= 1."""
    rho_over_radius = np.linspace(0.0, 1.0, PROFILE_SAMPLES)
    psi = NU * rho_over_radius
    return pd.DataFrame({
        "rho_over_R": rho_over_radius,
        "m_rho": np.zeros_like(psi),
        "m_phi": CHIRALITY * np.tanh(psi),
        "m_z": 1.0 / np.cosh(psi),
    })


def plot_field_and_profiles(field: pd.DataFrame, profiles: pd.DataFrame,
                            output: Path) -> None:
    """Compose both panels, the colorbar, and labels with native paperfig helpers."""
    vtkSMPTools.Initialize(2)
    vtkMultiThreader.SetGlobalMaximumNumberOfThreads(2)
    plt.rcParams.update({"legend.fontsize": 8})
    fig = pf.create_paper_figure(width_cm=17.8, height_cm=8.2, use_latex=True)
    xyz = field[["x_nm", "y_nm", "z_nm"]].to_numpy().T
    magnetization = field[["mx", "my", "mz"]].to_numpy().T
    pf.quiver3_advanced_panel_fast(
        fig, *xyz, *magnetization, field.mz.to_numpy(),
        Cmin=0.0, Cmax=1.0, cmap="RdBu", arrow_scale=0.13,
        head_length=0.35, stick_radius=0.14, head_radius=0.24,
        centering=False, subsample=1, view="custom", dpi=RENDER_DPI,
        cam_pos=(3, -3, 2), focal_point=(0, 0, 0), up_direction=(0, 0, 1),
        axes_width_cm=8.2, axes_pos_x_cm=0.45, axes_pos_y_cm=0.0,
    )
    pf.add_colorbar_cm(
        fig, pos_cm=(0.60, 4.90), size_cm=(0.16, 1.65),
        vmin=0.0, vmax=1.0, cmap="RdBu", clabel=r"$m_{z'}$", ticks=[0.0, 1.0],
    )
    curves = [
        {
            "x": profiles.rho_over_R, "y": profiles.m_phi,
            "label": r"$m_{\phi'}=\tanh(\nu\rho'/R)$", "color": "black",
            "linewidth": 1.0, "linestyle": "-",
        },
        {
            "x": profiles.rho_over_R, "y": profiles.m_z,
            "label": r"$m_{z'}=\mathrm{sech}(\nu\rho'/R)$", "color": "black",
            "linewidth": 1.0, "linestyle": "--",
        },
    ]
    pf.plotLinLin_panel_core(
        fig, curves, pos_cm=(11.05, 1.20), size_cm=(6.2, 6.2),
        xlabel=r"$\rho'/R$", ylabel=r"$m_{\phi'},\,m_{z'}$", xlim=(-0.05, 1.05),
        ylim=(-0.05, 1.05), xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        yticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    pf.add_label_cm(fig, "(a)", x_cm=0.45, y_cm=7.65)
    pf.add_label_cm(fig, "(b)", x_cm=9.50, y_cm=7.65)
    pf.add_label_cm(fig, r"$x'$", x_cm=2.05, y_cm=1.00)
    pf.add_label_cm(fig, r"$z'$", x_cm=0.80, y_cm=3.35)
    fig.savefig(output, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    field = sample_field()
    profiles = sample_profiles()
    field_path = OUTPUT_DIR / "magnetization_field.csv"
    profiles_path = OUTPUT_DIR / "magnetization_profiles.csv"
    figure_path = OUTPUT_DIR / "vortex_field_profile.png"
    field.to_csv(field_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    plot_field_and_profiles(field, profiles, figure_path)
    paths = (Path(__file__), field_path, profiles_path, figure_path)
    metadata = {
        "reference": {"doi": "10.1103/8rkf-j2kn", "figure": 1},
        "nu": NU, "radius_nm": RADIUS_NM, "grid_spacing_nm": GRID_SPACING_NM,
        "chirality": CHIRALITY, "core_polarity": 1,
        "vortex_axis": [0.0, 0.0, 1.0], "field_samples": len(field),
        "coordinate_frame": "body-fixed", "profile_ylim": [-0.05, 1.05],
        "legend_fontsize_pt": 8,
        "profile_samples": len(profiles), "color_component": "mz",
        "camera_position": [3, -3, 2], "colormap": "RdBu",
        "use_latex": True, "off_screen": True,
        "figure_dpi": FIGURE_DPI, "vector_render_dpi": RENDER_DPI,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {name: version(name) for name in ("numpy", "paperfig", "pyvista")},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (OUTPUT_DIR / "vortex_field_profile_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"Saved {len(field)} analytical field samples to {field_path}")
    print(f"Saved analytical profiles to {profiles_path}")
    print(f"Saved paperfig/LaTeX figure to {figure_path}")


if __name__ == "__main__":
    main()
