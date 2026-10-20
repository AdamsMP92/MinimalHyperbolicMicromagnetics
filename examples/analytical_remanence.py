"""Plot the analytical, field-connected remanence of the revised H' model.

The material parameters match the spectral Galerkin paper. The remanent
branch is reached from positive saturation: uniform, axial vortex, then
transverse vortex. The middle panel uses the SM notation |<m_z>_V_R|_rem(R)
for the dimensionless longitudinal remanence, which vanishes after
reorientation even though the vortex retains a finite moment along its own
axis. The remanent orientation is labeled tau_rem and displayed in degrees.
No hysteresis sweep is required.

Run with ``pixi run analytical-remanence``. CSV data and a paperfig PNG are
written to ``analytical_remanence_example_output/`` in the repository root.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paperfig as pf

from minimal_hyperbolic_micromagnetics import (
    g_z_z,
    remanent_nucleation_radius,
    remanent_reorientation_radius,
    remanent_vortex_parameter,
)


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "analytical_remanence_example_output"
)

# Material parameters used in the main paper and Supplemental Material.
KU = 4.8e4  # uniaxial anisotropy [J/m^3]
MS = 1.7e6  # saturation magnetization [A/m]
A = 1.0e-11  # exchange stiffness [J/m]
R_MIN_NM, R_MAX_NM = 5.0, 20.0
N_RADII = 601

MODEL_COLOR = "#CC79A7"
NUCLEATION_STYLE = {"color": "0.45", "linestyle": ":", "linewidth": 0.9}
REORIENTATION_STYLE = {"color": "0.25", "linestyle": "--", "linewidth": 0.9}


def evaluate_remanence() -> tuple[pd.DataFrame, float, float]:
    """Evaluate the piecewise analytical solution, including its endpoints."""
    r_nuc = remanent_nucleation_radius(KU, MS, A)
    r_star = remanent_reorientation_radius(KU, MS, A)
    # Include the exact critical radii and a point immediately to the right
    # of R_* so a coarse radius grid does not obscure the profile jump.
    radius_m = np.unique(np.concatenate([
        np.linspace(R_MIN_NM, R_MAX_NM, N_RADII) * 1.0e-9,
        [r_nuc, r_star, r_star + 1.0e-15],
    ]))
    radius_nm = radius_m * 1.0e9
    nu_rem = remanent_vortex_parameter(radius_m, KU, MS, A)
    transverse = radius_m > r_star
    tau_rem = np.where(transverse, 0.5 * np.pi, 0.0)

    # Eq. (65): g_z^z(nu_rem) on the uniform/axial branch, zero longitudinal
    # projection on the transverse branch. Reuse nu_rem rather than invert
    # the radius relation a second time inside remanent_magnetization_ratio.
    mz_rem = np.where(transverse, 0.0, g_z_z(nu_rem))
    branch = np.where(
        radius_m <= r_nuc, "uniform",
        np.where(transverse, "transverse_vortex", "axial_vortex"),
    )
    data = pd.DataFrame({
        "R_nm": radius_nm,
        "nu_rem": nu_rem,
        "mz_rem_over_Ms": mz_rem,
        "Mz_rem_A_per_m": MS * mz_rem,
        "tau_rem_rad": tau_rem,
        "tau_rem_deg": np.rad2deg(tau_rem),
        "branch": branch,
    })
    return data, r_nuc * 1.0e9, r_star * 1.0e9


def plot_remanence(
    data: pd.DataFrame, r_nuc_nm: float, r_star_nm: float, output: Path,
) -> None:
    """Use paperfig's linear panels and LaTeX typography for all three plots."""
    fig = pf.create_paper_figure(
        width_cm=17.8, height_cm=6.1, use_latex=True,
    )
    radius = data["R_nm"].to_numpy()
    transverse = data["branch"].to_numpy() == "transverse_vortex"
    nu_max = float(data["nu_rem"].max())
    panels = (
        (
            "nu_rem", r"$\nu_{\mathrm{rem}}(R)$", (-0.15, 1.1 * nu_max),
            np.arange(np.floor(1.1 * nu_max) + 1), None,
        ),
        (
            "mz_rem_over_Ms",
            r"$\left|\left\langle m_z\right\rangle_{V_R}\right|_{\mathrm{rem}}(R)$",
            (-0.05, 1.05), np.linspace(0.0, 1.0, 5), None,
        ),
        (
            "tau_rem_deg", r"$\tau_{\mathrm{rem}}$ [$^\circ$]",
            (-5.0, 95.0), [0.0, 45.0, 90.0], None,
        ),
    )
    for index, (column, ylabel, ylim, yticks, yticklabels) in enumerate(panels):
        left_cm, bottom_cm = 1.25 + 5.8 * index, 1.1
        curves = [
            {"x": [r_nuc_nm, r_nuc_nm], "y": ylim, **NUCLEATION_STYLE},
            {"x": [r_star_nm, r_star_nm], "y": ylim, **REORIENTATION_STYLE},
        ]
        values = data[column].to_numpy()
        for mask in (~transverse, transverse):
            curves.append({
                "x": radius[mask], "y": values[mask],
                "color": MODEL_COLOR, "linewidth": 1.4,
            })

        # The API assigns the axial limiting state at R = R_*. Show that
        # endpoint filled, and the immediately following transverse state open.
        endpoints = (
            np.flatnonzero(~transverse)[-1], np.flatnonzero(transverse)[0],
        )
        for endpoint in endpoints:
            curves.append({
                "x": [radius[endpoint]], "y": [values[endpoint]],
                "marker": "o", "markersize": 3.5, "linestyle": "None",
                "color": MODEL_COLOR, "linewidth": 0.9,
            })

        ax = pf.plotLinLin_panel_core(
            fig, curves,
            pos_cm=(left_cm, bottom_cm), size_cm=(4.15, 4.0),
            xlabel=r"$R$ [$\mathrm{nm}$]", ylabel=ylabel,
            xlim=(R_MIN_NM, R_MAX_NM), ylim=ylim,
            xticks=[5.0, 10.0, 15.0, 20.0], yticks=yticks,
            yticklabels=yticklabels,
        )
        # The installed paperfig panel API does not expose markerfacecolor.
        # Customize only the open endpoint on the returned Matplotlib artist.
        ax.lines[-1].set_markerfacecolor("white")
        pf.add_label_cm(
            fig, f"({chr(ord('a') + index)})",
            x_cm=left_cm + 0.17, y_cm=bottom_cm + 3.52, va="top",
        )

    # Paperfig has no shared-legend helper; compose the two reference keys
    # with its centimeter-based line and label functions.
    for x_cm, label, style in (
        (
            4.0,
            rf"$R_{{\mathrm{{nuc}}}}^{{\mathrm{{rem}}}}={r_nuc_nm:.2f}\,\mathrm{{nm}}$",
            NUCLEATION_STYLE,
        ),
        (
            10.1,
            rf"$R_{{\ast}}^{{\mathrm{{rem}}}}={r_star_nm:.2f}\,\mathrm{{nm}}$",
            REORIENTATION_STYLE,
        ),
    ):
        pf.add_line_cm(fig, x_cm, 5.65, x_cm + 0.6, 5.65, **style)
        pf.add_label_cm(fig, label, x_cm=x_cm + 0.8, y_cm=5.65, va="center")
    fig.savefig(output, dpi=600)
    plt.close(fig)


def main() -> None:
    data, r_nuc_nm, r_star_nm = evaluate_remanence()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "analytical_remanence.csv"
    png_path = OUTPUT_DIR / "analytical_remanence.png"
    data.to_csv(csv_path, index=False)
    plot_remanence(data, r_nuc_nm, r_star_nm, png_path)
    print(f"A = {A:g} J/m, Ms = {MS:g} A/m, Ku = {KU:g} J/m^3")
    print(f"R_nuc^rem = {r_nuc_nm:.8f} nm; R_*^rem = {r_star_nm:.8f} nm")
    print(f"Saved analytical remanence data to {csv_path}")
    print(f"Saved three-panel plot to {png_path}")


if __name__ == "__main__":
    main()
