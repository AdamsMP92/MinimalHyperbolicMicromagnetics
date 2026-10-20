"""Plot the analytical radius-dependent coercivity of the revised H' model.

The two panels show nu_c(R) and the positive coercive-field magnitude
mu0 H_c(R), following the Supplemental Material's coercive-state solution.
Below R_nuc^c the state is uniform. The axial coercive spinodal determines
the intermediate branch. Above R_*^rem the coercivity vanishes, while the
limiting profile is the transverse remanent vortex. No field sweep is needed.

Run with ``pixi run analytical-coercivity``. CSV data and a paperfig PNG are
written to ``analytical_coercivity_example_output/`` in the repository root.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paperfig as pf

from minimal_hyperbolic_micromagnetics import (
    coercive_field_from_vortex_parameter,
    coercive_nucleation_radius,
    coercive_vortex_parameter,
    remanent_reorientation_radius,
)


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "analytical_coercivity_example_output"
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


def evaluate_coercivity() -> tuple[pd.DataFrame, float, float]:
    """Evaluate the SM's piecewise analytical coercive-state solution."""
    # R_nuc^c is the onset of coercivity reduction, distinct from R_nuc^rem.
    r_nuc_c = coercive_nucleation_radius(MS, A)
    r_star = remanent_reorientation_radius(KU, MS, A)
    radius_m = np.unique(np.concatenate([
        np.linspace(R_MIN_NM, R_MAX_NM, N_RADII) * 1.0e-9,
        [r_nuc_c, r_star, r_star + 1.0e-15],
    ]))
    nu_c = coercive_vortex_parameter(radius_m, KU, MS, A)
    uniform = radius_m <= r_nuc_c
    axial = (radius_m > r_nuc_c) & (radius_m < r_star)
    transverse = radius_m > r_star

    # Reuse nu_c in the field parametrization instead of inverting the radius
    # relation again inside hyperbolic_vortex_coercive_field. At R = R_*^rem
    # the field is exactly zero and nu_c is the axial limiting value nu_*.
    field_t = np.zeros_like(radius_m)
    field_t[uniform] = 2.0 * KU / MS
    field_t[axial] = coercive_field_from_vortex_parameter(nu_c[axial], KU, MS)
    data = pd.DataFrame({
        "R_nm": radius_m * 1.0e9,
        "nu_c": nu_c,
        "mu0_Hc_T": field_t,
        "mu0_Hc_mT": field_t * 1.0e3,
        "branch": np.where(
            uniform, "uniform",
            np.where(transverse, "transverse_vortex", "axial_vortex"),
        ),
    })
    return data, r_nuc_c * 1.0e9, r_star * 1.0e9


def plot_coercivity(
    data: pd.DataFrame, r_nuc_c_nm: float, r_star_nm: float, output: Path,
) -> None:
    """Create two adjacent paperfig linear panels with LaTeX typography."""
    fig = pf.create_paper_figure(
        width_cm=12.0, height_cm=6.1, use_latex=True,
    )
    radius = data["R_nm"].to_numpy()
    transverse = data["branch"].to_numpy() == "transverse_vortex"
    nu_max = float(data["nu_c"].max())
    field_max = float(data["mu0_Hc_mT"].max())
    panels = (
        (
            "nu_c", r"$\nu_{\mathrm{c}}(R)$", (-0.15, 1.1 * nu_max),
            np.arange(np.floor(1.1 * nu_max) + 1),
        ),
        (
            "mu0_Hc_mT", r"$\mu_0H_{\mathrm{c}}(R)$ [$\mathrm{mT}$]",
            (-0.05 * field_max, 1.1 * field_max), np.arange(0.0, 61.0, 20.0),
        ),
    )
    for index, (column, ylabel, ylim, yticks) in enumerate(panels):
        left_cm, bottom_cm = 1.25 + 5.8 * index, 1.1
        curves = [
            {"x": [r_nuc_c_nm, r_nuc_c_nm], "y": ylim, **NUCLEATION_STYLE},
            {"x": [r_star_nm, r_star_nm], "y": ylim, **REORIENTATION_STYLE},
        ]
        values = data[column].to_numpy()
        # nu_c jumps at reorientation; the coercive field is continuous there.
        masks = (~transverse, transverse) if index == 0 else (np.ones_like(transverse),)
        for mask in masks:
            curves.append({
                "x": radius[mask], "y": values[mask],
                "color": MODEL_COLOR, "linewidth": 1.4,
            })
        if index == 0:
            for endpoint in (
                np.flatnonzero(~transverse)[-1], np.flatnonzero(transverse)[0],
            ):
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
        )
        if index == 0:
            # paperfig's panel API does not expose markerfacecolor.
            ax.lines[-1].set_markerfacecolor("white")
        pf.add_label_cm(
            fig, f"({chr(ord('a') + index)})",
            x_cm=left_cm + 0.17, y_cm=bottom_cm + 3.52, va="top",
        )

    for x_cm, label, style in (
        (
            1.5,
            rf"$R_{{\mathrm{{nuc}}}}^{{\mathrm{{c}}}}={r_nuc_c_nm:.2f}\,\mathrm{{nm}}$",
            NUCLEATION_STYLE,
        ),
        (
            7.0,
            rf"$R_{{\ast}}^{{\mathrm{{rem}}}}={r_star_nm:.2f}\,\mathrm{{nm}}$",
            REORIENTATION_STYLE,
        ),
    ):
        pf.add_line_cm(fig, x_cm, 5.65, x_cm + 0.6, 5.65, **style)
        pf.add_label_cm(fig, label, x_cm=x_cm + 0.8, y_cm=5.65, va="center")
    fig.savefig(output, dpi=600)
    plt.close(fig)


def main() -> None:
    data, r_nuc_c_nm, r_star_nm = evaluate_coercivity()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "analytical_coercivity.csv"
    png_path = OUTPUT_DIR / "analytical_coercivity.png"
    data.to_csv(csv_path, index=False)
    plot_coercivity(data, r_nuc_c_nm, r_star_nm, png_path)
    print(f"A = {A:g} J/m, Ms = {MS:g} A/m, Ku = {KU:g} J/m^3")
    print(f"R_nuc^c = {r_nuc_c_nm:.8f} nm; R_*^rem = {r_star_nm:.8f} nm")
    print(f"Uniform coercive-field magnitude = {2.0 * KU / MS * 1.0e3:.8f} mT")
    print(f"Saved analytical coercivity data to {csv_path}")
    print(f"Saved two-panel plot to {png_path}")


if __name__ == "__main__":
    main()
