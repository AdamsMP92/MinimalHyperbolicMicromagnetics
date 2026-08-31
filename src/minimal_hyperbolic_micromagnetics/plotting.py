"""Plot helpers for profile and hysteresis results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _as_dataframe(result_or_mapping):
    if isinstance(result_or_mapping, pd.DataFrame):
        return result_or_mapping
    if isinstance(result_or_mapping, (str, Path)):
        return pd.read_csv(result_or_mapping)
    if hasattr(result_or_mapping, "as_dict"):
        return pd.DataFrame(result_or_mapping.as_dict())
    return pd.DataFrame(result_or_mapping)


def plot_profiles(
    profiles,
    output_path,
    *,
    width_cm=17.8,
    height_cm=8.4,
    use_latex=False,
):
    """Plot maximum-normalized profile functions."""
    import paperfig as pf

    data = _as_dataframe(profiles)
    output_path = Path(output_path)
    nu = data["nu"].to_numpy()

    fig = pf.create_paper_figure(
        width_cm=width_cm,
        height_cm=height_cm,
        use_latex=use_latex,
    )

    def normalized(name):
        values = data[name].to_numpy()
        vmax = np.max(np.abs(values))
        return values / vmax if vmax > 0.0 else values

    curves = [
        {"x": nu, "y": normalized("g_ex"), "label": r"$g_{\rm ex}$", "color": "#D55E00", "linewidth": 0.9},
        {"x": nu, "y": normalized("g_u_x"), "label": r"$g_u^x$", "color": "#0072B2", "linewidth": 0.9},
        {"x": nu, "y": normalized("g_u_z"), "label": r"$g_u^z$", "color": "#009E73", "linewidth": 0.9},
        {"x": nu, "y": normalized("g_z_z"), "label": r"$g_z^z$", "color": "#CC79A7", "linewidth": 0.9},
        {"x": nu, "y": normalized("g_dem"), "label": r"$g_{\rm dem}$", "color": "#000000", "linewidth": 1.1},
    ]

    pf.plotLinLin_panel_core(
        fig,
        curves,
        pos_cm=(1.5, 1.25),
        size_cm=(15.0, 5.9),
        xlabel=r"$\nu$",
        ylabel=r"$g_i(\nu)/g_i^{\max}$",
        xlim=(float(nu.min()), float(nu.max())),
        ylim=(0.0, 1.05),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    return output_path


def plot_hysteresis(result_or_csv, output_path, *, scale=1.05, model_color="#1f77b4"):
    """Create the three-panel hysteresis plot used by the repository examples."""
    import paperfig as pf

    data = _as_dataframe(result_or_csv)
    output_path = Path(output_path)

    curves = [{
        "x": data["B_T"],
        "y": data["mz_avg"],
        "label": r"$\mathcal{H}^{\prime\prime}$",
        "plottype": "line",
        "linewidth": 1.0,
        "color": model_color,
        "marker": None,
        "linestyle": "-",
    }]
    curves_tau = [{
        "x": data["B_T"],
        "y": data["tau_min_rad"] * 180.0 / np.pi,
        "plottype": "line",
        "linewidth": 1.0,
        "color": model_color,
        "marker": None,
        "linestyle": "-",
    }]
    curves_nu = [{
        "x": data["B_T"],
        "y": data["nu_min"],
        "plottype": "line",
        "linewidth": 1.0,
        "color": model_color,
        "marker": None,
        "linestyle": "-",
    }]

    s = scale
    fig_width = 13.5 * s
    fig_height = 6.75 * s
    width = 4.5 * s
    dx = (5.9 + 0.85) * s
    dy = 3.1 * s
    x1, y1 = 1.3 * s, 1.0 * s
    x2, y2 = x1 + dx, y1 + dy

    fig = pf.create_paper_figure(width_cm=fig_width, height_cm=fig_height)
    pf.add_label_cm(fig, "(a)", x_cm=0.2 - 0.05 * s, y_cm=(5.30 + 0.85) * s)
    pf.add_label_cm(fig, "(b)", x_cm=(0.3 + 5.75 + 0.85 - 0.05) * s, y_cm=6.10 * s)
    pf.add_label_cm(fig, "(c)", x_cm=(0.3 + 5.75 + 0.85 - 0.05) * s, y_cm=3.00 * s)

    ax1 = pf.plotGeneral1D_panel_core(
        fig,
        curves,
        pos_cm=(x1, y1),
        size_cm=(width + 0.85 * s, width + 0.85 * s),
        xlabel=r"$B_0$ [T]",
        ylabel=r"$\langle m_z \rangle$",
        xlim=[-1.1, 1.1],
        ylim=[-1.1, 1.1],
        xticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    ax1.legend(
        frameon=True,
        loc="center left",
        bbox_to_anchor=(0.75, 0.10),
        handlelength=0.7,
        handletextpad=0.4,
        edgecolor="none",
    )

    pf.plotGeneral1D_panel_core(
        fig,
        curves_tau,
        pos_cm=(x2, y2),
        size_cm=(width + 0.85 * s, width / 2),
        xlabel=r"$B_0$ [T]",
        ylabel=r"$\tau$ [deg]",
        xlim=[-1.1, 1.1],
        ylim=[-400, 400],
        yticks=[-360, -270, -180, -90, 0, 90, 180, 270, 360],
    )

    pf.plotGeneral1D_panel_core(
        fig,
        curves_nu,
        pos_cm=(x2, y1),
        size_cm=(width + 0.85 * s, width / 2),
        xlabel=r"$B_0$ [T]",
        ylabel=r"$\nu$",
        xlim=[-1.1, 1.1],
        ylim=[-1.0, 15.0],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600)
    return output_path
