"""Browser-native Shiny interface for MinimalHyperbolicMicromagnetics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

from minimal_hyperbolic_micromagnetics import split_field_branches
from web_support import (
    WEB_PROFILE_SETTINGS,
    ProfileBundle,
    WebCalculation,
    WebInputs,
    build_profile_csv,
    build_result_archive,
    calculate_hysteresis,
    effective_n_mu,
    get_profile_bundle,
    profile_settings_from_web,
)


APP_DIR = Path(__file__).resolve().parent

PRESETS = {
    "reference20": {
        "Ku_kJ_m3": 48.0,
        "Ms_MA_m": 1.7,
        "A_pJ_m": 10.0,
        "radius_nm": 20.0,
        "beta_deg": 0.0,
        "Bmax_T": 1.0,
    },
    "reference12": {
        "Ku_kJ_m3": 48.0,
        "Ms_MA_m": 1.7,
        "A_pJ_m": 10.0,
        "radius_nm": 12.0,
        "beta_deg": 0.0,
        "Bmax_T": 1.0,
    },
    "reference8": {
        "Ku_kJ_m3": 48.0,
        "Ms_MA_m": 1.7,
        "A_pJ_m": 10.0,
        "radius_nm": 8.0,
        "beta_deg": 0.0,
        "Bmax_T": 1.0,
    },
}


def _field(
    input_id: str,
    label: str,
    value: float,
    *,
    minimum: float,
    maximum: float | None = None,
    step: float,
    note: str,
) -> ui.Tag:
    return ui.div(
        ui.input_numeric(
            input_id,
            label,
            value,
            min=minimum,
            max=maximum,
            step=step,
            update_on="change",
        ),
        ui.tags.small(note, class_="field-note"),
        class_="field-block",
    )


def _section_heading(index: str, label: str) -> ui.Tag:
    return ui.div(
        ui.tags.span(index, class_="section-index"),
        ui.tags.span(label, class_="section-label"),
        class_="input-section-title",
    )


def _metric_card(
    label: str,
    value: str,
    unit: str,
    detail: str,
    *,
    accent: str = "teal",
) -> ui.Tag:
    return ui.div(
        ui.div(
            ui.tags.span(label, class_="metric-label"),
            ui.tags.span(class_=f"metric-dot metric-dot-{accent}"),
            class_="metric-heading",
        ),
        ui.div(
            ui.tags.span(value, class_="metric-value"),
            ui.tags.span(unit, class_="metric-unit"),
            class_="metric-number",
        ),
        ui.tags.p(detail, class_="metric-detail"),
        class_=f"metric-card metric-card-{accent}",
    )


def _status_item(label: str, value: str) -> ui.Tag:
    return ui.div(
        ui.tags.dt(label),
        ui.tags.dd(value),
        class_="status-item",
    )


def _number(value: float, digits: int = 5) -> str:
    if not np.isfinite(value):
        return "—"
    if abs(value) < 1.0e-12:
        value = 0.0
    return f"{value:.{digits}g}"


def _millitesla(value: float) -> str:
    return _number(1.0e3 * value)


def _profile_placeholder_figure():
    figure, axis = plt.subplots(figsize=(8.0, 3.2))
    figure.patch.set_alpha(0.0)
    axis.set_facecolor("#fbfcfd")
    axis.text(
        0.5,
        0.56,
        "Prepare the universal profile table",
        ha="center",
        va="center",
        color="#344054",
        fontsize=13,
        fontweight="semibold",
        transform=axis.transAxes,
    )
    axis.text(
        0.5,
        0.42,
        "Choose the grid in the left panel, then select Compute profiles.",
        ha="center",
        va="center",
        color="#667085",
        fontsize=9,
        transform=axis.transAxes,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    return figure


def _profile_figure(profiles: ProfileBundle):
    data = profiles.values
    nu = data["nu"]
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True)
    figure.patch.set_alpha(0.0)
    panels = (
        (
            axes[0, 0],
            (("g_ex", "#d55e00", None),),
            r"Exchange $g_{\mathrm{ex}}(\nu)$",
        ),
        (
            axes[0, 1],
            (
                ("g_u_z", "#009e73", r"$g_{\mathrm{u}}^z$"),
                ("g_u_x", "#0072b2", r"$g_{\mathrm{u}}^x$"),
            ),
            "Anisotropy profiles",
        ),
        (
            axes[1, 0],
            (("g_z_z", "#cc79a7", None),),
            r"Zeeman moment $g_z^z(\nu)$",
        ),
        (
            axes[1, 1],
            (("g_dem", "#101828", None),),
            r"Demagnetizing $g_{\mathrm{dem}}(\nu)$",
        ),
    )
    for axis, curves, title in panels:
        axis.set_facecolor("#fbfcfd")
        for column, color, label in curves:
            axis.plot(
                nu,
                data[column],
                color=color,
                linewidth=1.8,
                label=label,
            )
        axis.set_title(title, loc="left", fontsize=10, fontweight="semibold")
        axis.grid(color="#e7eaee", linewidth=0.7)
        axis.set_xlim(profiles.settings.nu_min, profiles.settings.nu_max)
        if any(label is not None for _, _, label in curves):
            axis.legend(frameon=False, fontsize=8)
    axes[1, 0].set_xlabel(r"Vortex coordinate $\nu$")
    axes[1, 1].set_xlabel(r"Vortex coordinate $\nu$")
    figure.suptitle(
        "Universal hyperbolic-vortex profile functions",
        x=0.08,
        y=1.0,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    figure.tight_layout()
    return figure


def _hysteresis_figure(calculation: WebCalculation):
    figure, axis = plt.subplots(figsize=(7.6, 5.0))
    figure.patch.set_alpha(0.0)
    axis.set_facecolor("#fbfcfd")
    colors = {"descending": "#b42318", "ascending": "#136f7a"}
    labels_seen: set[str] = set()
    for branch in split_field_branches(calculation.result):
        label = branch.name if branch.name not in labels_seen else None
        labels_seen.add(branch.name)
        axis.plot(
            branch.B_T,
            branch.mz_avg,
            color=colors[branch.name],
            linewidth=2.0,
            label=label,
        )
    axis.axhline(0.0, color="#98a2b3", linewidth=0.8, zorder=0)
    axis.axvline(0.0, color="#98a2b3", linewidth=0.8, zorder=0)
    axis.set_xlim(-calculation.inputs.Bmax_T, calculation.inputs.Bmax_T)
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlabel(r"Applied field $B_{\mathrm{ext}}$ [T]")
    axis.set_ylabel(r"Projected magnetization $\langle m_z\rangle$")
    axis.set_title(
        f"{calculation.model_label} · R = {calculation.inputs.R_m * 1e9:g} nm · "
        f"β = {calculation.inputs.beta_deg:g}°",
        loc="left",
        fontweight="semibold",
    )
    axis.grid(color="#e7eaee", linewidth=0.7)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    return figure


def _state_figure(calculation: WebCalculation):
    result = calculation.result
    figure, axes = plt.subplots(3, 1, figsize=(7.6, 7.3), sharex=True)
    figure.patch.set_alpha(0.0)
    colors = {"descending": "#b42318", "ascending": "#136f7a"}
    branches = split_field_branches(result)
    for axis in axes:
        axis.set_facecolor("#fbfcfd")
        axis.grid(color="#e7eaee", linewidth=0.7)
        axis.axvline(0.0, color="#98a2b3", linewidth=0.7, zorder=0)
    for branch in branches:
        color = colors[branch.name]
        axes[0].plot(branch.B_T, branch.nu_min, color=color, linewidth=1.7)
        axes[1].plot(
            branch.B_T,
            np.rad2deg(branch.tau_rad),
            color=color,
            linewidth=1.7,
        )
        if result.uniform_vortex_curvature is not None:
            curvature = np.asarray(result.uniform_vortex_curvature)[branch.indices]
            axes[2].plot(branch.B_T, curvature, color=color, linewidth=1.7)
    axes[0].set_ylabel(r"Vortex $\nu$")
    axes[1].set_ylabel(r"Angle $\tau$ [deg]")
    axes[2].set_ylabel(r"Uniform $\mathcal{H}_{\nu\nu}$")
    axes[2].set_xlabel(r"Applied field $B_{\mathrm{ext}}$ [T]")
    axes[2].axhline(0.0, color="#101828", linewidth=0.8, zorder=0)
    figure.tight_layout()
    return figure


input_panel = ui.div(
    ui.div(
        ui.tags.span("MODEL INPUT", class_="eyebrow"),
        ui.h2("Configure the particle"),
        ui.tags.p(
            "Start from the article parameters or enter a custom spherical particle.",
            class_="panel-intro",
        ),
        class_="panel-heading",
    ),
    ui.div(
        ui.input_select(
            "preset",
            "Load a reference setup",
            {
                "reference20": "Article parameters · R = 20 nm",
                "reference12": "Article parameters · R = 12 nm",
                "reference8": "Article parameters · R = 8 nm",
            },
            selected="reference20",
        ),
        ui.tags.small(
            "Presets populate editable SI-derived fields below.",
            class_="field-note",
        ),
        class_="preset-block",
    ),
    _section_heading("01", "Profile table"),
    ui.div(
        _field(
            "nu_max",
            "Maximum vortex coordinate νmax",
            WEB_PROFILE_SETTINGS.nu_max,
            minimum=0.1,
            maximum=100.0,
            step=1.0,
            note="Upper end of the universal profile table",
        ),
        _field(
            "n_nu",
            "Profile points Nν",
            WEB_PROFILE_SETTINGS.n_nu,
            minimum=32,
            maximum=20_000,
            step=100,
            note="Uniform samples between 0 and νmax",
        ),
        class_="input-pair",
    ),
    ui.tags.details(
        ui.tags.summary("Advanced numerical controls"),
        ui.div(
            ui.div(
                _field(
                    "n_quad",
                    "Local quadrature Nquad",
                    WEB_PROFILE_SETTINGS.n_quad,
                    minimum=16,
                    maximum=2_048,
                    step=16,
                    note="Gauss–Legendre points for local profiles",
                ),
                _field(
                    "l_max_demag",
                    "Demagnetizing cutoff lmax",
                    WEB_PROFILE_SETTINGS.l_max_demag,
                    minimum=1,
                    maximum=501,
                    step=2,
                    note="Odd spherical-harmonic degree",
                ),
                class_="input-pair",
            ),
            _field(
                "n_mu_demag",
                "Demagnetizing quadrature Nμ",
                0,
                minimum=0,
                maximum=4_096,
                step=16,
                note="0 selects max(360, 4(lmax + 1)) automatically",
            ),
            class_="advanced-settings-body",
        ),
        class_="advanced-settings",
    ),
    ui.input_action_button(
        "compute_profiles",
        "Compute profiles",
        class_="btn-profile",
    ),
    ui.output_ui("profile_preparation_status"),
    _section_heading("02", "Material"),
    ui.div(
        _field(
            "Ku_kJ_m3",
            "Uniaxial anisotropy Ku [kJ/m³]",
            48.0,
            minimum=0.0,
            step=1.0,
            note="Converted to J/m³ internally",
        ),
        _field(
            "Ms_MA_m",
            "Saturation magnetization Ms [MA/m]",
            1.7,
            minimum=1.0e-6,
            step=0.05,
            note="Positive saturation magnitude",
        ),
        class_="input-pair",
    ),
    _field(
        "A_pJ_m",
        "Exchange stiffness A [pJ/m]",
        10.0,
        minimum=1.0e-6,
        step=0.5,
        note="Converted to J/m internally",
    ),
    _section_heading("03", "Geometry & orientation"),
    ui.div(
        _field(
            "radius_nm",
            "Particle radius R [nm]",
            20.0,
            minimum=0.01,
            step=0.5,
            note="Spherical-particle radius",
        ),
        _field(
            "beta_deg",
            "Easy-axis angle β [deg]",
            0.0,
            minimum=0.0,
            maximum=90.0,
            step=1.0,
            note="Relative to the field axis; β ≠ 0 uses the slower general solver",
        ),
        class_="input-pair",
    ),
    _section_heading("04", "Model"),
    ui.input_checkbox(
        "use_h_double_prime",
        "Use H″ approximation (omit the transverse anisotropy profile)",
        False,
    ),
    ui.input_checkbox(
        "stoner_wohlfarth",
        "Restrict to the uniform Stoner–Wohlfarth state",
        False,
    ),
    ui.tags.small(
        "H′ is the default. H″ uses the existing gux_factor = 0 switch.",
        class_="field-note checkbox-note",
    ),
    _section_heading("05", "Field protocol"),
    _field(
        "Bmax_T",
        "Maximum applied field Bmax [T]",
        1.0,
        minimum=0.001,
        step=0.05,
        note="Sweep: +Bmax → −Bmax → +Bmax",
    ),
    ui.input_select(
        "n_half",
        "Field resolution",
        {
            "160": "Fast · 320 total points",
            "250": "Standard · 500 total points",
            "500": "Fine · 1000 total points",
        },
        selected="250",
    ),
    ui.input_action_button(
        "calculate",
        "Calculate hysteresis",
        class_="btn-calculate",
        disabled=True,
    ),
    ui.div(
        ui.tags.span(class_="privacy-pulse"),
        ui.tags.span("Profiles and hysteresis are calculated in this browser tab."),
        class_="privacy-note",
    ),
    class_="input-panel",
)


simulator_panel = ui.div(
    ui.div(input_panel, class_="simulator-sidebar"),
    ui.tags.main(
        ui.div(
            ui.div(
                ui.div(
                    ui.tags.span("STEP 1 · UNIVERSAL INPUT", class_="eyebrow"),
                    ui.h3("Prepared profile functions"),
                    class_="card-title-group",
                ),
                ui.output_ui("profile_chip"),
                class_="visual-card-header",
            ),
            ui.output_plot("profile_plot", height="620px"),
            ui.output_ui("profile_card_footer"),
            class_="visual-card profile-card",
        ),
        ui.output_ui("calculation_message"),
        ui.output_ui("summary_cards"),
        ui.div(
            ui.div(
                ui.div(
                    ui.div(
                        ui.tags.span("MAJOR LOOP", class_="eyebrow"),
                        ui.h3("Quasistatic hysteresis"),
                        class_="card-title-group",
                    ),
                    ui.output_ui("model_chip"),
                    class_="visual-card-header",
                ),
                ui.output_plot("hysteresis_plot", height="510px"),
                class_="visual-card hysteresis-card",
            ),
            class_="visual-grid",
        ),
        ui.div(
            ui.div(
                ui.tags.span("STATE PATH", class_="eyebrow"),
                ui.h3("Vortex, orientation, and stability"),
                ui.output_plot("state_plot", height="710px"),
                class_="visual-card state-card",
            ),
            class_="visual-grid",
        ),
        ui.div(
            ui.div(
                ui.tags.span("NUMERICS", class_="eyebrow"),
                ui.h3("Calculation diagnostics"),
                ui.output_ui("diagnostics"),
                class_="diagnostics-card",
            ),
            ui.div(
                ui.tags.span("MODEL BOUNDARY", class_="eyebrow"),
                ui.h3("What this result represents"),
                ui.tags.p(
                    "A reduced two-coordinate hyperbolic-vortex model for a "
                    "spherical, noninteracting particle under a quasistatic field path."
                ),
                ui.tags.p(
                    "The horizontal axis is the signed external field quantity "
                    "Bext used by the package, measured in tesla and commonly read "
                    "as μ₀Hext in this protocol."
                ),
                ui.download_button(
                    "download_result",
                    "Download data & metadata",
                    class_="download-primary",
                ),
                class_="assumption-card",
            ),
            class_="support-grid",
        ),
        class_="simulator-results",
    ),
    class_="simulator-grid",
)


documentation_panel = ui.tags.article(
    ui.div(
        ui.tags.span("MODEL & METHOD", class_="eyebrow"),
        ui.h2("A field-following reduced micromagnetic model"),
        ui.tags.p(
            "The interface calls the same tested Python functions as the scripts and "
            "notebooks in the repository. No separate web approximation is introduced.",
            class_="docs-lead",
        ),
        class_="docs-intro",
    ),
    ui.div(
        ui.tags.section(
            ui.tags.span("01", class_="docs-number"),
            ui.h3("Universal profiles"),
            ui.tags.p(
                "The exchange, anisotropy, Zeeman-moment, and demagnetizing "
                "profiles depend only on ν. Choose νmax and Nν, optionally refine "
                "the quadratures, and prepare the table explicitly before running "
                "a material-dependent hysteresis calculation."
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("02", class_="docs-number"),
            ui.h3("H′ by default"),
            ui.tags.p(
                "H′ retains both longitudinal and transverse anisotropy profiles. "
                "The H″ switch omits the transverse term through the existing "
                "gux_factor multiplier; no package API is changed."
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("03", class_="docs-number"),
            ui.h3("Metastable continuation"),
            ui.tags.p(
                "At each field value the solver follows a nearby local minimum in "
                "(ν, τ), preserving the field history that produces the hysteresis loop."
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("04", class_="docs-number"),
            ui.h3("Interpreting presets"),
            ui.tags.p(
                "The included presets reproduce the article parameter set at selected "
                "radii. They are starting points, not a general materials database."
            ),
            class_="docs-section",
        ),
        class_="docs-grid",
    ),
    ui.div(
        ui.tags.span("OPEN SOURCE", class_="eyebrow"),
        ui.h3("Inspect the equations and implementation"),
        ui.tags.p(
            "The derivations, continuation protocol, examples, and physics checks "
            "are maintained with the calculation core."
        ),
        ui.tags.a(
            "View the GitHub repository",
            href="https://github.com/AdamsMP92/MinimalHyperbolicMicromagnetics",
            target="_blank",
            rel="noopener noreferrer",
            class_="docs-link",
        ),
        class_="source-card docs-source-card",
    ),
    class_="documentation-panel",
)


app_ui = ui.page_bootstrap(
    ui.head_content(
        ui.tags.meta(
            name="description",
            content=(
                "Interactive reduced micromagnetic hysteresis calculator for "
                "spherical magnetic nanoparticles."
            ),
        ),
        ui.tags.link(rel="icon", type="image/svg+xml", href="hysteresis-mark.svg"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(
            rel="preconnect",
            href="https://fonts.gstatic.com",
            crossorigin="anonymous",
        ),
        ui.tags.link(
            rel="stylesheet",
            href=(
                "https://fonts.googleapis.com/css2?"
                "family=DM+Sans:wght@400;500;600;700&"
                "family=Newsreader:opsz,wght@6..72,600&display=swap"
            ),
        ),
    ),
    ui.include_css(APP_DIR / "www" / "styles.css"),
    ui.div(
        ui.div(
            ui.tags.header(
                ui.div(
                    ui.div(
                        ui.img(
                            src="hysteresis-mark.svg",
                            alt="Stylized magnetic hysteresis loop",
                            class_="hero-mark",
                        ),
                        ui.div(
                            ui.tags.span(
                                "MICHAEL P. ADAMS · SCIENTIFIC SOFTWARE",
                                class_="hero-kicker",
                            ),
                            ui.h1("Minimal Hyperbolic Micromagnetics"),
                            ui.tags.p(
                                "Explore vortex nucleation, reversal, and quasistatic "
                                "hysteresis in spherical magnetic nanoparticles."
                            ),
                        ),
                        class_="hero-copy",
                    ),
                    ui.div(
                        ui.tags.span(class_="runtime-dot"),
                        ui.div(
                            ui.tags.strong("Browser-native Python"),
                            ui.tags.small("Shinylive · profiles stay local"),
                        ),
                        class_="runtime-badge",
                    ),
                    class_="hero-inner",
                ),
                class_="app-hero",
            ),
            ui.div(
                ui.navset_card_underline(
                    ui.nav_panel("Simulator", simulator_panel, value="simulator"),
                    ui.nav_panel(
                        "Model & usage",
                        documentation_panel,
                        value="documentation",
                    ),
                    id="main_navigation",
                    selected="simulator",
                ),
                class_="main-nav-wrap",
            ),
            ui.tags.footer(
                ui.tags.span("Minimal Hyperbolic Micromagnetics"),
                ui.tags.span("·"),
                ui.tags.span("Michael P. Adams"),
                ui.tags.span("·"),
                ui.tags.a(
                    "Source code",
                    href="https://github.com/AdamsMP92/MinimalHyperbolicMicromagnetics",
                    target="_blank",
                    rel="noopener noreferrer",
                ),
                class_="app-footer",
            ),
            class_="app-shell",
        ),
        class_="container-fluid app-page",
    ),
    title="Minimal Hyperbolic Micromagnetics",
    lang="en",
)


def server(input: Inputs, output: Outputs, session: Session) -> None:
    prepared_profiles = reactive.Value(None)
    prepared_profile_cache_hit = reactive.Value(False)
    profile_error = reactive.Value(None)
    calculation_state = reactive.Value((None, None))

    @reactive.effect
    @reactive.event(input.preset)
    def _apply_preset() -> None:
        for input_id, value in PRESETS[str(input.preset())].items():
            ui.update_numeric(input_id, value=value)

    def _collect_inputs() -> WebInputs:
        return WebInputs(
            Ku_J_per_m3=float(input.Ku_kJ_m3()) * 1.0e3,
            Ms_A_per_m=float(input.Ms_MA_m()) * 1.0e6,
            A_J_per_m=float(input.A_pJ_m()) * 1.0e-12,
            R_m=float(input.radius_nm()) * 1.0e-9,
            beta_deg=float(input.beta_deg()),
            Bmax_T=float(input.Bmax_T()),
            n_half=int(str(input.n_half())),
            use_h_double_prime=bool(input.use_h_double_prime()),
            stoner_wohlfarth=bool(input.stoner_wohlfarth()),
        )

    def _collect_profile_settings():
        return profile_settings_from_web(
            nu_max=float(input.nu_max()),
            n_nu=float(input.n_nu()),
            n_quad=float(input.n_quad()),
            l_max_demag=float(input.l_max_demag()),
            n_mu_demag=float(input.n_mu_demag()),
        )

    def _profiles_are_current() -> bool:
        profiles = prepared_profiles.get()
        if profiles is None:
            return False
        try:
            return profiles.settings == _collect_profile_settings()
        except (TypeError, ValueError):
            return False

    def _profiles() -> ProfileBundle:
        profiles = prepared_profiles.get()
        req(profiles is not None)
        assert profiles is not None
        return profiles

    @reactive.effect
    @reactive.event(input.compute_profiles)
    def _prepare_profiles() -> None:
        try:
            settings = _collect_profile_settings()
            with ui.Progress(min=0, max=1) as progress:
                progress.set(
                    0.05,
                    message="Preparing universal profiles",
                    detail=(
                        f"ν = 0…{settings.nu_max:g} with "
                        f"{settings.n_nu:,} samples…"
                    ),
                )
                profiles, cache_hit = get_profile_bundle(settings)
                progress.set(
                    1.0,
                    message="Profile table ready",
                    detail="The plot and profile download are available.",
                )
            prepared_profiles.set(profiles)
            prepared_profile_cache_hit.set(cache_hit)
            profile_error.set(None)
            calculation_state.set((None, None))
        except (ArithmeticError, TypeError, ValueError) as exc:
            profile_error.set(str(exc))
            calculation_state.set((None, None))
        except Exception as exc:
            profile_error.set(f"Profile calculation failed: {exc}")
            calculation_state.set((None, None))

    @reactive.effect
    def _synchronize_profile_state() -> None:
        # Reading these controls makes this effect invalidate immediately when
        # the requested table changes.  A previous table remains visible for
        # comparison, but cannot be used silently for a new hysteresis run.
        try:
            current = _profiles_are_current()
        except Exception:
            current = False
        ui.update_action_button(
            "calculate",
            disabled=not current,
            session=session,
        )
        if not current:
            calculation_state.set((None, None))

    @reactive.effect
    @reactive.event(
        input.nu_max,
        input.n_nu,
        input.n_quad,
        input.l_max_demag,
        input.n_mu_demag,
        ignore_init=True,
    )
    def _clear_profile_error_after_edit() -> None:
        profile_error.set(None)

    @reactive.effect
    @reactive.event(
        input.Ku_kJ_m3,
        input.Ms_MA_m,
        input.A_pJ_m,
        input.radius_nm,
        input.beta_deg,
        input.use_h_double_prime,
        input.stoner_wohlfarth,
        input.Bmax_T,
        input.n_half,
        ignore_init=True,
    )
    def _clear_calculation_after_model_edit() -> None:
        calculation_state.set((None, None))

    @reactive.effect
    @reactive.event(input.calculate)
    def _calculate_hysteresis() -> None:
        try:
            profiles = _profiles()
            requested = _collect_profile_settings()
            if profiles.settings != requested:
                raise ValueError(
                    "Profile settings changed. Compute the profile table again "
                    "before calculating hysteresis."
                )
            with ui.Progress(min=0, max=1) as progress:
                progress.set(
                    0.05,
                    message="Following the field path",
                    detail="Continuing local minima through the major loop…",
                )
                result = calculate_hysteresis(
                    _collect_inputs(),
                    profiles,
                    profile_cache_hit=bool(prepared_profile_cache_hit.get()),
                )
                progress.set(
                    1.0,
                    message="Calculation complete",
                    detail="Loop, state path, and observables are ready.",
                )
            calculation_state.set((result, None))
        except (ArithmeticError, ValueError) as exc:
            calculation_state.set((None, str(exc)))
        except Exception as exc:
            calculation_state.set((None, f"Numerical calculation failed: {exc}"))

    def _result() -> WebCalculation:
        result, _ = calculation_state.get()
        req(result is not None)
        assert result is not None
        return result

    @output
    @render.ui
    def profile_preparation_status() -> ui.Tag:
        error = profile_error.get()
        profiles = prepared_profiles.get()
        if error is not None:
            return ui.div(
                ui.tags.strong("Profile settings need attention"),
                ui.tags.span(error),
                class_="profile-status profile-status-error",
            )
        if profiles is None:
            return ui.div(
                ui.tags.strong("No table prepared"),
                ui.tags.span("Compute the profiles to unlock hysteresis."),
                class_="profile-status profile-status-neutral",
            )
        if not _profiles_are_current():
            return ui.div(
                ui.tags.strong("Settings changed"),
                ui.tags.span("Recompute before calculating hysteresis."),
                class_="profile-status profile-status-warning",
            )
        source = (
            "Loaded from this tab's cache"
            if prepared_profile_cache_hit.get()
            else f"Computed in {profiles.elapsed_s:.3f} s"
        )
        return ui.div(
            ui.tags.strong("Profile table ready"),
            ui.tags.span(source),
            class_="profile-status profile-status-ready",
        )

    @output
    @render.ui
    def profile_chip() -> ui.Tag:
        profiles = prepared_profiles.get()
        if profiles is None:
            return ui.tags.span("Not prepared", class_="method-chip profile-chip-muted")
        if _profiles_are_current():
            return ui.tags.span("Ready", class_="method-chip profile-chip-ready")
        return ui.tags.span(
            "Recompute required",
            class_="method-chip profile-chip-warning",
        )

    @output
    @render.plot
    def profile_plot():
        profiles = prepared_profiles.get()
        if profiles is None:
            return _profile_placeholder_figure()
        return _profile_figure(profiles)

    @output
    @render.ui
    def profile_card_footer() -> ui.Tag:
        profiles = prepared_profiles.get()
        if profiles is None:
            return ui.div(
                ui.tags.p(
                    "The material-independent table is calculated locally and "
                    "cached by its complete numerical configuration."
                ),
                class_="profile-card-footer profile-card-empty",
            )
        settings = profiles.settings
        usage_note = (
            "H″ is selected: gᵤˣ remains visible here but is omitted from the "
            "hysteresis Hamiltonian."
            if bool(input.use_h_double_prime())
            else "H′ is selected: both anisotropy profiles enter the Hamiltonian."
        )
        return ui.div(
            ui.div(
                ui.tags.strong(
                    f"ν = {settings.nu_min:g}…{settings.nu_max:g} · "
                    f"Nν = {settings.n_nu:,}"
                ),
                ui.tags.span(
                    f"Nquad = {settings.n_quad} · lmax = "
                    f"{settings.l_max_demag} · Nμ = {effective_n_mu(settings)}"
                ),
                ui.tags.span(usage_note),
                class_="profile-card-meta",
            ),
            ui.download_button(
                "download_profiles",
                "Download profile CSV",
                class_="download-secondary",
            ),
            class_="profile-card-footer",
        )

    @output
    @render.ui
    def calculation_message() -> ui.Tag | None:
        result, error = calculation_state.get()
        if error is not None:
            return ui.div(
                ui.tags.strong("Check the model definition."),
                ui.tags.span(error),
                class_="calculation-message message-error",
            )
        if result is None:
            profiles_ready = _profiles_are_current()
            detail = (
                "The profile table is ready. Configure the particle and run the "
                "field path."
                if profiles_ready
                else (
                    "A current profile table is required before the field path "
                    "can run."
                )
            )
            return ui.div(
                ui.tags.strong("Step 2 · Calculate hysteresis"),
                ui.tags.span(detail),
                class_="calculation-message message-neutral",
            )
        success = ui.div(
            ui.tags.strong(f"{result.model_label} major loop complete."),
            ui.tags.span(
                f"Used the prepared {result.profiles.settings.n_nu:,}-point "
                "profile table."
            ),
            class_="calculation-message message-success",
        )
        if not result.profile_boundary_warning:
            return success
        warning = ui.div(
            ui.tags.strong("Profile range reached."),
            ui.tags.span(
                f"The continued solution reached ν = "
                f"{result.max_vortex_coordinate:.5g}, close to νmax = "
                f"{result.profiles.settings.nu_max:g}. Increase νmax, recompute "
                "the profiles, and repeat the loop."
            ),
            class_="calculation-message message-warning",
        )
        return ui.TagList(success, warning)

    @output
    @render.ui
    def model_chip() -> ui.Tag:
        return ui.tags.span(_result().model_label, class_="method-chip")

    @output
    @render.ui
    def summary_cards() -> ui.Tag:
        result = _result()
        descending = result.analysis.descending
        req(descending is not None)
        assert descending is not None
        if result.inputs.stoner_wohlfarth:
            model_detail = "Uniform coherent-rotation reference"
        elif result.inputs.use_h_double_prime:
            model_detail = "Transverse anisotropy profile omitted"
        else:
            model_detail = "Transverse anisotropy profile retained"
        return ui.div(
            _metric_card(
                "Model",
                result.model_label,
                "",
                model_detail,
                accent="indigo",
            ),
            _metric_card(
                "Remanence",
                _number(descending.remanence),
                "",
                "Descending branch at Bext = 0",
            ),
            _metric_card(
                "Coercive field",
                _millitesla(descending.coercive_field_T),
                "mT",
                "Signed descending-branch zero crossing",
                accent="red",
            ),
            _metric_card(
                "Vortex nucleation",
                _millitesla(descending.vortex_nucleation_field_T),
                "mT",
                "Uniform-state stability criterion",
                accent="indigo",
            ),
            _metric_card(
                "Exchange length",
                _number(result.parameters.exchange_length_m * 1.0e9),
                "nm",
                "sqrt(2A / μ₀Ms²)",
            ),
            _metric_card(
                "Anisotropy field",
                _millitesla(result.parameters.anisotropy_field_T),
                "mT",
                "2Ku / Ms",
                accent="red",
            ),
            class_="metrics-grid",
        )

    @output
    @render.plot
    def hysteresis_plot():
        return _hysteresis_figure(_result())

    @output
    @render.plot
    def state_plot():
        return _state_figure(_result())

    @output
    @render.ui
    def diagnostics() -> ui.Tag:
        result = _result()
        settings = result.profiles.settings
        status_class = "status-warning" if result.profile_boundary_warning else "status-ok"
        status_text = (
            "Profile range reached"
            if result.profile_boundary_warning
            else "Profile range sufficient"
        )
        return ui.div(
            ui.div(
                ui.tags.span(class_=f"status-light {status_class}"),
                ui.tags.strong(status_text),
                class_="status-line",
            ),
            ui.tags.dl(
                _status_item("ν interval", f"{settings.nu_min:g}…{settings.nu_max:g}"),
                _status_item("Profile points", f"{settings.n_nu:,}"),
                _status_item("Local quadrature", str(settings.n_quad)),
                _status_item("Magnetostatic lmax", str(settings.l_max_demag)),
                _status_item("Magnetostatic Nμ", str(effective_n_mu(settings))),
                _status_item("Maximum selected ν", _number(result.max_vortex_coordinate)),
                _status_item("Profile time", f"{result.profiles.elapsed_s:.3f} s"),
                _status_item(
                    "Loop time",
                    f"{result.hysteresis_elapsed_s:.3f} s",
                ),
                class_="status-grid",
            ),
        )

    @render.download_button(
        filename=lambda: (
            f"minimal_hyperbolic_profiles_numax{_profiles().settings.nu_max:g}_"
            f"Nnu{_profiles().settings.n_nu}.csv"
        ),
        media_type="text/csv",
    )
    def download_profiles():
        yield build_profile_csv(_profiles())

    @render.download_button(
        filename=lambda: (
            f"minimal_hyperbolic_{_result().model_label.replace('′', 'prime').replace('″', 'doubleprime')}_"
            f"R{_result().inputs.R_m * 1e9:g}nm.zip"
        ),
        media_type="application/zip",
    )
    def download_result():
        yield build_result_archive(_result())


app = App(app_ui, server, static_assets=APP_DIR / "www")
