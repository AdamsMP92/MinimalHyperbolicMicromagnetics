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
    WebCalculation,
    WebInputs,
    build_result_archive,
    calculate_hysteresis,
    get_profile_bundle,
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
    _section_heading("01", "Material"),
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
    _section_heading("02", "Geometry & orientation"),
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
    _section_heading("03", "Model"),
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
    _section_heading("04", "Field protocol"),
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
    ui.div(
        ui.tags.strong("Fixed universal profile grid"),
        ui.tags.span(
            f"ν = 0…{WEB_PROFILE_SETTINGS.nu_max:g} · "
            f"{WEB_PROFILE_SETTINGS.n_nu} points · "
            f"quadrature {WEB_PROFILE_SETTINGS.n_quad} · "
            f"lmax {WEB_PROFILE_SETTINGS.l_max_demag}"
        ),
        class_="profile-grid-note",
    ),
    ui.input_action_button(
        "calculate",
        "Calculate hysteresis",
        class_="btn-calculate",
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
                "profiles depend only on ν. The app evaluates the fixed 2000-point "
                "production grid once and reuses it for every later material setup."
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

    @reactive.calc
    @reactive.event(input.calculate, ignore_none=False)
    def calculation() -> tuple[WebCalculation | None, str | None]:
        try:
            with ui.Progress(min=0, max=1) as progress:
                progress.set(
                    0.08,
                    message="Preparing universal profiles",
                    detail="Computing or loading the fixed ν grid…",
                )
                profiles, cache_hit = get_profile_bundle()
                progress.set(
                    0.52,
                    message="Following the field path",
                    detail="Continuing local minima through the major loop…",
                )
                result = calculate_hysteresis(
                    _collect_inputs(),
                    profiles,
                    profile_cache_hit=cache_hit,
                )
                progress.set(
                    1.0,
                    message="Calculation complete",
                    detail="Loop, state path, and observables are ready.",
                )
            return result, None
        except (ArithmeticError, ValueError) as exc:
            return None, str(exc)
        except Exception as exc:
            return None, f"Numerical calculation failed: {exc}"

    def _result() -> WebCalculation:
        result, _ = calculation()
        req(result is not None)
        assert result is not None
        return result

    @output
    @render.ui
    def calculation_message() -> ui.Tag | None:
        result, error = calculation()
        if error is not None:
            return ui.div(
                ui.tags.strong("Check the model definition."),
                ui.tags.span(error),
                class_="calculation-message message-error",
            )
        req(result is not None)
        assert result is not None
        profile_note = (
            "The existing universal profile table was reused."
            if result.profile_cache_hit
            else (
                "The universal profile table was computed in "
                f"{result.profiles.elapsed_s:.2f} s and cached for this tab."
            )
        )
        return ui.div(
            ui.tags.strong(f"{result.model_label} major loop complete."),
            ui.tags.span(profile_note),
            class_="calculation-message message-success",
        )

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
        return ui.div(
            ui.div(
                ui.tags.span(class_="status-light status-ok"),
                ui.tags.strong("Profile table ready"),
                class_="status-line",
            ),
            ui.tags.dl(
                _status_item("ν interval", f"{settings.nu_min:g}…{settings.nu_max:g}"),
                _status_item("Profile points", f"{settings.n_nu:,}"),
                _status_item("Local quadrature", str(settings.n_quad)),
                _status_item("Magnetostatic lmax", str(settings.l_max_demag)),
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
            f"minimal_hyperbolic_{_result().model_label.replace('′', 'prime').replace('″', 'doubleprime')}_"
            f"R{_result().inputs.R_m * 1e9:g}nm.zip"
        ),
        media_type="application/zip",
    )
    def download_result():
        yield build_result_archive(_result())


app = App(app_ui, server, static_assets=APP_DIR / "www")
