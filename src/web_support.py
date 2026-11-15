"""Calculation adapter for the browser-native Shiny interface.

This module deliberately sits next to ``app.py`` instead of extending the
public package API.  It translates web-form values into the existing
``ModelParameters`` and ``HysteresisSettings`` classes without changing their
signatures or defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import json
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from minimal_hyperbolic_micromagnetics import (
    HysteresisAnalysis,
    HysteresisResult,
    HysteresisSettings,
    ModelParameters,
    ProfileComputation,
    analyze_hysteresis,
    compute_profiles,
    run_hysteresis,
)


# The profiles are universal functions of the reduced vortex coordinate.  The
# app computes this established production grid once per Python process (one
# process per browser tab in Shinylive) and reuses it for subsequent runs.
WEB_PROFILE_SETTINGS = ProfileComputation(
    nu_min=0.0,
    nu_max=20.0,
    n_nu=2000,
    n_quad=360,
    l_max_demag=161,
)


@dataclass(frozen=True)
class WebInputs:
    """Validated SI inputs collected by the Shiny interface."""

    Ku_J_per_m3: float
    Ms_A_per_m: float
    A_J_per_m: float
    R_m: float
    beta_deg: float = 0.0
    Bmax_T: float = 1.0
    n_half: int = 250
    use_h_double_prime: bool = False
    stoner_wohlfarth: bool = False

    def validate(self) -> None:
        numeric = {
            "Ku": self.Ku_J_per_m3,
            "Ms": self.Ms_A_per_m,
            "A": self.A_J_per_m,
            "R": self.R_m,
            "beta": self.beta_deg,
            "Bmax": self.Bmax_T,
        }
        invalid = [name for name, value in numeric.items() if not np.isfinite(value)]
        if invalid:
            raise ValueError(f"Non-finite input: {', '.join(invalid)}")
        if self.Ku_J_per_m3 < 0.0:
            raise ValueError("Ku must be non-negative in the web interface.")
        if self.Ms_A_per_m <= 0.0:
            raise ValueError("Ms must be positive.")
        if self.A_J_per_m <= 0.0:
            raise ValueError("A must be positive.")
        if self.R_m <= 0.0:
            raise ValueError("R must be positive.")
        if not 0.0 <= self.beta_deg <= 90.0:
            raise ValueError("beta must lie between 0 and 90 degrees.")
        if self.Bmax_T <= 0.0:
            raise ValueError("Bmax must be positive.")
        if not 16 <= int(self.n_half) <= 2000:
            raise ValueError("n_half must lie between 16 and 2000.")


@dataclass(frozen=True)
class ProfileBundle:
    """One reusable material-independent profile table."""

    values: dict[str, np.ndarray]
    settings: ProfileComputation
    elapsed_s: float


@dataclass(frozen=True)
class WebCalculation:
    """Complete state retained by the web interface after one calculation."""

    inputs: WebInputs
    parameters: ModelParameters
    settings: HysteresisSettings
    profiles: ProfileBundle
    profile_cache_hit: bool
    result: HysteresisResult
    analysis: HysteresisAnalysis
    hysteresis_elapsed_s: float

    @property
    def model_label(self) -> str:
        if self.inputs.stoner_wohlfarth:
            return "Stoner–Wohlfarth"
        if self.inputs.use_h_double_prime:
            return "H″"
        return "H′"


_PROFILE_CACHE: dict[ProfileComputation, ProfileBundle] = {}


def get_profile_bundle(
    settings: ProfileComputation = WEB_PROFILE_SETTINGS,
) -> tuple[ProfileBundle, bool]:
    """Return a cached profile table and whether it was already available."""
    cached = _PROFILE_CACHE.get(settings)
    if cached is not None:
        return cached, True

    started = time.perf_counter()
    raw = compute_profiles(settings)
    values = {name: np.asarray(value, dtype=float) for name, value in raw.items()}
    for value in values.values():
        value.setflags(write=False)
    bundle = ProfileBundle(
        values=values,
        settings=settings,
        elapsed_s=time.perf_counter() - started,
    )
    _PROFILE_CACHE[settings] = bundle
    return bundle, False


def clear_profile_cache() -> None:
    """Clear the process-local profile cache (primarily for focused tests)."""
    _PROFILE_CACHE.clear()


def model_parameters(inputs: WebInputs) -> ModelParameters:
    """Map web inputs onto the unchanged public model-parameter API."""
    inputs.validate()
    return ModelParameters(
        Ku=inputs.Ku_J_per_m3,
        Ms=inputs.Ms_A_per_m,
        A=inputs.A_J_per_m,
        R=inputs.R_m,
        beta_deg=inputs.beta_deg,
        # H' is the web default.  The explicit H'' switch removes only the
        # transverse g_u^x term through the package's existing multiplier.
        gux_factor=0.0 if inputs.use_h_double_prime else 1.0,
    )


def calculate_hysteresis(
    inputs: WebInputs,
    profiles: ProfileBundle,
    *,
    profile_cache_hit: bool,
) -> WebCalculation:
    """Run one app calculation using an already prepared profile bundle."""
    parameters = model_parameters(inputs)
    settings = HysteresisSettings(
        Bmax=inputs.Bmax_T,
        n_half=int(inputs.n_half),
        stoner_wohlfarth=inputs.stoner_wohlfarth,
    )
    started = time.perf_counter()
    result = run_hysteresis(parameters, profiles.values, settings=settings)
    elapsed_s = time.perf_counter() - started
    return WebCalculation(
        inputs=inputs,
        parameters=parameters,
        settings=settings,
        profiles=profiles,
        profile_cache_hit=profile_cache_hit,
        result=result,
        analysis=analyze_hysteresis(result),
        hysteresis_elapsed_s=elapsed_s,
    )


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _branch_metadata(calculation: WebCalculation) -> list[dict[str, object]]:
    return [
        {
            "name": branch.name,
            "occurrence": branch.occurrence,
            "remanence": _finite_or_none(branch.remanence),
            "coercive_field_T": _finite_or_none(branch.coercive_field_T),
            "switching_field_T": _finite_or_none(branch.switching_field_T),
            "vortex_nucleation_field_T": _finite_or_none(
                branch.vortex_nucleation_field_T
            ),
        }
        for branch in calculation.analysis.branches
    ]


def build_result_archive(calculation: WebCalculation) -> bytes:
    """Build a compact reproducibility archive in browser memory."""
    result_csv = pd.DataFrame(calculation.result.as_dict()).to_csv(index=False)
    profile_settings = asdict(calculation.profiles.settings)
    metadata = {
        "model": calculation.model_label,
        "inputs": asdict(calculation.inputs),
        "profile_settings": profile_settings,
        "profile_elapsed_s": calculation.profiles.elapsed_s,
        "profile_cache_hit": calculation.profile_cache_hit,
        "hysteresis_elapsed_s": calculation.hysteresis_elapsed_s,
        "derived": {
            "anisotropy_field_T": calculation.parameters.anisotropy_field_T,
            "exchange_length_m": calculation.parameters.exchange_length_m,
            "remanent_nucleation_radius_m": _finite_or_none(
                calculation.parameters.remanent_nucleation_radius_m
            ),
            "coercive_nucleation_radius_m": _finite_or_none(
                calculation.parameters.coercive_nucleation_radius_m
            ),
        },
        "branches": _branch_metadata(calculation),
    }

    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("hysteresis.csv", result_csv)
        archive.writestr(
            "metadata.json",
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False),
        )
    return output.getvalue()
