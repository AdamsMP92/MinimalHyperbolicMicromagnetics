from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np

from minimal_hyperbolic_micromagnetics import ModelParameters, ProfileComputation
from web_support import (
    WebInputs,
    build_result_archive,
    calculate_hysteresis,
    clear_profile_cache,
    get_profile_bundle,
    model_parameters,
)


def _inputs(**overrides):
    values = {
        "Ku_J_per_m3": 4.8e4,
        "Ms_A_per_m": 1.7e6,
        "A_J_per_m": 1.0e-11,
        "R_m": 12.0e-9,
        "Bmax_T": 0.2,
        "n_half": 24,
    }
    values.update(overrides)
    return WebInputs(**values)


def _small_profile_settings():
    return ProfileComputation(
        nu_min=0.0,
        nu_max=4.0,
        n_nu=32,
        n_quad=48,
        l_max_demag=9,
        n_mu_demag=48,
    )


def test_web_defaults_to_hprime_without_changing_package_default():
    assert model_parameters(_inputs()).gux_factor == 1.0
    assert model_parameters(_inputs(use_h_double_prime=True)).gux_factor == 0.0

    package_default = ModelParameters(Ku=4.8e4, Ms=1.7e6, A=1e-11, R=12e-9)
    assert package_default.gux_factor == 0.0


def test_profile_table_is_computed_once_for_identical_settings():
    clear_profile_cache()
    settings = _small_profile_settings()

    first, first_hit = get_profile_bundle(settings)
    second, second_hit = get_profile_bundle(settings)

    assert first_hit is False
    assert second_hit is True
    assert second is first
    assert len(first.values["nu"]) == settings.n_nu
    assert all(not values.flags.writeable for values in first.values.values())


def test_web_calculation_and_download_archive_use_existing_core():
    clear_profile_cache()
    profiles, cache_hit = get_profile_bundle(_small_profile_settings())
    calculation = calculate_hysteresis(
        _inputs(),
        profiles,
        profile_cache_hit=cache_hit,
    )

    assert calculation.model_label == "H′"
    assert calculation.analysis.descending is not None
    assert calculation.analysis.ascending is not None
    assert len(calculation.result.B_T) == 2 * calculation.inputs.n_half
    assert np.all(np.isfinite(calculation.result.mz_avg))

    archive_bytes = build_result_archive(calculation)
    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == {"hysteresis.csv", "metadata.json"}
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["model"] == "H′"
        assert metadata["inputs"]["use_h_double_prime"] is False
        assert metadata["profile_settings"]["n_nu"] == 32
