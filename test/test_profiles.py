import numpy as np
import pandas as pd

from minimal_hyperbolic_micromagnetics import (
    ProfileComputation,
    all_profiles,
    compute_and_store_profiles,
    compute_profiles,
)


def test_uniform_limit_profiles():
    profiles = all_profiles(np.array([0.0]), l_max=41)

    assert np.isclose(profiles["g_ex"][0], 0.0)
    assert np.isclose(profiles["g_u_x"][0], 0.0)
    assert np.isclose(profiles["g_u_z"][0], 1.0)
    assert np.isclose(profiles["g_z_z"][0], 1.0)
    assert np.isclose(profiles["g_H"][0], 1.0 / 6.0, rtol=2e-10, atol=2e-10)
    assert np.isclose(profiles["g_s_B_energy"][0], -1.0 / 3.0, rtol=2e-10, atol=2e-10)
    assert np.isclose(profiles["g_dem"][0], 1.0 / 3.0, rtol=2e-10, atol=2e-10)


def test_gux_reduces_to_guz_identity():
    nu = np.linspace(0.0, 8.0, 17)
    profiles = all_profiles(nu, n_quad=160, l_max=41)
    expected = 4.0 / 3.0 * (1.0 - profiles["g_u_z"])
    assert np.allclose(profiles["g_u_x"], expected, rtol=1e-11, atol=1e-11)


def test_compute_profiles_uses_parametrized_grid():
    settings = ProfileComputation(nu_min=0.0, nu_max=1.0, n_nu=5, n_quad=80, l_max_demag=21)
    profiles = compute_profiles(settings)

    assert np.allclose(profiles["nu"], np.linspace(0.0, 1.0, 5))
    assert set(["g_ex", "g_u_x", "g_u_z", "g_z_z", "g_dem"]).issubset(profiles)


def test_compute_and_store_profiles_writes_csv(tmp_path):
    settings = ProfileComputation(nu_min=0.0, nu_max=1.0, n_nu=4, n_quad=80, l_max_demag=21)
    output_csv = tmp_path / "profiles.csv"

    dataframe = compute_and_store_profiles(output_csv, settings)
    stored = pd.read_csv(output_csv)

    assert output_csv.exists()
    assert len(dataframe) == 4
    assert len(stored) == 4
    assert np.allclose(stored["nu"], np.linspace(0.0, 1.0, 4))
