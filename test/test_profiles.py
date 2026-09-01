import numpy as np
import pandas as pd

from minimal_hyperbolic_micromagnetics import (
    ProfileComputation,
    all_profiles,
    compute_and_store_profiles,
    compute_profiles,
    profile_derivatives,
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


def test_compute_profiles_reports_completed_integration_stages(capsys):
    settings = ProfileComputation(
        nu_min=0.0,
        nu_max=1.0,
        n_nu=3,
        n_quad=40,
        l_max_demag=9,
        n_mu_demag=40,
    )

    compute_profiles(settings, verbose=True)
    output = capsys.readouterr().out

    assert "local energy-profile integrals: done in" in output
    assert "magnetostatic Legendre-moment sum: done in" in output
    assert "analytic local-profile derivatives: done in" in output
    assert "analytic magnetostatic derivatives: done in" in output
    assert "tabulation: done in" in output


def test_compute_and_store_profiles_writes_csv(tmp_path):
    settings = ProfileComputation(nu_min=0.0, nu_max=1.0, n_nu=4, n_quad=80, l_max_demag=21)
    output_csv = tmp_path / "profiles.csv"

    dataframe = compute_and_store_profiles(output_csv, settings)
    stored = pd.read_csv(output_csv)

    assert output_csv.exists()
    assert len(dataframe) == 4
    assert len(stored) == 4
    assert np.allclose(stored["nu"], np.linspace(0.0, 1.0, 4))


def test_profile_derivatives_have_exact_uniform_limits():
    derivatives = profile_derivatives(0.0, n_quad=80, l_max=21)

    assert derivatives["g_ex_d1"] == 0.0
    assert derivatives["g_ex_d2"] == 4.0
    assert derivatives["g_u_z_d2"] == -4.0 / 5.0
    assert derivatives["g_u_x_d2"] == 16.0 / 15.0
    assert derivatives["g_z_z_d2"] == -2.0 / 5.0
    assert derivatives["g_dem_d2"] == 2.0 / 15.0


def test_analytic_profile_derivatives_match_centered_differences():
    center = 1.0
    step = 1.0e-3
    nu = np.array([center - step, center, center + step])
    profiles = all_profiles(nu, n_quad=160, l_max=41)

    for name in ("g_ex", "g_u_x", "g_u_z", "g_z_z", "g_dem"):
        finite_d1 = (profiles[name][2] - profiles[name][0]) / (2.0 * step)
        finite_d2 = (
            profiles[name][2]
            - 2.0 * profiles[name][1]
            + profiles[name][0]
        ) / step**2
        assert np.isclose(profiles[f"{name}_d1"][1], finite_d1, rtol=2e-5)
        assert np.isclose(profiles[f"{name}_d2"][1], finite_d2, rtol=2e-5)
