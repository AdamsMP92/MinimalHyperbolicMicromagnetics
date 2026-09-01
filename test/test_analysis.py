import numpy as np

from minimal_hyperbolic_micromagnetics import (
    HysteresisResult,
    HysteresisSettings,
    ModelParameters,
    all_profiles,
    analyze_hysteresis,
    coercive_field_from_hysteresis,
    hamiltonian,
    hamiltonian_hessian,
    remanent_magnetization,
    run_hysteresis,
    split_field_branches,
    vortex_nucleation_field_from_hysteresis,
)


def _synthetic_loop():
    fields = np.array([1.0, 0.5, 0.0, -0.5, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0])
    magnetization = np.array([1.0, 0.8, 0.6, -0.5, -1.0, -1.0, -0.8, -0.6, 0.5, 1.0])
    tau = np.array([0.0, 0.1, 0.2, 2.8, 3.1, 3.1, 3.0, 2.9, 0.3, 0.0])
    return HysteresisResult(
        B_T=fields,
        mz_avg=magnetization,
        nu_min=np.zeros_like(fields),
        tau_rad=tau,
        energy=np.zeros_like(fields),
    )


def test_branch_split_and_interpolated_remanence_and_coercivity():
    result = _synthetic_loop()
    branches = split_field_branches(result)

    assert [branch.name for branch in branches] == ["descending", "ascending"]
    assert len(branches[0].B_T) == 5
    assert len(branches[1].B_T) == 5
    assert np.isclose(remanent_magnetization(result), 0.6)
    assert np.isclose(
        remanent_magnetization(result, branch="ascending"),
        -0.6,
    )
    assert np.isclose(
        coercive_field_from_hysteresis(result),
        -0.5 * 0.6 / 1.1,
    )
    assert np.isclose(
        coercive_field_from_hysteresis(result, branch="ascending"),
        0.5 * 0.6 / 1.1,
    )


def test_uniform_vortex_curvature_recovers_analytic_nucleation_field():
    params = ModelParameters(Ku=4.8e4, Ms=1.7e6, A=1.0e-11, R=20.0e-9)
    fields = np.linspace(1.0, -1.0, 101)
    result = run_hysteresis(
        params,
        settings=HysteresisSettings(fields=fields, stoner_wohlfarth=True),
    )

    numerical = vortex_nucleation_field_from_hysteresis(result)
    assert np.isclose(numerical, params.nucleation_field_T(), atol=1.0e-14)
    assert np.all(np.isfinite(result.stability_eigenvalue_min))
    assert np.all(np.isfinite(result.stability_eigenvalue_max))


def test_analytic_hessian_matches_reduced_energy_finite_differences():
    params = ModelParameters(
        Ku=4.8e4,
        Ms=1.7e6,
        A=1.0e-11,
        R=20.0e-9,
        beta_deg=23.0,
        gux_factor=1.0,
    )
    B = 0.2
    nu = 1.0
    tau = 0.7
    step = 1.0e-3
    profiles = all_profiles(
        np.array([nu - step, nu, nu + step]),
        n_quad=160,
        l_max=41,
    )

    def energy(nu_index, angle):
        return hamiltonian(
            params,
            B,
            angle,
            profiles["g_ex"][nu_index],
            profiles["g_u_z"][nu_index],
            profiles["g_u_x"][nu_index],
            profiles["g_z_z"][nu_index],
            profiles["g_dem"][nu_index],
        )

    finite_hessian = np.array(
        [
            [
                (energy(2, tau) - 2.0 * energy(1, tau) + energy(0, tau))
                / step**2,
                (
                    energy(2, tau + step)
                    - energy(2, tau - step)
                    - energy(0, tau + step)
                    + energy(0, tau - step)
                )
                / (4.0 * step**2),
            ],
            [
                0.0,
                (
                    energy(1, tau + step)
                    - 2.0 * energy(1, tau)
                    + energy(1, tau - step)
                )
                / step**2,
            ],
        ]
    )
    finite_hessian[1, 0] = finite_hessian[0, 1]

    analytic_hessian = hamiltonian_hessian(
        params,
        B,
        tau,
        guz=profiles["g_u_z"][1],
        gux=profiles["g_u_x"][1],
        gzz=profiles["g_z_z"][1],
        gex_d2=profiles["g_ex_d2"][1],
        guz_d1=profiles["g_u_z_d1"][1],
        guz_d2=profiles["g_u_z_d2"][1],
        gux_d1=profiles["g_u_x_d1"][1],
        gux_d2=profiles["g_u_x_d2"][1],
        gzz_d1=profiles["g_z_z_d1"][1],
        gzz_d2=profiles["g_z_z_d2"][1],
        gdem_d2=profiles["g_dem_d2"][1],
    )

    assert np.allclose(analytic_hessian, finite_hessian, rtol=2.0e-5)


def test_analysis_returns_signed_branch_observables():
    analysis = analyze_hysteresis(_synthetic_loop(), nucleation_method="threshold")

    assert analysis.descending is not None
    assert analysis.ascending is not None
    assert analysis.descending.remanence > 0.0
    assert analysis.descending.coercive_field_T < 0.0
    assert analysis.ascending.remanence < 0.0
    assert analysis.ascending.coercive_field_T > 0.0


def test_hysteresis_result_keeps_elapsed_time_positional_compatibility():
    values = np.arange(2.0)
    result = HysteresisResult(values, values, values, values, values, 1.25)

    assert result.elapsed_s == 1.25


def test_single_point_legacy_profile_still_runs_without_derivative_columns():
    profiles = {
        "nu": np.array([0.0]),
        "g_ex": np.array([0.0]),
        "g_u_x": np.array([0.0]),
        "g_u_z": np.array([1.0]),
        "g_z_z": np.array([1.0]),
        "g_dem": np.array([1.0 / 3.0]),
    }
    params = ModelParameters(Ku=4.8e4, Ms=1.7e6, A=1.0e-11, R=10.0e-9)
    result = run_hysteresis(
        params,
        profiles,
        settings=HysteresisSettings(fields=np.array([1.0, 0.0, -1.0])),
    )

    assert np.all(np.isfinite(result.stability_eigenvalue_min))
