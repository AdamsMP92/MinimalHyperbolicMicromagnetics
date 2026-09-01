import numpy as np

from minimal_hyperbolic_micromagnetics import (
    MU_0,
    HysteresisSettings,
    ModelParameters,
    STONER_WOHLFARTH_ENSEMBLE_COERCIVE_RATIO,
    STONER_WOHLFARTH_ENSEMBLE_REMANENCE_RATIO,
    critical_anisotropy_for_zero_nucleation_field,
    exchange_length,
    run_hysteresis,
    stoner_wohlfarth_astroid,
    stoner_wohlfarth_coercive_field,
    stoner_wohlfarth_ensemble_coercive_field,
    stoner_wohlfarth_ensemble_remanence,
    stoner_wohlfarth_switching_field,
    vortex_nucleation_field,
    vortex_nucleation_radius,
)


def test_model_parameters_use_analytic_nucleation_formulas():
    params = ModelParameters(Ku=4.8e4, Ms=1.7e6, A=1e-11, R=20e-9)

    assert np.isclose(
        params.nucleation_field_T(),
        vortex_nucleation_field(params.Ku, params.Ms, params.A, params.R),
    )
    assert np.isclose(
        params.nucleation_radius_m,
        vortex_nucleation_radius(params.Ku, params.Ms, params.A),
    )


def test_zero_nucleation_field_condition():
    Ms = 1.7e6
    A = 1e-11
    R = 20e-9
    Ku = critical_anisotropy_for_zero_nucleation_field(Ms, A, R)

    assert np.isclose(vortex_nucleation_field(Ku, Ms, A, R), 0.0, atol=1e-14)


def test_exchange_length_formula():
    A = 1e-11
    Ms = 1.7e6

    assert np.isclose(exchange_length(A, Ms), np.sqrt(2.0 * A / (MU_0 * Ms**2)))


def test_stoner_wohlfarth_astroid_residual():
    psi = np.linspace(0.0, 2.0 * np.pi, 2001)
    h_parallel, h_perpendicular = stoner_wohlfarth_astroid(psi)

    residual = (
        np.abs(h_parallel) ** (2.0 / 3.0)
        + np.abs(h_perpendicular) ** (2.0 / 3.0)
        - 1.0
    )
    assert np.max(np.abs(residual)) < 1e-14


def test_stoner_wohlfarth_coercive_field_piecewise_limit():
    anisotropy_field = 1.0
    theta = np.deg2rad(np.array([30.0, 60.0]))

    switching = stoner_wohlfarth_switching_field(theta, anisotropy_field)
    coercive = stoner_wohlfarth_coercive_field(theta, anisotropy_field)

    assert np.isclose(coercive[0], switching[0])
    assert np.isclose(coercive[1], 0.5 * np.sin(2.0 * theta[1]))


def test_isotropic_stoner_wohlfarth_ensemble_reference_values():
    assert STONER_WOHLFARTH_ENSEMBLE_REMANENCE_RATIO == 0.5
    assert np.isclose(
        STONER_WOHLFARTH_ENSEMBLE_COERCIVE_RATIO,
        0.48221204600969,
        rtol=0.0,
        atol=5.0e-15,
    )
    assert stoner_wohlfarth_ensemble_remanence() == 0.5
    assert stoner_wohlfarth_ensemble_remanence(2.0) == 1.0
    assert np.isclose(
        stoner_wohlfarth_ensemble_coercive_field(2.0),
        2.0 * STONER_WOHLFARTH_ENSEMBLE_COERCIVE_RATIO,
    )


def test_ensemble_coercive_ratio_zeros_orientation_averaged_sw_branch():
    nodes, weights = np.polynomial.legendre.leggauss(32)
    cos_beta = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    beta_deg = np.rad2deg(np.arccos(cos_beta))
    fields = np.array([1.2, -STONER_WOHLFARTH_ENSEMBLE_COERCIVE_RATIO])

    ensemble_mz_at_coercivity = 0.0
    for angle_deg, weight in zip(beta_deg, weights):
        result = run_hysteresis(
            ModelParameters(
                Ku=0.5,
                Ms=1.0,
                A=1.0,
                R=1.0,
                beta_deg=angle_deg,
            ),
            settings=HysteresisSettings(
                fields=fields,
                stoner_wohlfarth=True,
            ),
        )
        ensemble_mz_at_coercivity += weight * result.mz_avg[-1]

    assert abs(ensemble_mz_at_coercivity) < 1.0e-12
