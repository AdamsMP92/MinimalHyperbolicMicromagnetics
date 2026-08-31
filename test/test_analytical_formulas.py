import numpy as np

from minimal_hyperbolic_micromagnetics import (
    MU_0,
    ModelParameters,
    critical_anisotropy_for_zero_nucleation_field,
    exchange_length,
    stoner_wohlfarth_astroid,
    stoner_wohlfarth_coercive_field,
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
