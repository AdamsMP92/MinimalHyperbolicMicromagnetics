import numpy as np
import pytest

from minimal_hyperbolic_micromagnetics import (
    MU_0,
    HysteresisSettings,
    ModelParameters,
    STONER_WOHLFARTH_ENSEMBLE_COERCIVE_RATIO,
    STONER_WOHLFARTH_ENSEMBLE_REMANENCE_RATIO,
    coercive_field_from_vortex_parameter,
    coercive_nucleation_radius,
    coercive_profile_curvature,
    coercive_radius_from_vortex_parameter,
    coercive_vortex_parameter,
    critical_reduced_anisotropy_for_two_stage_remanence,
    critical_anisotropy_for_zero_nucleation_field,
    exchange_length,
    g_z_z,
    hyperbolic_vortex_coercive_field,
    remanent_magnetization_ratio,
    remanent_nucleation_radius,
    remanent_profile_curvature,
    remanent_radius_from_vortex_parameter,
    remanent_reorientation_parameter,
    remanent_reorientation_radius,
    remanent_transverse_jump_parameter,
    remanent_vortex_parameter,
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


def test_remanent_characteristic_parameter_and_radii():
    Ku = 4.8e4
    Ms = 1.7e6
    A = 1.0e-11

    nu_star = remanent_reorientation_parameter()
    nucleation_radius = vortex_nucleation_radius(Ku, Ms, A)
    reorientation_radius = remanent_reorientation_radius(Ku, Ms, A)

    assert np.isclose(nu_star, 2.164008238, rtol=0.0, atol=5.0e-9)
    assert np.isclose(
        critical_reduced_anisotropy_for_two_stage_remanence(),
        0.379882286,
        rtol=0.0,
        atol=5.0e-9,
    )
    assert np.isclose(
        remanent_radius_from_vortex_parameter(0.0, Ku, Ms, A),
        nucleation_radius,
        rtol=2.0e-12,
    )
    assert np.isclose(reorientation_radius, 11.10164566e-9, rtol=2.0e-8)
    assert np.isclose(
        remanent_transverse_jump_parameter(Ku, Ms, A),
        2.488879262,
        rtol=0.0,
        atol=2.0e-9,
    )
    assert remanent_profile_curvature(
        nu_star,
        Ku,
        Ms,
        A,
        orientation="axial",
        radius=reorientation_radius,
    ) > 0.0


def test_piecewise_remanent_inverse_and_magnetization():
    Ku = 4.8e4
    Ms = 1.7e6
    A = 1.0e-11
    nu_star = remanent_reorientation_parameter()
    nucleation_radius = vortex_nucleation_radius(Ku, Ms, A)
    reorientation_radius = remanent_reorientation_radius(Ku, Ms, A)
    axial_nu = 1.25
    transverse_nu = 4.0
    axial_radius = remanent_radius_from_vortex_parameter(
        axial_nu,
        Ku,
        Ms,
        A,
        orientation="axial",
    )
    transverse_radius = remanent_radius_from_vortex_parameter(
        transverse_nu,
        Ku,
        Ms,
        A,
        orientation="transverse",
    )
    radii = np.array(
        [
            0.9 * nucleation_radius,
            nucleation_radius,
            axial_radius,
            reorientation_radius,
            reorientation_radius * (1.0 + 1.0e-7),
            transverse_radius,
        ]
    )

    nu_rem = remanent_vortex_parameter(radii, Ku, Ms, A)
    assert np.allclose(nu_rem[:2], 0.0)
    assert np.isclose(nu_rem[2], axial_nu, rtol=0.0, atol=1.0e-9)
    assert np.isclose(nu_rem[3], nu_star, rtol=0.0, atol=1.0e-9)
    assert np.isclose(nu_rem[4], 2.4888798, rtol=0.0, atol=2.0e-6)
    assert np.isclose(nu_rem[5], transverse_nu, rtol=0.0, atol=1.0e-9)

    remanence = remanent_magnetization_ratio(radii, Ku, Ms, A)
    assert np.allclose(remanence[:2], 1.0)
    assert np.isclose(remanence[2], g_z_z(axial_nu), rtol=0.0, atol=1.0e-12)
    assert np.isclose(remanence[3], g_z_z(nu_star), rtol=0.0, atol=1.0e-12)
    assert np.allclose(remanence[4:], 0.0)


def test_piecewise_remanent_formula_rejects_collapsed_axial_window():
    Ms = 1.7e6
    A = 1.0e-11
    kappa_u = 0.39
    Ku = kappa_u**2 * MU_0 * Ms**2

    with pytest.raises(ValueError, match="two-stage regime"):
        remanent_vortex_parameter(12.0e-9, Ku, Ms, A)


def test_coercive_parameterization_endpoints():
    Ku = 4.8e4
    Ms = 1.7e6
    A = 1.0e-11
    nu_star = remanent_reorientation_parameter()
    coercive_radius = coercive_nucleation_radius(Ms, A)
    reorientation_radius = remanent_reorientation_radius(Ku, Ms, A)

    assert np.isclose(coercive_radius, 9.088804764e-9, rtol=2.0e-10)
    assert np.isclose(
        coercive_radius_from_vortex_parameter(0.0, Ku, Ms, A),
        coercive_radius,
        rtol=2.0e-12,
    )
    assert np.isclose(
        coercive_radius_from_vortex_parameter(nu_star, Ku, Ms, A),
        reorientation_radius,
        rtol=2.0e-8,
    )
    assert np.isclose(
        coercive_field_from_vortex_parameter(0.0, Ku, Ms),
        2.0 * Ku / Ms,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert np.isclose(
        coercive_field_from_vortex_parameter(nu_star, Ku, Ms),
        0.0,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_piecewise_coercive_inverse_and_field():
    Ku = 4.8e4
    Ms = 1.7e6
    A = 1.0e-11
    nu_star = remanent_reorientation_parameter()
    coercive_radius = coercive_nucleation_radius(Ms, A)
    reorientation_radius = remanent_reorientation_radius(Ku, Ms, A)
    test_nu = 1.25
    test_radius = coercive_radius_from_vortex_parameter(
        test_nu,
        Ku,
        Ms,
        A,
    )
    radii = np.array(
        [
            0.9 * coercive_radius,
            coercive_radius,
            test_radius,
            reorientation_radius,
            1.1 * reorientation_radius,
        ]
    )

    nu_c = coercive_vortex_parameter(radii, Ku, Ms, A)
    field = hyperbolic_vortex_coercive_field(radii, Ku, Ms, A)

    assert np.allclose(nu_c[:2], 0.0)
    assert np.isclose(nu_c[2], test_nu, rtol=0.0, atol=1.0e-9)
    assert np.isclose(nu_c[3], nu_star, rtol=0.0, atol=1.0e-9)
    assert nu_c[4] > nu_star
    assert np.allclose(field[:2], 2.0 * Ku / Ms)
    assert np.isclose(
        field[2],
        coercive_field_from_vortex_parameter(test_nu, Ku, Ms),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.allclose(field[3:], 0.0)
    assert coercive_profile_curvature(test_nu, Ku, Ms, A) > 0.0


def test_model_parameters_expose_distinct_nucleation_radii():
    params = ModelParameters(Ku=4.8e4, Ms=1.7e6, A=1.0e-11, R=12.0e-9)

    assert np.isclose(
        params.coercive_nucleation_radius_m,
        coercive_nucleation_radius(params.Ms, params.A),
    )
    assert np.isclose(
        params.remanent_nucleation_radius_m,
        remanent_nucleation_radius(params.Ku, params.Ms, params.A),
    )
    assert params.nucleation_radius_m == params.remanent_nucleation_radius_m
    assert (
        params.coercive_nucleation_radius_m
        < params.remanent_nucleation_radius_m
    )


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
