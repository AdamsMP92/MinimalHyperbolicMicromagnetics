"""Analytic formulas and standard reference values for the reduced model."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .profiles import MU_0, g_u_z, g_z_z, local_profiles, profile_derivatives


# Zero-temperature reference values for a noninteracting ensemble of identical
# uniaxial Stoner-Wohlfarth particles with isotropically distributed easy axes.
STONER_WOHLFARTH_ENSEMBLE_REMANENCE_RATIO = 0.5

# High-accuracy numerical zero of the orientation-averaged descending branch.
# This is not a closed-form constant.  Values in the literature are commonly
# rounded to 0.48 and are also quoted as either 0.479 or 0.482.  The latter is
# reported for the matching isotropic uniaxial ensemble in Table 1 and Fig. 2(b)
# of Adams, Sinaga, and Michels, IUCrJ 10 (2023),
# DOI: 10.1107/S205225252300180X.
STONER_WOHLFARTH_ENSEMBLE_COERCIVE_RATIO = 0.48221204600969


def exchange_length(A, Ms, mu0=MU_0):
    """Return the micromagnetic exchange length sqrt(2A/(mu0*Ms^2))."""
    return np.sqrt(2.0 * A / (mu0 * Ms**2))


def vortex_nucleation_field(Ku, Ms, A, R, mu0=MU_0):
    """Analytic vortex-nucleation field of the uniform reduced-model branch."""
    return mu0 * Ms / 3.0 - 2.0 * Ku / Ms - 10.0 * A / (Ms * np.asarray(R) ** 2)


def critical_anisotropy_for_zero_nucleation_field(Ms, A, R, mu0=MU_0):
    """Return Ku for which the analytic vortex-nucleation field is zero.
    """
    lex = exchange_length(A, Ms, mu0=mu0)
    return mu0 * Ms**2 / 6.0 * (1.0 - 15.0 * lex**2 / np.asarray(R) ** 2)


def vortex_nucleation_radius(Ku, Ms, A, mu0=MU_0):
    """Return the remanent nucleation radius (backward-compatible name)."""
    denominator = 1.0 - 6.0 * Ku / (mu0 * Ms**2)
    return np.sqrt(15.0) * exchange_length(A, Ms, mu0=mu0) / np.sqrt(denominator)


def remanent_nucleation_radius(Ku, Ms, A, mu0=MU_0):
    """Return the remanent vortex-nucleation radius ``R_nuc^rem``."""
    return vortex_nucleation_radius(Ku, Ms, A, mu0=mu0)


def coercive_nucleation_radius(Ms, A, mu0=MU_0):
    """Return the critical radius ``R_nuc^c = sqrt(15) l_ex``.

    At this radius the descending coercive field first departs from the
    coherent Stoner--Wohlfarth value.  Unlike the remanent nucleation radius,
    it is independent of the uniaxial anisotropy constant ``Ku``.
    """
    if Ms <= 0.0:
        raise ValueError("Ms must be positive.")
    if A <= 0.0:
        raise ValueError("A must be positive.")
    if mu0 <= 0.0:
        raise ValueError("mu0 must be positive.")
    return np.sqrt(15.0) * exchange_length(A, Ms, mu0=mu0)


@lru_cache(maxsize=16)
def _remanent_reorientation_parameter_cached(
    n_quad,
    tolerance,
    max_iterations,
):
    lower = 0.0
    upper = 4.0

    def residual(value):
        return g_u_z(value, n_quad=n_quad) - 1.0 / 3.0

    lower_residual = residual(lower)
    upper_residual = residual(upper)
    while upper_residual > 0.0 and upper < 256.0:
        upper *= 2.0
        upper_residual = residual(upper)
    if lower_residual < 0.0 or upper_residual > 0.0:
        raise RuntimeError("Could not bracket the remanent reorientation root.")

    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_residual = residual(midpoint)
        if midpoint_residual > 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance:
            break
    return 0.5 * (lower + upper)


def remanent_reorientation_parameter(
    *,
    n_quad=360,
    tolerance=1.0e-12,
    max_iterations=80,
):
    """Return the universal profile value ``nu_*`` at remanent reorientation.

    The angular curvature of the axial hyperbolic vortex vanishes when
    ``g_u_z(nu_*) = g_u_x(nu_*) = 1/3``.  The identity
    ``g_u_x = (1 - g_u_z)/2`` reduces this condition to one scalar root.
    """
    return _remanent_reorientation_parameter_cached(
        int(n_quad),
        float(tolerance),
        int(max_iterations),
    )


def _remanent_orientation_key(orientation):
    normalized = str(orientation).strip().lower()
    aliases = {
        "axial": "z",
        "z": "z",
        "transverse": "x",
        "x": "x",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(
            "orientation must be 'axial'/'z' or 'transverse'/'x'"
        ) from error


def _validate_remanent_material_parameters(Ku, Ms, A, mu0):
    if Ku < 0.0:
        raise ValueError("Ku must be non-negative.")
    if Ms <= 0.0:
        raise ValueError("Ms must be positive.")
    if A <= 0.0:
        raise ValueError("A must be positive.")
    if mu0 <= 0.0:
        raise ValueError("mu0 must be positive.")


def remanent_radius_from_vortex_parameter(
    nu,
    Ku,
    Ms,
    A,
    *,
    orientation="axial",
    mu0=MU_0,
    n_quad=360,
    l_max=161,
    n_mu=None,
):
    """Evaluate the stationary remanent radius ``R_alpha(nu)``.

    This is Eq. (61) of the Supplementary Material.  ``orientation`` selects
    the axial (``z``) or transverse (``x``) anisotropy profile.  At ``nu=0``
    the ratio of first derivatives is evaluated by its second-derivative
    limit; on the axial branch this equals the remanent nucleation radius.
    Non-real portions of the parametric stationary branch are returned as
    ``nan``.
    """
    _validate_remanent_material_parameters(Ku, Ms, A, mu0)
    orientation_key = _remanent_orientation_key(orientation)
    nu_array = np.asarray(nu, dtype=float)
    scalar_input = nu_array.ndim == 0
    if np.any(nu_array < 0.0):
        raise ValueError("nu must be non-negative.")

    derivatives = profile_derivatives(
        nu_array,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )
    anisotropy_key = f"g_u_{orientation_key}"
    numerator = np.asarray(derivatives["g_ex_d1"])
    denominator = 2.0 * (
        np.asarray(derivatives["g_dem_d1"])
        + Ku / (mu0 * Ms**2)
        * np.asarray(derivatives[f"{anisotropy_key}_d1"])
    )

    zero = nu_array == 0.0
    if np.any(zero):
        numerator = np.where(zero, derivatives["g_ex_d2"], numerator)
        denominator = np.where(
            zero,
            2.0
            * (
                np.asarray(derivatives["g_dem_d2"])
                + Ku / (mu0 * Ms**2)
                * np.asarray(derivatives[f"{anisotropy_key}_d2"])
            ),
            denominator,
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        radius = exchange_length(A, Ms, mu0=mu0) * np.sqrt(
            numerator / denominator
        )
    if scalar_input:
        return float(radius)
    return radius


def remanent_profile_curvature(
    nu,
    Ku,
    Ms,
    A,
    *,
    orientation="axial",
    radius=None,
    mu0=MU_0,
    n_quad=360,
    l_max=161,
    n_mu=None,
):
    """Return ``d²H'/dnu²`` on a stationary remanent profile branch."""
    _validate_remanent_material_parameters(Ku, Ms, A, mu0)
    orientation_key = _remanent_orientation_key(orientation)
    nu_array = np.asarray(nu, dtype=float)
    scalar_input = nu_array.ndim == 0
    if radius is None:
        radius = remanent_radius_from_vortex_parameter(
            nu_array,
            Ku,
            Ms,
            A,
            orientation=orientation_key,
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
        )
    radius = np.asarray(radius, dtype=float)
    derivatives = profile_derivatives(
        nu_array,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )
    curvature = (
        np.asarray(derivatives["g_ex_d2"])
        - Ku * radius**2 / A
        * np.asarray(derivatives[f"g_u_{orientation_key}_d2"])
        - mu0 * Ms**2 * radius**2 / A
        * np.asarray(derivatives["g_dem_d2"])
    )
    if scalar_input:
        return float(curvature)
    return curvature


def remanent_reorientation_radius(
    Ku,
    Ms,
    A,
    *,
    mu0=MU_0,
    n_quad=360,
    l_max=161,
    n_mu=None,
):
    """Return the axial-vortex reorientation radius ``R_*^rem``."""
    nu_star = remanent_reorientation_parameter(n_quad=n_quad)
    return remanent_radius_from_vortex_parameter(
        nu_star,
        Ku,
        Ms,
        A,
        orientation="axial",
        mu0=mu0,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )


@lru_cache(maxsize=16)
def _critical_reduced_anisotropy_cached(n_quad, l_max, n_mu):
    nu_star = remanent_reorientation_parameter(n_quad=n_quad)
    derivatives = profile_derivatives(
        nu_star,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )
    numerator = (
        derivatives["g_ex_d1"] - 30.0 * derivatives["g_dem_d1"]
    )
    denominator = (
        6.0 * derivatives["g_ex_d1"]
        + 30.0 * derivatives["g_u_z_d1"]
    )
    return float(np.sqrt(numerator / denominator))


def critical_reduced_anisotropy_for_two_stage_remanence(
    *,
    n_quad=360,
    l_max=161,
    n_mu=None,
):
    """Return the upper ``kappa_u`` for separate nucleation and reorientation."""
    return _critical_reduced_anisotropy_cached(
        int(n_quad),
        int(l_max),
        None if n_mu is None else int(n_mu),
    )


def _invert_monotonic_remanent_radius(
    target_radius,
    Ku,
    Ms,
    A,
    *,
    orientation,
    lower_nu,
    upper_nu,
    mu0,
    n_quad,
    l_max,
    n_mu,
    tolerance,
    max_iterations,
):
    target = np.asarray(target_radius, dtype=float).reshape(-1)
    if target.size == 0:
        return target

    upper = float(upper_nu)
    target_max = float(np.max(target))
    while True:
        upper_radius = remanent_radius_from_vortex_parameter(
            upper,
            Ku,
            Ms,
            A,
            orientation=orientation,
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
        )
        if np.isfinite(upper_radius) and upper_radius >= target_max:
            break
        upper *= 2.0
        if upper > 256.0:
            raise RuntimeError(
                "Could not bracket the requested remanent radius below nu=256."
            )

    lower_values = np.full_like(target, float(lower_nu))
    upper_values = np.full_like(target, upper)
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower_values + upper_values)
        midpoint_radius = remanent_radius_from_vortex_parameter(
            midpoint,
            Ku,
            Ms,
            A,
            orientation=orientation,
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
        )
        below = midpoint_radius < target
        lower_values = np.where(below, midpoint, lower_values)
        upper_values = np.where(below, upper_values, midpoint)
        if np.max(upper_values - lower_values) <= tolerance:
            break
    return 0.5 * (lower_values + upper_values)


def remanent_transverse_jump_parameter(
    Ku,
    Ms,
    A,
    *,
    mu0=MU_0,
    n_quad=360,
    l_max=161,
    n_mu=None,
    tolerance=1.0e-10,
    max_iterations=60,
):
    """Return the transverse right limit ``nu_perp,*`` at ``R_*^rem``."""
    _validate_remanent_material_parameters(Ku, Ms, A, mu0)
    kappa_u = np.sqrt(Ku / (mu0 * Ms**2))
    kappa_critical = critical_reduced_anisotropy_for_two_stage_remanence(
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )
    if kappa_u >= kappa_critical:
        raise ValueError(
            "The transverse jump belongs to the two-stage regime "
            f"kappa_u < {kappa_critical:.8g}; received {kappa_u:.8g}."
        )

    nu_star = remanent_reorientation_parameter(n_quad=n_quad)
    reorientation_radius = remanent_radius_from_vortex_parameter(
        nu_star,
        Ku,
        Ms,
        A,
        orientation="axial",
        mu0=mu0,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )
    result = _invert_monotonic_remanent_radius(
        np.array([reorientation_radius]),
        Ku,
        Ms,
        A,
        orientation="transverse",
        lower_nu=nu_star,
        upper_nu=max(4.0, 1.5 * nu_star),
        mu0=mu0,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    curvature = remanent_profile_curvature(
        result[0],
        Ku,
        Ms,
        A,
        orientation="transverse",
        radius=reorientation_radius,
        mu0=mu0,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )
    if curvature <= 0.0:
        raise RuntimeError("The transverse jump state is not locally stable.")
    return float(result[0])


def remanent_vortex_parameter(
    R,
    Ku,
    Ms,
    A,
    *,
    mu0=MU_0,
    n_quad=360,
    l_max=161,
    n_mu=None,
    tolerance=1.0e-10,
    max_iterations=60,
):
    """Return the field-connected remanent profile coordinate ``nu_rem(R)``.

    This implements the piecewise inverse-function relation in Eq. (64) of
    the Supplementary Material.  It applies to the two-stage regime
    ``kappa_u < kappa_u,c``.  At ``R = R_*^rem`` the axial limiting value is
    returned; immediately above it the transverse inverse branch produces
    the finite reoptimization jump.
    """
    _validate_remanent_material_parameters(Ku, Ms, A, mu0)
    radius = np.asarray(R, dtype=float)
    scalar_input = radius.ndim == 0
    if np.any(radius <= 0.0):
        raise ValueError("R must be positive.")

    kappa_u = np.sqrt(Ku / (mu0 * Ms**2))
    kappa_critical = critical_reduced_anisotropy_for_two_stage_remanence(
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )
    if kappa_u >= kappa_critical:
        raise ValueError(
            "The piecewise remanent inverse requires the two-stage regime "
            f"kappa_u < {kappa_critical:.8g}; received {kappa_u:.8g}."
        )

    nu_star = remanent_reorientation_parameter(n_quad=n_quad)
    nucleation_radius = vortex_nucleation_radius(Ku, Ms, A, mu0=mu0)
    reorientation_radius = remanent_radius_from_vortex_parameter(
        nu_star,
        Ku,
        Ms,
        A,
        orientation="axial",
        mu0=mu0,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )

    flat_radius = radius.reshape(-1)
    result = np.zeros_like(flat_radius)
    axial = (
        (flat_radius > nucleation_radius)
        & (flat_radius <= reorientation_radius)
    )
    transverse = flat_radius > reorientation_radius
    if np.any(axial):
        result[axial] = _invert_monotonic_remanent_radius(
            flat_radius[axial],
            Ku,
            Ms,
            A,
            orientation="axial",
            lower_nu=0.0,
            upper_nu=nu_star,
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if np.any(transverse):
        result[transverse] = _invert_monotonic_remanent_radius(
            flat_radius[transverse],
            Ku,
            Ms,
            A,
            orientation="transverse",
            lower_nu=nu_star,
            upper_nu=max(4.0, 1.5 * nu_star),
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    at_reorientation = np.isclose(
        flat_radius,
        reorientation_radius,
        rtol=1.0e-13,
        atol=0.0,
    )
    result[at_reorientation] = nu_star

    if np.any(axial):
        curvature = remanent_profile_curvature(
            result[axial],
            Ku,
            Ms,
            A,
            orientation="axial",
            radius=flat_radius[axial],
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
        )
        if np.any(curvature < -1.0e-8):
            raise RuntimeError("The inverted axial branch is not locally stable.")
    if np.any(transverse):
        curvature = remanent_profile_curvature(
            result[transverse],
            Ku,
            Ms,
            A,
            orientation="transverse",
            radius=flat_radius[transverse],
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
        )
        if np.any(curvature < -1.0e-8):
            raise RuntimeError(
                "The inverted transverse branch is not locally stable."
            )

    result = result.reshape(radius.shape)
    if scalar_input:
        return float(result)
    return result


def remanent_magnetization_ratio(
    R,
    Ku,
    Ms,
    A,
    **kwargs,
):
    """Return ``|<m_z>|_rem`` from the piecewise relation in Eq. (65)."""
    radius = np.asarray(R, dtype=float)
    scalar_input = radius.ndim == 0
    nu_rem = np.asarray(
        remanent_vortex_parameter(R, Ku, Ms, A, **kwargs),
        dtype=float,
    )
    mu0 = kwargs.get("mu0", MU_0)
    n_quad = kwargs.get("n_quad", 360)
    nucleation_radius = vortex_nucleation_radius(Ku, Ms, A, mu0=mu0)
    reorientation_radius = remanent_reorientation_radius(
        Ku,
        Ms,
        A,
        mu0=mu0,
        n_quad=n_quad,
        l_max=kwargs.get("l_max", 161),
        n_mu=kwargs.get("n_mu"),
    )

    result = np.zeros_like(radius)
    uniform = radius <= nucleation_radius
    axial = (radius > nucleation_radius) & (radius <= reorientation_radius)
    result[uniform] = 1.0
    if np.any(axial):
        result[axial] = g_z_z(nu_rem[axial], n_quad=n_quad)
    if scalar_input:
        return float(result)
    return result


def _validate_coercive_vortex_parameter(nu, *, n_quad):
    nu_array = np.asarray(nu, dtype=float)
    if np.any(nu_array < 0.0):
        raise ValueError("nu must be non-negative.")
    nu_star = remanent_reorientation_parameter(n_quad=n_quad)
    if np.any(nu_array > nu_star * (1.0 + 1.0e-12)):
        raise ValueError(
            "The axial coercive spinodal is defined only for "
            f"0 <= nu <= nu_* = {nu_star:.10g}."
        )
    return nu_array, nu_star


def coercive_field_from_vortex_parameter(
    nu,
    Ku,
    Ms,
    *,
    n_quad=360,
):
    """Return the positive coercive-field magnitude ``B_c(nu)`` in tesla.

    The corresponding spinodal on the descending branch has the signed field
    ``-B_c``.  The expression follows from vanishing angular curvature of the
    axial hyperbolic vortex.
    """
    if Ku < 0.0:
        raise ValueError("Ku must be non-negative.")
    if Ms <= 0.0:
        raise ValueError("Ms must be positive.")
    nu_array, _ = _validate_coercive_vortex_parameter(
        nu,
        n_quad=n_quad,
    )
    scalar_input = nu_array.ndim == 0
    _, g_u_x_values, g_u_z_values, g_z_z_values = local_profiles(
        nu_array,
        n_quad=n_quad,
    )
    delta_g_u = np.asarray(g_u_z_values) - np.asarray(g_u_x_values)
    magnitude = (
        2.0
        * Ku
        / Ms
        * np.abs(delta_g_u / np.asarray(g_z_z_values))
    )
    if scalar_input:
        return float(magnitude)
    return magnitude


def coercive_radius_from_vortex_parameter(
    nu,
    Ku,
    Ms,
    A,
    *,
    mu0=MU_0,
    n_quad=360,
    l_max=161,
    n_mu=None,
):
    """Evaluate the coercive-spinodal radius ``R_c(nu)``.

    The result parametrizes the nonuniform coercive branch for
    ``0 <= nu <= nu_*``.  At ``nu=0`` the formally indeterminate ratio of
    first derivatives is evaluated through its exact second-derivative limit.
    """
    _validate_remanent_material_parameters(Ku, Ms, A, mu0)
    nu_array, _ = _validate_coercive_vortex_parameter(
        nu,
        n_quad=n_quad,
    )
    scalar_input = nu_array.ndim == 0
    _, g_u_x_values, g_u_z_values, g_z_z_values = local_profiles(
        nu_array,
        n_quad=n_quad,
    )
    derivatives = profile_derivatives(
        nu_array,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )
    delta_over_gzz = (
        (np.asarray(g_u_z_values) - np.asarray(g_u_x_values))
        / np.asarray(g_z_z_values)
    )
    kappa_u_squared = Ku / (mu0 * Ms**2)
    numerator = np.asarray(derivatives["g_ex_d1"])
    denominator_core = (
        np.asarray(derivatives["g_dem_d1"])
        + kappa_u_squared * np.asarray(derivatives["g_u_z_d1"])
        - 2.0
        * kappa_u_squared
        * delta_over_gzz
        * np.asarray(derivatives["g_z_z_d1"])
    )

    zero = nu_array == 0.0
    if np.any(zero):
        numerator = np.where(zero, derivatives["g_ex_d2"], numerator)
        denominator_core = np.where(
            zero,
            np.asarray(derivatives["g_dem_d2"])
            + kappa_u_squared * np.asarray(derivatives["g_u_z_d2"])
            - 2.0
            * kappa_u_squared
            * delta_over_gzz
            * np.asarray(derivatives["g_z_z_d2"]),
            denominator_core,
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        radius = exchange_length(A, Ms, mu0=mu0) * np.sqrt(
            numerator / (2.0 * denominator_core)
        )
    if scalar_input:
        return float(radius)
    return radius


def coercive_profile_curvature(
    nu,
    Ku,
    Ms,
    A,
    *,
    radius=None,
    mu0=MU_0,
    n_quad=360,
    l_max=161,
    n_mu=None,
):
    """Return ``d²H'/dnu²`` at the descending coercive spinodal."""
    _validate_remanent_material_parameters(Ku, Ms, A, mu0)
    nu_array, _ = _validate_coercive_vortex_parameter(
        nu,
        n_quad=n_quad,
    )
    scalar_input = nu_array.ndim == 0
    if radius is None:
        radius = coercive_radius_from_vortex_parameter(
            nu_array,
            Ku,
            Ms,
            A,
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
        )
    radius = np.asarray(radius, dtype=float)
    descending_field = -np.asarray(
        coercive_field_from_vortex_parameter(
            nu_array,
            Ku,
            Ms,
            n_quad=n_quad,
        )
    )
    derivatives = profile_derivatives(
        nu_array,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )
    curvature = (
        np.asarray(derivatives["g_ex_d2"])
        - Ku * radius**2 / A * np.asarray(derivatives["g_u_z_d2"])
        - Ms
        * descending_field
        * radius**2
        / A
        * np.asarray(derivatives["g_z_z_d2"])
        - mu0
        * Ms**2
        * radius**2
        / A
        * np.asarray(derivatives["g_dem_d2"])
    )
    if scalar_input:
        return float(curvature)
    return curvature


def _invert_monotonic_coercive_radius(
    target_radius,
    Ku,
    Ms,
    A,
    *,
    upper_nu,
    mu0,
    n_quad,
    l_max,
    n_mu,
    tolerance,
    max_iterations,
):
    target = np.asarray(target_radius, dtype=float).reshape(-1)
    lower_values = np.zeros_like(target)
    upper_values = np.full_like(target, float(upper_nu))
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower_values + upper_values)
        midpoint_radius = coercive_radius_from_vortex_parameter(
            midpoint,
            Ku,
            Ms,
            A,
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
        )
        below = midpoint_radius < target
        lower_values = np.where(below, midpoint, lower_values)
        upper_values = np.where(below, upper_values, midpoint)
        if np.max(upper_values - lower_values, initial=0.0) <= tolerance:
            break
    return 0.5 * (lower_values + upper_values)


def coercive_vortex_parameter(
    R,
    Ku,
    Ms,
    A,
    *,
    mu0=MU_0,
    n_quad=360,
    l_max=161,
    n_mu=None,
    tolerance=1.0e-10,
    max_iterations=60,
):
    """Return the limiting profile coordinate ``nu_c(R)`` at coercivity.

    Below ``R_nuc^c`` the coercive state is uniform.  Between ``R_nuc^c``
    and ``R_*^rem`` the value is obtained by inverting the axial coercive
    spinodal.  Above ``R_*^rem`` coercivity is zero and the limiting profile
    is the transverse remanent profile.
    """
    _validate_remanent_material_parameters(Ku, Ms, A, mu0)
    radius = np.asarray(R, dtype=float)
    scalar_input = radius.ndim == 0
    if np.any(radius <= 0.0):
        raise ValueError("R must be positive.")

    kappa_u = np.sqrt(Ku / (mu0 * Ms**2))
    kappa_critical = critical_reduced_anisotropy_for_two_stage_remanence(
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )
    if kappa_u >= kappa_critical:
        raise ValueError(
            "The piecewise coercive inverse requires the two-stage regime "
            f"kappa_u < {kappa_critical:.8g}; received {kappa_u:.8g}."
        )

    nu_star = remanent_reorientation_parameter(n_quad=n_quad)
    nucleation_radius = coercive_nucleation_radius(Ms, A, mu0=mu0)
    reorientation_radius = remanent_reorientation_radius(
        Ku,
        Ms,
        A,
        mu0=mu0,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )

    flat_radius = radius.reshape(-1)
    result = np.zeros_like(flat_radius)
    axial = (
        (flat_radius > nucleation_radius)
        & (flat_radius <= reorientation_radius)
    )
    transverse = flat_radius > reorientation_radius
    if np.any(axial):
        result[axial] = _invert_monotonic_coercive_radius(
            flat_radius[axial],
            Ku,
            Ms,
            A,
            upper_nu=nu_star,
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if np.any(transverse):
        result[transverse] = remanent_vortex_parameter(
            flat_radius[transverse],
            Ku,
            Ms,
            A,
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    at_reorientation = np.isclose(
        flat_radius,
        reorientation_radius,
        rtol=1.0e-13,
        atol=0.0,
    )
    result[at_reorientation] = nu_star

    if np.any(axial):
        curvature = coercive_profile_curvature(
            result[axial],
            Ku,
            Ms,
            A,
            radius=flat_radius[axial],
            mu0=mu0,
            n_quad=n_quad,
            l_max=l_max,
            n_mu=n_mu,
        )
        if np.any(curvature < -1.0e-8):
            raise RuntimeError("The inverted coercive branch is not stable in nu.")

    result = result.reshape(radius.shape)
    if scalar_input:
        return float(result)
    return result


def hyperbolic_vortex_coercive_field(
    R,
    Ku,
    Ms,
    A,
    **kwargs,
):
    """Return the analytic positive coercive-field magnitude in tesla.

    This is the radius-dependent ``mu0 H_c`` of the reduced hyperbolic-vortex
    model.  It equals the Stoner--Wohlfarth value below ``R_nuc^c``, follows
    the finite-profile angular spinodal up to ``R_*^rem``, and vanishes for
    the reversible transverse branch above that radius.
    """
    radius = np.asarray(R, dtype=float)
    scalar_input = radius.ndim == 0
    mu0 = kwargs.get("mu0", MU_0)
    n_quad = kwargs.get("n_quad", 360)
    l_max = kwargs.get("l_max", 161)
    n_mu = kwargs.get("n_mu")
    flat_radius = radius.reshape(-1)
    nu_c = np.asarray(
        coercive_vortex_parameter(R, Ku, Ms, A, **kwargs),
        dtype=float,
    ).reshape(-1)
    nucleation_radius = coercive_nucleation_radius(Ms, A, mu0=mu0)
    reorientation_radius = remanent_reorientation_radius(
        Ku,
        Ms,
        A,
        mu0=mu0,
        n_quad=n_quad,
        l_max=l_max,
        n_mu=n_mu,
    )

    result = np.zeros_like(flat_radius)
    uniform = flat_radius <= nucleation_radius
    nonuniform = (
        (flat_radius > nucleation_radius)
        & (flat_radius < reorientation_radius)
    )
    result[uniform] = 2.0 * Ku / Ms
    if np.any(nonuniform):
        result[nonuniform] = coercive_field_from_vortex_parameter(
            nu_c[nonuniform],
            Ku,
            Ms,
            n_quad=n_quad,
        )

    result = result.reshape(radius.shape)
    if scalar_input:
        return float(result)
    return result


def stoner_wohlfarth_switching_field(theta_rad, anisotropy_field_T):
    """Stoner-Wohlfarth switching field from the astroid."""
    theta_rad = np.asarray(theta_rad, dtype=float)
    return anisotropy_field_T * (
        np.abs(np.sin(theta_rad)) ** (2.0 / 3.0)
        + np.abs(np.cos(theta_rad)) ** (2.0 / 3.0)
    ) ** (-1.5)


def stoner_wohlfarth_coercive_field(theta_rad, anisotropy_field_T):
    """Projected-magnetization zero crossing for a Stoner-Wohlfarth loop."""
    theta_rad = np.asarray(theta_rad, dtype=float)
    switching = stoner_wohlfarth_switching_field(theta_rad, anisotropy_field_T)
    zero_crossing = 0.5 * anisotropy_field_T * np.abs(np.sin(2.0 * theta_rad))
    return np.where(np.abs(theta_rad) <= np.pi / 4.0, switching, zero_crossing)


def stoner_wohlfarth_ensemble_remanence(Ms=1.0):
    """Return the exact remanence of an isotropic Stoner-Wohlfarth ensemble.

    At zero field, the moment of every saturated particle relaxes onto the
    closest direction of its uniaxial easy axis.  Isotropic orientation
    averaging therefore gives ``M_r / M_s = 1/2``.
    """
    return STONER_WOHLFARTH_ENSEMBLE_REMANENCE_RATIO * np.asarray(Ms)


def stoner_wohlfarth_ensemble_coercive_field(anisotropy_field_T):
    """Return the positive coercive-field magnitude of an isotropic ensemble.

    The ratio ``B_c / B_K = 0.48221204600969`` is the converged numerical zero
    of the exact orientation-averaged quasistatic Stoner-Wohlfarth branch.  In
    contrast to the remanence ratio, it is not known here in closed form.
    """
    return (
        STONER_WOHLFARTH_ENSEMBLE_COERCIVE_RATIO
        * np.asarray(anisotropy_field_T)
    )


def stoner_wohlfarth_astroid(psi_rad):
    """Parametric Stoner-Wohlfarth astroid in reduced field components."""
    psi_rad = np.asarray(psi_rad, dtype=float)
    return -np.cos(psi_rad) ** 3, np.sin(psi_rad) ** 3
