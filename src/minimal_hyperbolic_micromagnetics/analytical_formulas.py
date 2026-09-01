"""Analytic formulas and standard reference values for the reduced model."""

from __future__ import annotations

import numpy as np

from .profiles import MU_0


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
    """Analytic vortex-nucleation field for the reduced H'' model."""
    return mu0 * Ms / 3.0 - 2.0 * Ku / Ms - 10.0 * A / (Ms * np.asarray(R) ** 2)


def critical_anisotropy_for_zero_nucleation_field(Ms, A, R, mu0=MU_0):
    """Return Ku for which the analytic vortex-nucleation field is zero.
    """
    lex = exchange_length(A, Ms, mu0=mu0)
    return mu0 * Ms**2 / 6.0 * (1.0 - 15.0 * lex**2 / np.asarray(R) ** 2)


def vortex_nucleation_radius(Ku, Ms, A, mu0=MU_0):
    """Critical radius where the analytic vortex-nucleation field is zero."""
    denominator = 1.0 - 6.0 * Ku / (mu0 * Ms**2)
    return np.sqrt(15.0) * exchange_length(A, Ms, mu0=mu0) / np.sqrt(denominator)


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
