"""Closed-form reference formulas for the reduced micromagnetic model."""

from __future__ import annotations

import numpy as np

from .profiles import MU_0


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


def stoner_wohlfarth_astroid(psi_rad):
    """Parametric Stoner-Wohlfarth astroid in reduced field components."""
    psi_rad = np.asarray(psi_rad, dtype=float)
    return -np.cos(psi_rad) ** 3, np.sin(psi_rad) ** 3
