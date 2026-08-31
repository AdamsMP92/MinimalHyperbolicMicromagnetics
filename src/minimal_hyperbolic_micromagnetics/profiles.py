"""Optimized profile functions for the hyperbolic vortex in a sphere."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MU_0 = 4.0 * np.pi * 1e-7


@dataclass(frozen=True)
class ProfileComputation:
    """Numerical settings for profile tabulation."""

    nu_min: float = 0.0
    nu_max: float = 20.0
    n_nu: int = 2000
    n_quad: int = 360
    chunk_size: int = 256
    l_max_demag: int = 161
    n_mu_demag: int | None = None

    def nu_grid(self) -> np.ndarray:
        return np.linspace(self.nu_min, self.nu_max, self.n_nu)


def _sech(x):
    return 1.0 / np.cosh(x)


def _gauss_legendre_interval(n, a, b):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    midpoint = 0.5 * (a + b)
    half_width = 0.5 * (b - a)
    return midpoint + half_width * nodes, half_width * weights


def _as_array(nu):
    nu_arr = np.asarray(nu, dtype=float)
    return nu_arr, nu_arr.ndim == 0, nu_arr.reshape(-1)


def _restore_shape(values, nu_arr, scalar_input):
    if scalar_input:
        return float(values[0])
    return values.reshape(nu_arr.shape)


def local_profiles(nu, n_quad=360, chunk_size=256):
    """Return g_ex, g_u_x, g_u_z, and g_z_z using one-dimensional integrals."""
    nu_arr, scalar_input, nu_flat = _as_array(nu)

    x, w = _gauss_legendre_interval(n_quad, 0.0, 1.0)
    sqrt_weight = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    volume_weight = x * sqrt_weight

    gex = np.empty_like(nu_flat)
    gux = np.empty_like(nu_flat)
    guz = np.empty_like(nu_flat)
    gzz = np.empty_like(nu_flat)

    for start in range(0, len(nu_flat), chunk_size):
        stop = min(start + chunk_size, len(nu_flat))
        nu_chunk = nu_flat[start:stop, None]
        psi = nu_chunk * x[None, :]
        sech_psi = _sech(psi)
        tanh_psi = np.tanh(psi)

        guz[start:stop] = 3.0 * np.sum(
            w[None, :] * volume_weight[None, :] * sech_psi**2,
            axis=1,
        )
        gzz[start:stop] = 3.0 * np.sum(
            w[None, :] * volume_weight[None, :] * sech_psi,
            axis=1,
        )
        gux[start:stop] = 4.0 / 3.0 * (1.0 - guz[start:stop])

        winding_term = np.divide(
            tanh_psi**2,
            x[None, :],
            out=np.zeros_like(tanh_psi),
            where=x[None, :] > 0.0,
        )
        exchange_integrand = sqrt_weight[None, :] * (
            nu_chunk**2 * x[None, :] * sech_psi**2 + winding_term
        )
        gex[start:stop] = 3.0 * np.sum(w[None, :] * exchange_integrand, axis=1)

    zero_mask = nu_flat == 0.0
    if np.any(zero_mask):
        gex[zero_mask] = 0.0
        gux[zero_mask] = 0.0
        guz[zero_mask] = 1.0
        gzz[zero_mask] = 1.0

    if scalar_input:
        return tuple(float(v[0]) for v in (gex, gux, guz, gzz))

    return (
        gex.reshape(nu_arr.shape),
        gux.reshape(nu_arr.shape),
        guz.reshape(nu_arr.shape),
        gzz.reshape(nu_arr.shape),
    )


def g_ex(nu, n_quad=360, chunk_size=256):
    return local_profiles(nu, n_quad=n_quad, chunk_size=chunk_size)[0]


def g_u_x(nu, n_quad=360, chunk_size=256):
    return local_profiles(nu, n_quad=n_quad, chunk_size=chunk_size)[1]


def g_u_z(nu, n_quad=360, chunk_size=256):
    return local_profiles(nu, n_quad=n_quad, chunk_size=chunk_size)[2]


def g_z_z(nu, n_quad=360, chunk_size=256):
    return local_profiles(nu, n_quad=n_quad, chunk_size=chunk_size)[3]


def g_hd_positive(nu, l_max=161, n_mu=None):
    """Positive demagnetizing-field coefficient from spherical harmonics."""
    if l_max < 1:
        raise ValueError("l_max must be at least 1")
    if l_max % 2 == 0:
        l_max -= 1
    if n_mu is None:
        n_mu = max(360, 4 * (l_max + 1))

    nu_arr, scalar_input, nu_flat = _as_array(nu)

    x, weights = _gauss_legendre_interval(n_mu, 0.0, 1.0)
    mu_positive = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    legendre_values = np.polynomial.legendre.legvander(mu_positive, l_max)
    odd_l = np.arange(1, l_max + 1, 2)
    p_odd = legendre_values[:, odd_l]

    out = np.empty_like(nu_flat)
    for i, nu_i in enumerate(nu_flat):
        if nu_i == 0.0:
            out[i] = 1.0 / 6.0
        else:
            moments = (weights * x * _sech(nu_i * x)) @ p_odd
            out[i] = 1.5 * np.sum(moments * moments)

    return _restore_shape(out, nu_arr, scalar_input)


def g_s(nu, l_max=161, n_mu=None):
    """Coefficient for E_s = -1/2 int M.B dV."""
    return g_hd_positive(nu, l_max=l_max, n_mu=n_mu) - 0.5


def g_dem(nu, l_max=161, n_mu=None):
    """Positive paper/Hamiltonian demagnetizing profile: g_dem = -g_s."""
    return -g_s(nu, l_max=l_max, n_mu=n_mu)


def all_profiles(nu, n_quad=360, chunk_size=256, l_max=161, n_mu=None):
    """Return a dict with all optimized profile functions."""
    gex, gux, guz, gzz = local_profiles(
        nu,
        n_quad=n_quad,
        chunk_size=chunk_size,
    )
    gh = g_hd_positive(nu, l_max=l_max, n_mu=n_mu)
    gs_b = gh - 0.5
    gdem = -gs_b

    return {
        "g_ex": gex,
        "g_u_x": gux,
        "g_u_z": guz,
        "g_z_z": gzz,
        "g_dem": gdem,
        "g_dem_spherical_harmonics": gdem,
        "g_H": gh,
        "g_s_B_energy": gs_b,
    }


def compute_profiles(settings: ProfileComputation | None = None, nu=None):
    """Compute a profile table as a dictionary of NumPy arrays."""
    settings = settings or ProfileComputation()
    nu_grid = settings.nu_grid() if nu is None else np.asarray(nu, dtype=float)
    profiles = all_profiles(
        nu_grid,
        n_quad=settings.n_quad,
        chunk_size=settings.chunk_size,
        l_max=settings.l_max_demag,
        n_mu=settings.n_mu_demag,
    )
    return {"nu": nu_grid, **profiles}


def compute_and_store_profiles(
    output_csv,
    settings: ProfileComputation | None = None,
    *,
    nu=None,
    output_png=None,
    make_plot=False,
):
    """Compute profiles, store them as CSV, and optionally create a profile plot."""
    profiles = compute_profiles(settings, nu=nu)
    dataframe = pd.DataFrame(profiles)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)

    if make_plot or output_png is not None:
        if output_png is None:
            output_png = output_csv.with_suffix(".png")
        from .plotting import plot_profiles

        plot_profiles(dataframe, output_png)

    return dataframe
