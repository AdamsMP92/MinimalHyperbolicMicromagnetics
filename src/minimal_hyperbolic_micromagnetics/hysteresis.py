"""Hysteresis and Stoner-Wohlfarth checks for the reduced model."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .analytical_formulas import (
    stoner_wohlfarth_astroid,
    stoner_wohlfarth_coercive_field,
    stoner_wohlfarth_switching_field,
    vortex_nucleation_field,
    vortex_nucleation_radius,
)
from .profiles import MU_0, compute_profiles


@dataclass(frozen=True)
class ModelParameters:
    """Material and reduced-model parameters."""

    Ku: float
    Ms: float
    A: float
    R: float
    beta_deg: float = 0.0
    gux_factor: float = 0.0

    @property
    def beta_rad(self) -> float:
        return float(np.deg2rad(self.beta_deg))

    @property
    def anisotropy_field_T(self) -> float:
        return 2.0 * self.Ku / self.Ms

    @property
    def exchange_length_m(self) -> float:
        return float(np.sqrt(2.0 * self.A / (MU_0 * self.Ms**2)))

    @property
    def nucleation_radius_m(self) -> float:
        return float(vortex_nucleation_radius(self.Ku, self.Ms, self.A))

    def nucleation_field_T(self) -> float:
        return float(vortex_nucleation_field(self.Ku, self.Ms, self.A, self.R))


@dataclass(frozen=True)
class HysteresisSettings:
    """Numerical settings for a field-following hysteresis calculation."""

    Bmax: float = 1.0
    n_half: int = 250
    fields: np.ndarray | None = None
    stoner_wohlfarth: bool = False


@dataclass(frozen=True)
class HysteresisResult:
    """Field-following hysteresis result."""

    B_T: np.ndarray
    mz_avg: np.ndarray
    nu_min: np.ndarray
    tau_rad: np.ndarray
    energy: np.ndarray
    elapsed_s: float | None = None
    stability_nu_curvature: np.ndarray | None = None
    stability_mixed_curvature: np.ndarray | None = None
    stability_tau_curvature: np.ndarray | None = None
    stability_eigenvalue_min: np.ndarray | None = None
    stability_eigenvalue_max: np.ndarray | None = None
    uniform_tau_rad: np.ndarray | None = None
    uniform_vortex_curvature: np.ndarray | None = None
    uniform_orientation_curvature: np.ndarray | None = None
    uniform_stability_eigenvalue_min: np.ndarray | None = None
    uniform_stability_eigenvalue_max: np.ndarray | None = None

    def as_dict(self):
        values = {
            "B_T": self.B_T,
            "mz_avg": self.mz_avg,
            "nu_min": self.nu_min,
            "tau_min_rad": self.tau_rad,
            "energy": self.energy,
            "stability_nu_curvature": self.stability_nu_curvature,
            "stability_mixed_curvature": self.stability_mixed_curvature,
            "stability_tau_curvature": self.stability_tau_curvature,
            "stability_eigenvalue_min": self.stability_eigenvalue_min,
            "stability_eigenvalue_max": self.stability_eigenvalue_max,
            "uniform_tau_rad": self.uniform_tau_rad,
            "uniform_vortex_curvature": self.uniform_vortex_curvature,
            "uniform_orientation_curvature": self.uniform_orientation_curvature,
            "uniform_stability_eigenvalue_min": (
                self.uniform_stability_eigenvalue_min
            ),
            "uniform_stability_eigenvalue_max": (
                self.uniform_stability_eigenvalue_max
            ),
        }
        return {name: value for name, value in values.items() if value is not None}


def stoner_wohlfarth_profiles():
    """Profiles for the pure Stoner-Wohlfarth limit at nu=0."""
    return {
        "nu": np.array([0.0]),
        "g_ex": np.array([0.0]),
        "g_u_x": np.array([0.0]),
        "g_u_z": np.array([1.0]),
        "g_z_z": np.array([1.0]),
        "g_dem": np.array([0.0]),
        "g_ex_d1": np.array([0.0]),
        "g_ex_d2": np.array([4.0]),
        "g_u_x_d1": np.array([0.0]),
        "g_u_x_d2": np.array([16.0 / 15.0]),
        "g_u_z_d1": np.array([0.0]),
        "g_u_z_d2": np.array([-4.0 / 5.0]),
        "g_z_z_d1": np.array([0.0]),
        "g_z_z_d2": np.array([-2.0 / 5.0]),
        "g_dem_d1": np.array([0.0]),
        "g_dem_d2": np.array([2.0 / 15.0]),
    }


def _nearest_periodic(values, reference):
    values = np.asarray(values, dtype=float)
    values = np.concatenate([values - 2*np.pi, values, values + 2*np.pi])
    return values[np.argmin((values - reference)**2)]


def _stable_tau_beta0_array(b, curvature_sign, tau_previous, tol=1e-12):
    """Stable stationary tau values for beta=0 with signed curvature."""
    b = np.asarray(b, dtype=float)
    curvature_sign = np.asarray(curvature_sign, dtype=float)
    out = np.empty_like(b)

    for i, (bi, si) in enumerate(zip(b, curvature_sign)):
        candidates = []

        if si * (1.0 + bi) > tol:
            candidates.append(0.0)
        if si * (1.0 - bi) > tol:
            candidates.append(np.pi)

        if abs(bi) < 1.0 - tol and si * (bi * bi - 1.0) > tol:
            tau_mid = np.arccos(-bi)
            candidates.extend([tau_mid, -tau_mid])

        if candidates:
            out[i] = _nearest_periodic(candidates, tau_previous)
        else:
            fallback = [0.0, np.pi]
            if abs(bi) <= 1.0:
                tau_mid = np.arccos(-bi)
                fallback.extend([tau_mid, -tau_mid])
            out[i] = _nearest_periodic(fallback, tau_previous)

    return out


def _stable_tau_general(b, beta, curvature_sign, tau_previous, tol=1e-10):
    """Companion-matrix roots with signed stability filtering."""
    companion = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [
            np.exp(4j * beta),
            2 * b * np.exp(2j * beta),
            0,
            -2 * b * np.exp(2j * beta),
        ],
    ], dtype=complex)

    z = np.linalg.eigvals(companion)
    tau = np.angle(z[np.abs(np.abs(z) - 1.0) < tol])
    tau = np.concatenate([
        tau - 4*np.pi,
        tau - 2*np.pi,
        tau,
        tau + 2*np.pi,
        tau + 4*np.pi,
    ])

    stable = curvature_sign * (
        np.cos(2.0 * (tau - beta)) + b * np.cos(tau)
    ) > 0.0
    tau = tau[stable]

    if len(tau) == 0:
        raise RuntimeError("No stable tau root found on the unit circle.")

    return tau[np.argmin((tau - tau_previous)**2)]


def hamiltonian(params, B, tau, gex, guz, gux, gzz, gdem):
    """Dimensionless reduced Hamiltonian H'' or H' for arrays of profile values."""
    return (
        gex
        - params.Ku * params.R**2 / params.A
        * (guz * np.cos(tau - params.beta_rad)**2
           + params.gux_factor * gux * np.sin(tau - params.beta_rad)**2)
        - params.Ms * B * params.R**2 / params.A * gzz * np.cos(tau)
        - MU_0 * params.Ms**2 * params.R**2 / params.A * gdem
    )


def hamiltonian_hessian(
    params,
    B,
    tau,
    *,
    guz,
    gux,
    gzz,
    gex_d2,
    guz_d1,
    guz_d2,
    gux_d1,
    gux_d2,
    gzz_d1,
    gzz_d2,
    gdem_d2,
):
    """Return the analytic reduced-energy Hessian in ``(nu, tau)``.

    Both coordinates are dimensionless (angles are measured in radians), and
    the energy is normalized by ``(4*pi/3) A R``.  Profile derivatives are
    analytic derivatives with respect to ``nu``.
    """
    anisotropy = params.Ku * params.R**2 / params.A
    zeeman = params.Ms * B * params.R**2 / params.A
    demag = MU_0 * params.Ms**2 * params.R**2 / params.A
    delta = tau - params.beta_rad

    nu_curvature = (
        gex_d2
        - anisotropy
        * (
            guz_d2 * np.cos(delta) ** 2
            + params.gux_factor * gux_d2 * np.sin(delta) ** 2
        )
        - zeeman * gzz_d2 * np.cos(tau)
        - demag * gdem_d2
    )
    mixed_curvature = (
        anisotropy
        * (guz_d1 - params.gux_factor * gux_d1)
        * np.sin(2.0 * delta)
        + zeeman * gzz_d1 * np.sin(tau)
    )
    tau_curvature = (
        2.0
        * anisotropy
        * (guz - params.gux_factor * gux)
        * np.cos(2.0 * delta)
        + zeeman * gzz * np.cos(tau)
    )
    return np.array(
        [
            [nu_curvature, mixed_curvature],
            [mixed_curvature, tau_curvature],
        ],
        dtype=float,
    )


def _profile_derivative_arrays(profiles, nu):
    """Read analytic profile derivatives, with a legacy-table fallback."""
    names = ("g_ex", "g_u_x", "g_u_z", "g_z_z", "g_dem")
    derivatives = {}
    missing = []
    for name in names:
        for order in (1, 2):
            key = f"{name}_d{order}"
            if key in profiles:
                derivatives[key] = np.asarray(profiles[key], dtype=float).copy()
            else:
                missing.append(key)

    if missing:
        can_reconstruct = len(nu) >= 3 and np.all(np.diff(nu) > 0.0)
        for name in names:
            if can_reconstruct:
                values = np.asarray(profiles[name], dtype=float)
                derivatives[f"{name}_d1"] = np.gradient(
                    values, nu, edge_order=2
                )
                derivatives[f"{name}_d2"] = np.gradient(
                    derivatives[f"{name}_d1"], nu, edge_order=2
                )
            else:
                derivatives[f"{name}_d1"] = np.full(len(nu), np.nan)
                derivatives[f"{name}_d2"] = np.full(len(nu), np.nan)

    zero = np.isclose(nu, 0.0, rtol=0.0, atol=1.0e-14)
    if np.any(zero):
        exact_at_zero = {
            "g_ex_d1": 0.0,
            "g_ex_d2": 4.0,
            "g_u_x_d1": 0.0,
            "g_u_x_d2": 16.0 / 15.0,
            "g_u_z_d1": 0.0,
            "g_u_z_d2": -4.0 / 5.0,
            "g_z_z_d1": 0.0,
            "g_z_z_d2": -2.0 / 5.0,
            "g_dem_d1": 0.0,
            "g_dem_d2": 2.0 / 15.0,
        }
        for name, value in exact_at_zero.items():
            derivatives[name][zero] = value
    return derivatives


def _symmetric_eigenvalues(matrix):
    """Return the two analytic eigenvalues of a symmetric 2x2 matrix."""
    mean = 0.5 * (matrix[0, 0] + matrix[1, 1])
    radius = np.hypot(0.5 * (matrix[0, 0] - matrix[1, 1]), matrix[0, 1])
    return mean - radius, mean + radius


def _local_minimum_indices(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 1:
        return np.array([0], dtype=int)

    local = []
    if values[0] <= values[1]:
        local.append(0)
    for i in range(1, n - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            local.append(i)
    if values[-1] <= values[-2]:
        local.append(n - 1)

    if not local:
        return np.array([int(np.argmin(values))], dtype=int)
    return np.asarray(local, dtype=int)


def _follow_nu_minimum(values, previous_idx):
    """Choose the local nu-minimum connected to the previous field step."""
    local = _local_minimum_indices(values)
    distance = np.abs(local - previous_idx)
    nearest = local[distance == distance.min()]
    return int(nearest[np.argmin(values[nearest])])


def make_field_sweep(Bmax, n_half):
    """Return a descending and ascending symmetric field sweep."""
    half = np.linspace(-Bmax, Bmax, n_half)
    return np.concatenate([half[::-1], half])


def _resolve_hysteresis_settings(settings, Bmax, n_half, fields, stoner_wohlfarth):
    if settings is None:
        settings = HysteresisSettings()

    return HysteresisSettings(
        Bmax=settings.Bmax if Bmax is None else Bmax,
        n_half=settings.n_half if n_half is None else n_half,
        fields=settings.fields if fields is None else fields,
        stoner_wohlfarth=(
            settings.stoner_wohlfarth
            if stoner_wohlfarth is None
            else stoner_wohlfarth
        ),
    )


def run_hysteresis(
    params: ModelParameters,
    profiles=None,
    *,
    settings: HysteresisSettings | None = None,
    Bmax: float | None = None,
    n_half: int | None = None,
    fields: np.ndarray | None = None,
    stoner_wohlfarth: bool | None = None,
    verbose: bool = False,
) -> HysteresisResult:
    """Compute a field-following hysteresis loop.

    If ``stoner_wohlfarth`` is true, only the nu=0 profile is retained.
    Otherwise, ``profiles`` may be a precomputed dict from ``compute_profiles``;
    if omitted, the default 2000-point profile grid is computed.
    """
    settings = _resolve_hysteresis_settings(
        settings,
        Bmax,
        n_half,
        fields,
        stoner_wohlfarth,
    )

    if settings.stoner_wohlfarth:
        profiles = stoner_wohlfarth_profiles()
    elif profiles is None:
        profiles = compute_profiles()

    fields = (
        make_field_sweep(settings.Bmax, settings.n_half)
        if settings.fields is None
        else np.asarray(settings.fields, dtype=float)
    )

    nu_all = np.asarray(profiles["nu"], dtype=float)
    profile_derivatives = _profile_derivative_arrays(profiles, nu_all)
    if settings.stoner_wohlfarth:
        idx_map = np.array([np.argmin(np.abs(nu_all))])
    else:
        idx_map = np.arange(len(nu_all))
    nu = nu_all[idx_map]

    gex = np.asarray(profiles["g_ex"], dtype=float)[idx_map]
    guz = np.asarray(profiles["g_u_z"], dtype=float)[idx_map]
    gux = np.asarray(profiles["g_u_x"], dtype=float)[idx_map]
    gzz = np.asarray(profiles["g_z_z"], dtype=float)[idx_map]
    gdem = np.asarray(profiles["g_dem"], dtype=float)[idx_map]
    derivative_values = {
        name: values[idx_map]
        for name, values in profile_derivatives.items()
    }

    nu_min = np.zeros(len(fields))
    tau_min = np.zeros(len(fields))
    mz_avg = np.zeros(len(fields))
    energy = np.zeros(len(fields))
    stability_nu_curvature = np.zeros(len(fields))
    stability_mixed_curvature = np.zeros(len(fields))
    stability_tau_curvature = np.zeros(len(fields))
    stability_eigenvalue_min = np.zeros(len(fields))
    stability_eigenvalue_max = np.zeros(len(fields))
    uniform_tau = np.full(len(fields), np.nan)
    uniform_vortex_curvature = np.full(len(fields), np.nan)
    uniform_orientation_curvature = np.full(len(fields), np.nan)
    uniform_stability_eigenvalue_min = np.full(len(fields), np.nan)
    uniform_stability_eigenvalue_max = np.full(len(fields), np.nan)

    uniform_candidates = np.flatnonzero(
        np.isclose(nu, 0.0, rtol=0.0, atol=1.0e-14)
    )
    uniform_index = int(uniform_candidates[0]) if len(uniform_candidates) else None

    tau_previous = 0.0
    idx_previous = 0
    beta0 = abs(params.beta_rad) < 1e-14

    for i, Bi in enumerate(fields):
        denominator = guz - params.gux_factor * gux

        if params.Ku == 0.0:
            tau_candidates = np.zeros_like(nu) if Bi >= 0.0 else np.full_like(nu, np.pi)
        else:
            near_zero = np.abs(denominator) < 1e-14
            safe_denominator = np.where(near_zero, np.nan, denominator)
            b_eff = (params.Ms * Bi) / (2.0 * params.Ku) * (gzz / safe_denominator)
            curvature_sign = np.sign(safe_denominator)

            tau_candidates = np.empty_like(nu)
            if np.any(near_zero):
                tau_candidates[near_zero] = (
                    np.zeros(np.count_nonzero(near_zero))
                    if Bi >= 0.0
                    else np.full(np.count_nonzero(near_zero), np.pi)
                )

            valid = ~near_zero
            if np.any(valid):
                if beta0:
                    tau_candidates[valid] = _stable_tau_beta0_array(
                        b_eff[valid],
                        curvature_sign[valid],
                        tau_previous,
                    )
                else:
                    tau_candidates[valid] = np.array([
                        _stable_tau_general(bi, params.beta_rad, si, tau_previous)
                        for bi, si in zip(b_eff[valid], curvature_sign[valid])
                    ])

        h_values = hamiltonian(params, Bi, tau_candidates, gex, guz, gux, gzz, gdem)
        idx_min = _follow_nu_minimum(h_values, idx_previous)

        nu_min[i] = nu[idx_min]
        tau_min[i] = tau_candidates[idx_min]
        mz_avg[i] = gzz[idx_min] * np.cos(tau_min[i])
        energy[i] = h_values[idx_min]

        selected_hessian = hamiltonian_hessian(
            params,
            Bi,
            tau_min[i],
            guz=guz[idx_min],
            gux=gux[idx_min],
            gzz=gzz[idx_min],
            gex_d2=derivative_values["g_ex_d2"][idx_min],
            guz_d1=derivative_values["g_u_z_d1"][idx_min],
            guz_d2=derivative_values["g_u_z_d2"][idx_min],
            gux_d1=derivative_values["g_u_x_d1"][idx_min],
            gux_d2=derivative_values["g_u_x_d2"][idx_min],
            gzz_d1=derivative_values["g_z_z_d1"][idx_min],
            gzz_d2=derivative_values["g_z_z_d2"][idx_min],
            gdem_d2=derivative_values["g_dem_d2"][idx_min],
        )
        selected_eigenvalues = _symmetric_eigenvalues(selected_hessian)
        stability_nu_curvature[i] = selected_hessian[0, 0]
        stability_mixed_curvature[i] = selected_hessian[0, 1]
        stability_tau_curvature[i] = selected_hessian[1, 1]
        stability_eigenvalue_min[i] = selected_eigenvalues[0]
        stability_eigenvalue_max[i] = selected_eigenvalues[1]

        if uniform_index is not None:
            uniform_tau[i] = tau_candidates[uniform_index]
            uniform_hessian = hamiltonian_hessian(
                params,
                Bi,
                uniform_tau[i],
                guz=guz[uniform_index],
                gux=gux[uniform_index],
                gzz=gzz[uniform_index],
                gex_d2=derivative_values["g_ex_d2"][uniform_index],
                guz_d1=derivative_values["g_u_z_d1"][uniform_index],
                guz_d2=derivative_values["g_u_z_d2"][uniform_index],
                gux_d1=derivative_values["g_u_x_d1"][uniform_index],
                gux_d2=derivative_values["g_u_x_d2"][uniform_index],
                gzz_d1=derivative_values["g_z_z_d1"][uniform_index],
                gzz_d2=derivative_values["g_z_z_d2"][uniform_index],
                gdem_d2=derivative_values["g_dem_d2"][uniform_index],
            )
            uniform_eigenvalues = _symmetric_eigenvalues(uniform_hessian)
            uniform_vortex_curvature[i] = uniform_hessian[0, 0]
            uniform_orientation_curvature[i] = uniform_hessian[1, 1]
            uniform_stability_eigenvalue_min[i] = uniform_eigenvalues[0]
            uniform_stability_eigenvalue_max[i] = uniform_eigenvalues[1]

        tau_previous = tau_min[i]
        idx_previous = idx_min

        if verbose:
            print(f"B = {Bi:.3f} T,  m_z = {mz_avg[i]:.3f},  nu = {nu_min[i]:.3f}")

    return HysteresisResult(
        B_T=fields,
        mz_avg=mz_avg,
        nu_min=nu_min,
        tau_rad=tau_min,
        energy=energy,
        stability_nu_curvature=stability_nu_curvature,
        stability_mixed_curvature=stability_mixed_curvature,
        stability_tau_curvature=stability_tau_curvature,
        stability_eigenvalue_min=stability_eigenvalue_min,
        stability_eigenvalue_max=stability_eigenvalue_max,
        uniform_tau_rad=uniform_tau,
        uniform_vortex_curvature=uniform_vortex_curvature,
        uniform_orientation_curvature=uniform_orientation_curvature,
        uniform_stability_eigenvalue_min=uniform_stability_eigenvalue_min,
        uniform_stability_eigenvalue_max=uniform_stability_eigenvalue_max,
    )


def compute_and_store_hysteresis(
    output_csv,
    params: ModelParameters,
    profiles=None,
    *,
    settings: HysteresisSettings | None = None,
    Bmax: float | None = None,
    n_half: int | None = None,
    fields: np.ndarray | None = None,
    stoner_wohlfarth: bool | None = None,
    output_png=None,
    make_plot=False,
    print_runtime: bool = False,
    verbose: bool = False,
) -> HysteresisResult:
    """Compute hysteresis, store it as CSV, and optionally create a plot."""
    start = time.perf_counter()
    result = run_hysteresis(
        params,
        profiles,
        settings=settings,
        Bmax=Bmax,
        n_half=n_half,
        fields=fields,
        stoner_wohlfarth=stoner_wohlfarth,
        verbose=verbose,
    )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.as_dict()).to_csv(output_csv, index=False)

    if make_plot or output_png is not None:
        if output_png is None:
            output_png = output_csv.with_suffix(".png")
        from .plotting import plot_hysteresis

        plot_hysteresis(result, output_png)

    elapsed_s = time.perf_counter() - start
    if print_runtime:
        print(f"Hysteresis runtime: {elapsed_s:.6f} s")

    return replace(result, elapsed_s=elapsed_s)


def vortex_hysteresis(
    Profiles,
    Ku,
    beta,
    Ms,
    R,
    Bmax,
    A,
    N_B,
    gux_flag,
    StonerWohlfarth=False,
    verbose=False,
):
    """Backward-compatible wrapper for the original Zenodo script API."""
    params = ModelParameters(Ku=Ku, Ms=Ms, A=A, R=R, beta_deg=beta, gux_factor=gux_flag)
    result = run_hysteresis(
        params,
        Profiles,
        settings=HysteresisSettings(
            Bmax=Bmax,
            n_half=N_B,
            stoner_wohlfarth=StonerWohlfarth,
        ),
        verbose=verbose,
    )
    return result.mz_avg, result.nu_min, result.tau_rad, result.B_T


def switching_field_astroid(theta_rad, anisotropy_field_T):
    """Backward-compatible alias for stoner_wohlfarth_switching_field."""
    return stoner_wohlfarth_switching_field(theta_rad, anisotropy_field_T)


def coercive_field_zero_crossing(theta_rad, anisotropy_field_T):
    """Backward-compatible alias for stoner_wohlfarth_coercive_field."""
    return stoner_wohlfarth_coercive_field(theta_rad, anisotropy_field_T)


def astroid(psi_rad):
    """Backward-compatible alias for stoner_wohlfarth_astroid."""
    return stoner_wohlfarth_astroid(psi_rad)
