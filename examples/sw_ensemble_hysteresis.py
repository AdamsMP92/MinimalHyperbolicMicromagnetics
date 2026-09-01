"""Stoner-Wohlfarth hysteresis of an isotropic easy-axis ensemble.

For randomly oriented, uniaxial particles the easy axis is an undirected axis.
It is therefore sufficient to average over beta in [0, pi/2] with the
three-dimensional orientation measure sin(beta) d beta.  The implementation
uses Gauss-Legendre quadrature in mu = cos(beta), for which the measure is
simply d mu on [0, 1].
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minimal_hyperbolic_micromagnetics import (
    HysteresisResult,
    HysteresisSettings,
    ModelParameters,
    MU_0,
    coercive_field_from_hysteresis,
    remanent_magnetization,
    run_hysteresis,
)


# =============================================================================
# Output configuration
# =============================================================================
HERE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = HERE / "sw_ensemble_hysteresis_output"
OUTPUT_CSV = OUTPUT_DIR / "sw_ensemble_hysteresis.csv"
ORIENTATION_CSV = OUTPUT_DIR / "sw_ensemble_orientations.csv"
OUTPUT_PNG = OUTPUT_DIR / "sw_ensemble_hysteresis.png"


# =============================================================================
# Stoner-Wohlfarth material parameters
# =============================================================================
KU = 4.3e6           # uniaxial anisotropy constant [J/m^3]
JS = 1.6             # saturation polarization mu_0 M_s [T]
MS = JS / MU_0       # saturation magnetization [A/m]

# A and R cancel in the Stoner-Wohlfarth restriction nu=0. They remain in the
# common model container so this example uses the same API as the vortex model.
A = 1.0
R = 1.0


# =============================================================================
# Ensemble and field discretization
# =============================================================================
N_ORIENTATIONS = 96
N_HALF = 1201
BMAX_OVER_BK = 1.2
PROGRESS_EVERY = 16


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

base_model = ModelParameters(Ku=KU, Ms=MS, A=A, R=R)
anisotropy_field = base_model.anisotropy_field_T
settings = HysteresisSettings(
    Bmax=BMAX_OVER_BK * anisotropy_field,
    n_half=N_HALF,
    stoner_wohlfarth=True,
)

# Gauss-Legendre nodes and weights on mu = cos(beta) in [0, 1]. Because
# integral_0^(pi/2) f(beta) sin(beta) d beta = integral_0^1 f(arccos(mu)) d mu,
# these weights directly form the normalized isotropic ensemble average.
nodes, weights = np.polynomial.legendre.leggauss(N_ORIENTATIONS)
mu = 0.5 * (nodes + 1.0)
weights = 0.5 * weights
beta_rad = np.arccos(mu)
beta_deg = np.rad2deg(beta_rad)

ensemble_mz = None
fields = None
start = time.perf_counter()

for index, (angle_deg, weight) in enumerate(zip(beta_deg, weights), start=1):
    model = ModelParameters(
        Ku=KU,
        Ms=MS,
        A=A,
        R=R,
        beta_deg=float(angle_deg),
    )
    result = run_hysteresis(model, settings=settings)

    if ensemble_mz is None:
        fields = result.B_T.copy()
        ensemble_mz = np.zeros_like(result.mz_avg)
    ensemble_mz += weight * result.mz_avg

    if index == 1 or index % PROGRESS_EVERY == 0 or index == N_ORIENTATIONS:
        print(
            f"[ensemble] orientations {index}/{N_ORIENTATIONS}: "
            f"beta={angle_deg:.3f} deg",
            flush=True,
        )

elapsed = time.perf_counter() - start
print(f"[ensemble] averaging: done in {elapsed:.3f} s", flush=True)

# A beta=0 loop is included as a useful single-particle reference.
aligned_result = run_hysteresis(base_model, settings=settings)

# The shared analysis functions only require the field path and magnetization;
# the remaining arrays are neutral placeholders for this ensemble-level result.
ensemble_result = HysteresisResult(
    B_T=fields,
    mz_avg=ensemble_mz,
    nu_min=np.zeros_like(fields),
    tau_rad=np.zeros_like(fields),
    energy=np.zeros_like(fields),
)
remanence = remanent_magnetization(ensemble_result, branch="descending")
coercive_field = coercive_field_from_hysteresis(
    ensemble_result,
    branch="descending",
)

hysteresis_data = pd.DataFrame({
    "B_T": fields,
    "B_over_BK": fields / anisotropy_field,
    "mz_ensemble": ensemble_mz,
    "mz_beta0": aligned_result.mz_avg,
})
hysteresis_data.to_csv(OUTPUT_CSV, index=False)

orientation_data = pd.DataFrame({
    "beta_deg": beta_deg,
    "cos_beta": mu,
    "quadrature_weight": weights,
})
orientation_data.to_csv(ORIENTATION_CSV, index=False)

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.axhline(0.0, color="0.8", linewidth=0.7, zorder=0)
ax.axvline(0.0, color="0.8", linewidth=0.7, zorder=0)
ax.plot(
    fields / anisotropy_field,
    aligned_result.mz_avg,
    color="0.55",
    linewidth=0.9,
    linestyle="--",
    label=r"single axis, $\beta=0$",
)
ax.plot(
    fields / anisotropy_field,
    ensemble_mz,
    color="#0072B2",
    linewidth=1.5,
    label="isotropic ensemble",
)
ax.set_xlim(-BMAX_OVER_BK, BMAX_OVER_BK)
ax.set_ylim(-1.05, 1.05)
ax.set_xlabel(r"$B_{\mathrm{ext}}/B_K$")
ax.set_ylabel(r"$\langle m_z\rangle_{\mathrm{ens}}$")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300)
plt.close(fig)

print(f"N_orientations = {N_ORIENTATIONS}")
print(f"sum(weights) = {np.sum(weights):.16f}")
print(f"descending remanence = {remanence:.8f}")
print(
    "descending coercive field = "
    f"{coercive_field:.8f} T "
    f"({coercive_field / anisotropy_field:.8f} B_K)"
)
print(f"Saved hysteresis data to {OUTPUT_CSV}")
print(f"Saved orientation quadrature to {ORIENTATION_CSV}")
print(f"Saved plot to {OUTPUT_PNG}")
