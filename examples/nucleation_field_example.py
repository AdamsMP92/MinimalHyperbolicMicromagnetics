"""Example comparing analytic vortex-nucleation fields with hysteresis.

The analytic formula follows from the small-nu stability of the uniform state.
The numerical estimate below uses the same reduced Hamiltonian, follows the
metastable state from large positive field downward, and interpolates the zero
of the analytically evaluated uniform-state vortex curvature.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minimal_hyperbolic_micromagnetics import (
    HysteresisSettings,
    ModelParameters,
    ProfileComputation,
    compute_profiles,
    run_hysteresis,
    vortex_nucleation_field,
    vortex_nucleation_field_from_hysteresis,
    vortex_nucleation_radius,
)


# =============================================================================
# Output configuration
# =============================================================================
HERE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = HERE / "nucleation_field_example_output"
OUTPUT_CSV = OUTPUT_DIR / "nucleation_field_comparison.csv"
OUTPUT_PNG = OUTPUT_DIR / "nucleation_field_comparison.png"


# =============================================================================
# Material parameters
# =============================================================================
KU = 1.0 * 4.8e4   # uniaxial anisotropy constant [J/m^3]
MS = 1.7e6         # saturation magnetization [A/m]
A = 1e-11          # exchange stiffness [J/m]


# =============================================================================
# Numerical settings
# =============================================================================
# A smaller nu_max is enough here because nucleation happens close to the
# uniform state. Analytic profile derivatives provide the tracked
# uniform-state vortex curvature throughout the field sweep.
PROFILE_SETTINGS = ProfileComputation(
    nu_min=0.0,
    nu_max=6.0,
    n_nu=2000,
    n_quad=360,
    l_max_demag=161,
)

# The field sweep is custom and descending only: +Bmax -> -Bmax. This isolates
# the first vortex nucleation event from the positive saturation branch.
BMAX = 1.0
N_FIELD = 1601


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

profiles = compute_profiles(PROFILE_SETTINGS, verbose=True)
fields = np.linspace(BMAX, -BMAX, N_FIELD)

critical_radius = vortex_nucleation_radius(KU, MS, A)
radii = np.linspace(1.25 * critical_radius, 6.0 * critical_radius, 10)

rows = []
for radius in radii:
    model = ModelParameters(
        Ku=KU,
        Ms=MS,
        A=A,
        R=radius,
        beta_deg=0.0,
        gux_factor=0.0,
    )
    settings = HysteresisSettings(fields=fields, stoner_wohlfarth=False)
    result = run_hysteresis(model, profiles, settings=settings)

    analytic = vortex_nucleation_field(KU, MS, A, radius)
    numerical = vortex_nucleation_field_from_hysteresis(
        result,
        branch="descending",
        method="stability",
    )
    rows.append({
        "R_nm": radius * 1e9,
        "B_nuc_analytic_T": analytic,
        "B_nuc_hysteresis_T": numerical,
        "abs_error_T": abs(numerical - analytic),
    })

comparison = pd.DataFrame(rows)
comparison.to_csv(OUTPUT_CSV, index=False)

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(comparison["R_nm"], comparison["B_nuc_analytic_T"], label="analytic", color="black")
ax.scatter(
    comparison["R_nm"],
    comparison["B_nuc_hysteresis_T"],
    label="hysteresis",
    color="#0072B2",
    s=24,
)
ax.set_xlabel("R [nm]")
ax.set_ylabel(r"$B_{\rm nuc}$ [T]")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300)

print(comparison)
print(f"Saved comparison data to {OUTPUT_CSV}")
print(f"Saved comparison plot to {OUTPUT_PNG}")
