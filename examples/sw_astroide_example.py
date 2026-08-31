"""Example comparing the Stoner-Wohlfarth astroid with switching fields.

The reduced model is evaluated in the Stoner-Wohlfarth limit by setting
stoner_wohlfarth=True. For each field angle beta, the script follows the
metastable branch from positive to negative field and extracts the jump field.
The resulting field components are compared with the analytic astroid.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minimal_hyperbolic_micromagnetics import (
    HysteresisSettings,
    ModelParameters,
    run_hysteresis,
    stoner_wohlfarth_astroid,
    stoner_wohlfarth_switching_field,
)


# =============================================================================
# Output configuration
# =============================================================================
HERE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = HERE / "sw_astroide_example_output"
OUTPUT_CSV = OUTPUT_DIR / "sw_astroide_comparison.csv"
OUTPUT_PNG = OUTPUT_DIR / "sw_astroide_comparison.png"


# =============================================================================
# Stoner-Wohlfarth material parameters
# =============================================================================
KU = 4.3e6
JS = 1.6
MU0 = 4.0 * np.pi * 1e-7
MS = JS / MU0

# A and R are irrelevant in the strict nu=0 Stoner-Wohlfarth limit, but the
# ModelParameters dataclass keeps the same signature as the vortex model.
A = 1.0
R = 1.0


# =============================================================================
# Numerical settings
# =============================================================================
ANGLES_DEG = np.linspace(5.0, 85.0, 17)
BMAX = 8.1
N_FIELD = 6001


def extract_switching_field_from_descending_branch(result):
    """Return the field where the SW branch jumps during the descending sweep."""
    all_fields = np.asarray(result.B_T)
    all_tau = np.asarray(result.tau_rad)

    if np.all(np.diff(all_fields) < 0.0):
        fields = all_fields
        tau = np.unwrap(all_tau)
    else:
        n = len(all_fields) // 2
        fields = all_fields[:n]
        tau = np.unwrap(all_tau[:n])

    jump_index = int(np.argmax(np.abs(np.diff(tau))))
    return 0.5 * (fields[jump_index] + fields[jump_index + 1])


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

base_model = ModelParameters(Ku=KU, Ms=MS, A=A, R=R)
anisotropy_field = base_model.anisotropy_field_T
fields = np.linspace(BMAX, -BMAX, N_FIELD)
settings = HysteresisSettings(fields=fields, stoner_wohlfarth=True)

rows = []
for beta_deg in ANGLES_DEG:
    beta = np.deg2rad(beta_deg)
    model = ModelParameters(Ku=KU, Ms=MS, A=A, R=R, beta_deg=beta_deg)
    result = run_hysteresis(model, settings=settings)

    switching_hysteresis = abs(extract_switching_field_from_descending_branch(result))
    switching_analytic = stoner_wohlfarth_switching_field(beta, anisotropy_field)

    rows.append({
        "beta_deg": beta_deg,
        "B_switch_analytic_T": switching_analytic,
        "B_switch_hysteresis_T": switching_hysteresis,
        "h_parallel_hysteresis": -switching_hysteresis * np.cos(beta) / anisotropy_field,
        "h_perpendicular_hysteresis": switching_hysteresis * np.sin(beta) / anisotropy_field,
        "h_parallel_analytic": -switching_analytic * np.cos(beta) / anisotropy_field,
        "h_perpendicular_analytic": switching_analytic * np.sin(beta) / anisotropy_field,
        "abs_error_T": abs(switching_hysteresis - switching_analytic),
    })

comparison = pd.DataFrame(rows)
comparison.to_csv(OUTPUT_CSV, index=False)

psi = np.linspace(0.0, 2.0 * np.pi, 2001)
h_parallel, h_perpendicular = stoner_wohlfarth_astroid(psi)

fig, (ax_astroid, ax_field) = plt.subplots(1, 2, figsize=(8.5, 3.8))

ax_astroid.plot(h_parallel, h_perpendicular, color="black", linewidth=1.0, label="analytic")
ax_astroid.scatter(
    comparison["h_parallel_hysteresis"],
    comparison["h_perpendicular_hysteresis"],
    color="#0072B2",
    s=22,
    label="hysteresis",
)
ax_astroid.set_xlabel(r"$B_\parallel/B_K$")
ax_astroid.set_ylabel(r"$B_\perp/B_K$")
ax_astroid.set_aspect("equal", adjustable="box")
ax_astroid.legend(frameon=False)

ax_field.plot(
    comparison["beta_deg"],
    comparison["B_switch_analytic_T"],
    color="black",
    linewidth=1.0,
    label="analytic",
)
ax_field.scatter(
    comparison["beta_deg"],
    comparison["B_switch_hysteresis_T"],
    color="#0072B2",
    s=22,
    label="hysteresis",
)
ax_field.set_xlabel(r"$\beta$ [deg]")
ax_field.set_ylabel(r"$B_{\rm sw}$ [T]")
ax_field.legend(frameon=False)

fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300)

print(comparison)
print(f"Saved comparison data to {OUTPUT_CSV}")
print(f"Saved comparison plot to {OUTPUT_PNG}")
