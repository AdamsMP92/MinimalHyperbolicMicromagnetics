"""Example comparing Stoner-Wohlfarth coercive fields with theory.

The coercive field used here is the zero crossing of the magnetization
projected onto the field direction. This is distinct from the switching field
for field angles beta > 45 deg.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minimal_hyperbolic_micromagnetics import (
    HysteresisSettings,
    ModelParameters,
    coercive_field_from_hysteresis,
    run_hysteresis,
    stoner_wohlfarth_coercive_field,
    stoner_wohlfarth_switching_field,
)


# =============================================================================
# Output configuration
# =============================================================================
HERE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = HERE / "sw_coercive_field_example_output"
OUTPUT_CSV = OUTPUT_DIR / "coercive_field_comparison.csv"
OUTPUT_PNG = OUTPUT_DIR / "coercive_field_comparison.png"


# =============================================================================
# Stoner-Wohlfarth material parameters
# =============================================================================
KU = 4.3e6
JS = 1.6
MU0 = 4.0 * np.pi * 1e-7
MS = JS / MU0

# A and R are irrelevant when stoner_wohlfarth=True, but the reduced model uses
# the same parameter container for all variants.
A = 1.0
R = 1.0


# =============================================================================
# Numerical settings
# =============================================================================
ANGLES_DEG = np.linspace(5.0, 85.0, 17)
ANALYTIC_ANGLES_DEG = np.linspace(0.0, 90.0, 1001)
BMAX = 8.1
N_FIELD = 6001


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

    coercive_hysteresis = abs(
        coercive_field_from_hysteresis(result, branch="descending")
    )
    coercive_analytic = stoner_wohlfarth_coercive_field(beta, anisotropy_field)
    switching_analytic = stoner_wohlfarth_switching_field(beta, anisotropy_field)

    rows.append({
        "beta_deg": beta_deg,
        "B_coercive_analytic_T": coercive_analytic,
        "B_coercive_hysteresis_T": coercive_hysteresis,
        "B_switch_analytic_T": switching_analytic,
        "abs_error_T": abs(coercive_hysteresis - coercive_analytic),
    })

comparison = pd.DataFrame(rows)
comparison.to_csv(OUTPUT_CSV, index=False)

analytic_beta = np.deg2rad(ANALYTIC_ANGLES_DEG)
coercive_analytic_fine = stoner_wohlfarth_coercive_field(analytic_beta, anisotropy_field)
switching_analytic_fine = stoner_wohlfarth_switching_field(analytic_beta, anisotropy_field)

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(
    ANALYTIC_ANGLES_DEG,
    coercive_analytic_fine,
    color="black",
    linewidth=1.0,
    label=r"analytic $B_c$",
)
ax.scatter(
    comparison["beta_deg"],
    comparison["B_coercive_hysteresis_T"],
    color="#0072B2",
    s=24,
    label=r"hysteresis $B_c$",
)
ax.plot(
    ANALYTIC_ANGLES_DEG,
    switching_analytic_fine,
    color="#D55E00",
    linewidth=0.9,
    linestyle="--",
    label=r"analytic $B_{\rm sw}$",
)
ax.set_xlabel(r"$\beta$ [deg]")
ax.set_ylabel("field [T]")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300)

print(comparison)
print(f"Saved comparison data to {OUTPUT_CSV}")
print(f"Saved comparison plot to {OUTPUT_PNG}")
