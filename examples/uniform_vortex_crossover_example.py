"""Radius sweep through the uniform-to-vortex crossover.

For every radius, the script calculates a complete major hysteresis loop for
the two hyperbolic models H' and H''.  The package stores every field-resolved
state and a small metadata table automatically.  This script only constructs
the physical field path, runs the calculations, extracts a few remanent
observables, and creates a diagnostic radius plot.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from minimal_hyperbolic_micromagnetics import (
    HysteresisSettings,
    ModelParameters,
    ProfileComputation,
    analyze_hysteresis,
    compute_and_store_hysteresis,
    compute_profiles,
    vortex_nucleation_field,
)

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "uniform_vortex_crossover_example_output"
)

# Material parameters of the spherical particle in SI units.
KU, MS, A = 4.8e4, 1.7e6, 1.0e-11
RADII_NM = np.arange(6.0, 20.5, 0.5)

# gux_factor=0 gives H'', while gux_factor=1 retains the transverse anisotropy
# contribution and gives H'.  All other inputs are identical for both models.
MODELS = {"Hpp": 0.0, "Hp": 1.0}

# Maximum field and target increments for the three resolution regions.
BMAX, LOW_STEP, MID_STEP, OUTER_STEP = 1.0, 1e-4, 1e-3, 5e-3


def segment(start: float, stop: float, step: float) -> np.ndarray:
    """Return an endpoint-inclusive field segment with approximately ``step``."""

    return np.linspace(start, stop, round(abs(stop - start) / step) + 1)


# The coercive region is resolved around the anisotropy field B_K.  Vortex
# creation and annihilation can occur at much larger fields, so the middle
# region extends beyond the largest analytic B_nuc in the selected radius set.
# The factors 1.25 and 1.5 provide a safety margin; ceil rounds each boundary
# outward to a value compatible with the adjacent regular field grid.
BK = 2.0 * KU / MS
BNUC = max(0.0, np.max(vortex_nucleation_field(KU, MS, A, RADII_NM * 1e-9)))
B_FINE = np.ceil(1.25 * BK / MID_STEP) * MID_STEP
B_VORTEX = np.ceil((BNUC + 1.5 * BK) / OUTER_STEP) * OUTER_STEP

# Descend from positive to negative saturation.  The [1:] slices prevent
# duplicate points where neighboring field segments meet.
descending = np.concatenate([
    segment(BMAX, B_VORTEX, OUTER_STEP),
    segment(B_VORTEX, B_FINE, MID_STEP)[1:],
    segment(B_FINE, -B_FINE, LOW_STEP)[1:],
    segment(-B_FINE, -B_VORTEX, MID_STEP)[1:],
    segment(-B_VORTEX, -BMAX, OUTER_STEP)[1:],
])

# Append the reversed path to obtain the ascending branch.  The negative
# turning point is intentionally present once in each branch.
fields = np.concatenate([descending, descending[::-1]])

# The radius changes the Hamiltonian coefficients but not the dimensionless
# profile functions.  Computing the profile table only once saves most of the
# setup cost of the radius sweep.
profiles = compute_profiles(
    ProfileComputation(n_nu=2000, n_quad=360, l_max_demag=161)
)

OUTPUT.mkdir(parents=True, exist_ok=True)
rows = []
for name, gux_factor in MODELS.items():
    for radius_nm in RADII_NM:
        model = ModelParameters(KU, MS, A, radius_nm * 1e-9, gux_factor=gux_factor)

        # This core function writes both the complete loop and a one-row
        # metadata CSV containing the model parameters and runtime.
        result = compute_and_store_hysteresis(
            OUTPUT / name / f"hysteresis_r{radius_nm:.1f}.csv",
            model,
            profiles,
            settings=HysteresisSettings(fields=fields),
            print_runtime=True,
        )
        analysis = analyze_hysteresis(result)

        # The first half of the path is the descending branch.  Because the
        # field construction contains B=0 exactly, no interpolation is needed
        # for the remanent values of nu and tau.
        zero = np.flatnonzero(np.isclose(result.B_T[: len(descending)], 0.0))[0]

        # At beta=0 the major loop is inversion symmetric.  The magnitude of
        # the descending uniform-state instability is therefore the positive
        # vortex-to-uniform return field of the ascending branch.
        rows.append({
            "model": name,
            "radius_nm": radius_nm,
            "remanence": analysis.descending.remanence,
            "coercive_field_T": abs(analysis.descending.coercive_field_T),
            "vortex_uniform_field_T": abs(analysis.descending.vortex_nucleation_field_T),
            "remanent_nu": result.nu_min[zero],
            "remanent_tau_rad": result.tau_rad[zero],
        })

# One compact table collects the radius-dependent quantities; the full field
# histories remain in the individual hysteresis CSV files.
summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT / "radius_observables.csv", index=False)

# This is intentionally a simple diagnostic plot.  Publication styling and
# further derived quantities belong to the paper repository.
columns = [
    "remanence",
    "vortex_uniform_field_T",
    "remanent_tau_rad",
    "coercive_field_T",
]
axes = summary.pivot(index="radius_nm", columns="model", values=columns).plot(
    subplots=True, layout=(2, 2), figsize=(8, 6), grid=True
)
axes.flat[0].figure.savefig(OUTPUT / "uniform_vortex_crossover.png", dpi=300)
