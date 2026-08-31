"""Example for computing profiles and a hysteresis loop.

The script is intentionally written as a flat file so it can be started with
VS Code's "Run Python File" button. All parameters that users are likely to
change are collected near the top.
"""

from pathlib import Path

# Results are written next to the repository root, but into a separate folder.
HERE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = HERE / "hysteresis_example_output"

from minimal_hyperbolic_micromagnetics import (
    HysteresisSettings,
    ModelParameters,
    ProfileComputation,
    compute_and_store_hysteresis,
    compute_and_store_profiles,
)


# =============================================================================
# Output configuration
# =============================================================================
MAKE_PROFILE_PLOT = True
MAKE_HYSTERESIS_PLOT = True

PROFILE_CSV = OUTPUT_DIR / "vortex_energy_profiles.csv"
PROFILE_PNG = OUTPUT_DIR / "vortex_energy_profiles.png"
HYSTERESIS_CSV = OUTPUT_DIR / "model_hysteresis.csv"
HYSTERESIS_PNG = OUTPUT_DIR / "fig.png"

# =============================================================================
# Physical model
# =============================================================================
# These parameters define the reduced hyperbolic-vortex Hamiltonian.
# Setting gux_factor=0 gives the H'' model used for the minimal description.
# Setting gux_factor=1 retains the transverse anisotropy profile g_u^x and gives
# the extended H' variant.
MODEL = ModelParameters(
    Ku=1.0 * 4.8e4,   # uniaxial anisotropy constant [J/m^3]
    beta_deg=0.0,     # anisotropy inclination angle [deg]
    Ms=1.7e6,         # saturation magnetization [A/m]
    R=20e-9,          # particle radius [m]
    A=1e-11,          # exchange stiffness [J/m]
    gux_factor=0.0,   # 0: H'', 1: H' with g_u^x retained
)

# =============================================================================
# Hysteresis settings
# =============================================================================
# The field sweep runs from +Bmax to -Bmax and back. The number of computed
# field values is therefore 2*n_half.
#
# stoner_wohlfarth=True keeps only the nu=0 state. This is useful as a check
# against the analytic Stoner-Wohlfarth limit. Use False for the vortex model.
HYSTERESIS_SETTINGS = HysteresisSettings(
    Bmax=1.0,               # maximum applied field [T]
    n_half=250,             # half-number of field steps
    stoner_wohlfarth=False,  # True: SW limit, False: hyperbolic-vortex model
)

# =============================================================================
# Profile settings
# =============================================================================
# n_nu controls the resolution of the vortex parameter grid. n_quad and
# l_max_demag control the numerical precision of the profile integrals.
PROFILE_SETTINGS = ProfileComputation(
    nu_min=0.0,
    nu_max=20.0,
    n_nu=2000,
    n_quad=360,
    l_max_demag=161,
)

# =============================================================================
# Run computation
# =============================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


profiles = compute_and_store_profiles(
    PROFILE_CSV,
    PROFILE_SETTINGS,
    output_png=PROFILE_PNG,
    make_plot=MAKE_PROFILE_PLOT,
)

result = compute_and_store_hysteresis(
    HYSTERESIS_CSV,
    MODEL,
    profiles.to_dict(orient="list"),
    settings=HYSTERESIS_SETTINGS,
    output_png=HYSTERESIS_PNG if MAKE_HYSTERESIS_PLOT else None,
    print_runtime=True,
)
