import numpy as np
import pandas as pd

from minimal_hyperbolic_micromagnetics import (
    HysteresisSettings,
    ModelParameters,
    astroid,
    coercive_field_from_hysteresis,
    coercive_field_zero_crossing,
    compute_and_store_hysteresis,
    run_hysteresis,
    switching_field_astroid,
)


def test_astroid_from_minimal_model_matches_closed_form():
    psi = np.linspace(0.0, 2.0 * np.pi, 2001)
    h_parallel, h_perp = astroid(psi)

    residual = np.abs(h_parallel) ** (2.0 / 3.0) + np.abs(h_perp) ** (2.0 / 3.0) - 1.0
    assert np.max(np.abs(residual)) < 1e-14


def test_sw_coercive_field_is_not_always_switching_field():
    anisotropy_field = 1.0
    theta = np.deg2rad(np.array([30.0, 60.0]))

    switching = switching_field_astroid(theta, anisotropy_field)
    coercive = coercive_field_zero_crossing(theta, anisotropy_field)

    assert np.isclose(coercive[0], switching[0])
    assert coercive[1] < switching[1]


def test_sw_hysteresis_zero_crossing_matches_piecewise_theory():
    params = ModelParameters(Ku=4.3e6, Ms=1.6 / (4.0 * np.pi * 1e-7), A=1.0, R=1.0)
    anisotropy_field = params.anisotropy_field_T

    for angle_deg in [15, 30, 45, 60, 75]:
        params_angle = ModelParameters(
            Ku=params.Ku,
            Ms=params.Ms,
            A=params.A,
            R=params.R,
            beta_deg=angle_deg,
        )
        result = run_hysteresis(
            params_angle,
            settings=HysteresisSettings(
                Bmax=8.1,
                n_half=6001,
                stoner_wohlfarth=True,
            ),
        )
        model = abs(coercive_field_from_hysteresis(result))
        theory = coercive_field_zero_crossing(np.deg2rad(angle_deg), anisotropy_field)
        assert np.isclose(model, theory, rtol=1e-3, atol=2e-4)


def test_compute_and_store_hysteresis_writes_csv(tmp_path):
    params = ModelParameters(Ku=4.3e6, Ms=1.6 / (4.0 * np.pi * 1e-7), A=1.0, R=1.0)
    output_csv = tmp_path / "hysteresis.csv"

    result = compute_and_store_hysteresis(
        output_csv,
        params,
        settings=HysteresisSettings(
            Bmax=1.0,
            n_half=7,
            stoner_wohlfarth=True,
        ),
    )
    stored = pd.read_csv(output_csv)

    assert output_csv.exists()
    assert len(result.B_T) == 14
    assert len(stored) == 14
    assert set(result.as_dict()).issubset(stored.columns)
