"""Small script entrypoint for the Pixi sw-astroid task."""

import numpy as np

from minimal_hyperbolic_micromagnetics import astroid, switching_field_astroid


def main():
    psi = np.linspace(0.0, 2.0 * np.pi, 2001)
    h_parallel, h_perp = astroid(psi)
    residual = np.abs(h_parallel) ** (2.0 / 3.0) + np.abs(h_perp) ** (2.0 / 3.0) - 1.0

    theta = np.deg2rad(np.array([15, 30, 45, 60, 75], dtype=float))
    h_theory = switching_field_astroid(theta, 1.0)

    print("Astroid residual max:", np.max(np.abs(residual)))
    print("angle_deg  h_theory")
    for deg, ht in zip(np.rad2deg(theta), h_theory):
        print(f"{deg:9.0f}  {ht:8.6f}")


if __name__ == "__main__":
    main()
