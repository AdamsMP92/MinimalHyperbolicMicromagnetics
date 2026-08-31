# MinimalHyperbolicMicromagnetics

Reduced hyperbolic-vortex micromagnetics for spherical magnetic nanoparticles.

This repository provides optimized profile functions, analytic reference
formulas, and a field-following hysteresis evaluator for the minimal
hyperbolic-vortex model. It is intended as a compact Python package that can be
used from scripts, notebooks, tests, or repository examples.

The code is related to the article *Minimal model for vortex nucleation and
reversal in spherical magnetic nanoparticles*. This version focuses on a faster
and more precise implementation of the reduced energy profiles and on explicit
analytic checks for the vortex-nucleation field and the Stoner-Wohlfarth limit.

## References

- Published article: [https://doi.org/10.1103/8rkf-j2kn](https://doi.org/10.1103/8rkf-j2kn)
- arXiv preprint: [https://arxiv.org/abs/2601.17176](https://arxiv.org/abs/2601.17176)
- Supplementary Zenodo repository: [https://doi.org/10.5281/zenodo.21481863](https://doi.org/10.5281/zenodo.21481863)

## Model

The micromagnetic Hamiltonian is

```math
\mathcal{H}(\mathbf{m})
=
A\int_V (\nabla \mathbf m)^2\,dV
-
K_u\int_V (\mathbf m\cdot \mathbf e_u)^2\,dV
-
M_s\int_V \mathbf B_{\mathrm{ext}}\cdot \mathbf m\,dV
-
\frac{1}{2}\int_V \mathbf M\cdot\mathbf B_{\mathrm d}\,dV .
```

The local hyperbolic-vortex Ansatz is

```math
\mathbf{M}'(\mathbf{r}')
=
M_s
\left[
\tanh\!\left(\nu\rho'/R\right)\mathbf e_\phi'
+
\mathrm{sech}\!\left(\nu\rho'/R\right)\mathbf e_z'
\right].
```

The global vortex is obtained by rotation,

```math
\mathbf{M}(\mathbf r)
=
\mathbf R(\tau,\omega)
\mathbf M'\!\left(\mathbf R^T(\tau,\omega)\mathbf r\right),
```

with polar rotation angle $\tau$, azimuthal rotation angle $\omega$, and

```math
\mathbf R(\tau,\omega)=\mathbf R_z(\omega)\mathbf R_y(\tau).
```

The reduced Hamiltonians are dimensionless energies obtained by dividing the
physical energy defined above by the exchange scale $(4\pi/3) A R$:

```math
\mathcal H_{\mathrm{red}}
=
\frac{\mathcal H}{(4\pi/3) A R}.
```

With the sphere volume

```math
V_s=\frac{4\pi}{3}R^3,
```

the anisotropy, Zeeman, and magnetostatic prefactors become

```math
\frac{K_uV_s}{(4\pi/3)AR}=\frac{K_uR^2}{A},
\qquad
\frac{M_sBV_s}{(4\pi/3)AR}=\frac{M_sBR^2}{A},
\qquad
\frac{\mu_0M_s^2V_s}{(4\pi/3)AR}=\frac{\mu_0M_s^2R^2}{A}.
```

This is the normalization used in the code and yields the compact prefactors
shown in the reduced Hamiltonians:

For a uniaxial anisotropy direction tilted by an angle $\beta$ relative to the
field axis, the reduced Hamiltonian used by the package is

```math
\mathcal H'(\nu,\tau,B,\beta)
=
g_{\mathrm{ex}}(\nu)
-
\frac{K_uR^2}{A}
\left[
g_u^z(\nu)\cos^2(\tau-\beta)
+
g_u^x(\nu)\sin^2(\tau-\beta)
\right]
-
\frac{M_sR^2B}{A}g_z^z(\nu)\cos\tau
-
\frac{\mu_0M_s^2R^2}{A}g_{\mathrm{dem}}(\nu).
```

The minimal form used in the article is recovered by omitting the transverse
anisotropy profile, which in the code corresponds to `gux_factor=0`:

```math
\mathcal H''(\nu,\tau,B,\beta)
=
g_{\mathrm{ex}}(\nu)
-
\frac{K_uR^2}{A}
g_u^z(\nu)\cos^2(\tau-\beta)
-
\frac{M_sR^2B}{A}g_z^z(\nu)\cos\tau
-
\frac{\mu_0M_s^2R^2}{A}g_{\mathrm{dem}}(\nu).
```

## Profile Functions

The profile functions are the reduced dimensionless volume integrals obtained
after inserting the hyperbolic-vortex Ansatz, including the rotation matrix,
into the original Hamiltonian. They collect the dependence on the vortex
parameter $\nu$, while the material parameters, particle radius, applied field,
and orientation angle remain as explicit prefactors in the reduced
Hamiltonians.

The optimized dimensionless profile functions are

```math
g_u^z(\nu)
=
3\int_0^1
x\sqrt{1-x^2}\,
\mathrm{sech}^2(\nu x)\,dx,
```

```math
g_z^z(\nu)
=
3\int_0^1
x\sqrt{1-x^2}\,
\mathrm{sech}(\nu x)\,dx,
```

```math
g_u^x(\nu)
=
4\int_0^1
x\sqrt{1-x^2}\,
\tanh^2(\nu x)\,dx
=
\frac{4}{3}\left[1-g_u^z(\nu)\right],
```

```math
g_{\mathrm{ex}}(\nu)
=
3\int_0^1\sqrt{1-x^2}
\left[
\nu^2x\,\mathrm{sech}^2(\nu x)
+
\frac{\tanh^2(\nu x)}{x}
\right]dx.
```

The positive demagnetizing-energy profile entering the reduced Hamiltonian is
computed from the spherical-harmonic form

```math
g_{\mathrm{dem}}(\nu)
=
\frac{1}{2}
-
\frac{3}{2}
\sum_{\substack{\ell=1\\ \ell\ \mathrm{odd}}}^{\infty}
\left[
\int_0^1
x\,\mathrm{sech}(\nu x)\,
P_\ell\!\left(\sqrt{1-x^2}\right)
\,dx
\right]^2 .
```

The magnetization projected along the field axis is

```math
\langle m_z\rangle = g_z^z(\nu)\cos\tau .
```

## Analytic Formulas

### Vortex Nucleation

The nucleation field follows from the local stability of the saturated state.
For $\beta=0$ and $\tau=0$, the uniform state corresponds to $\nu=0$. At this
point the additional transverse anisotropy term of $\mathcal H'$ is multiplied
by $\sin^2(\tau-\beta)=0$, so the same nucleation condition is obtained from
both $\mathcal H'$ and $\mathcal H''$. Expanding the reduced profiles around
$\nu=0$ gives

```math
g_{\mathrm{ex}}(\nu)=2\nu^2+\mathcal O(\nu^4),
\qquad
g_u^z(\nu)=1-\frac{2}{5}\nu^2+\mathcal O(\nu^4),
```

```math
g_z^z(\nu)=1-\frac{1}{5}\nu^2+\mathcal O(\nu^4),
\qquad
g_{\mathrm{dem}}(\nu)=\frac{1}{3}+\frac{1}{15}\nu^2+\mathcal O(\nu^4).
```

Inserted into the minimal Hamiltonian, this gives the second variation

```math
\left.
\frac{\partial^2\mathcal H'}{\partial\nu^2}
\right|_{\nu=0,\tau=0,\beta=0}
=
\left.
\frac{\partial^2\mathcal H''}{\partial\nu^2}
\right|_{\nu=0,\tau=0,\beta=0}
=
4
+
\frac{4}{5}\frac{K_uR^2}{A}
+
\frac{2}{5}\frac{M_sBR^2}{A}
-
\frac{2}{15}\frac{\mu_0M_s^2R^2}{A}.
```

The vortex nucleation field is obtained by setting this curvature to zero.
The exact small-$\nu$ vortex-nucleation field is therefore

```math
B_{\mathrm{nuc}}
=
\frac{1}{3}\mu_0M_s
-
\frac{2K_u}{M_s}
-
\frac{10A}{M_sR^2}.
```

The exchange length is

```math
\ell_{\mathrm{ex}}
=
\sqrt{\frac{2A}{\mu_0M_s^2}}.
```

The anisotropy for which $B_{\mathrm{nuc}}=0$ is

```math
K_u
=
\frac{1}{6}\mu_0M_s^2
\left(
1-15\frac{\ell_{\mathrm{ex}}^2}{R^2}
\right),
```

and the corresponding critical vortex-nucleation radius is

```math
R_{\mathrm{nuc}}
=
\sqrt{15}\,\ell_{\mathrm{ex}}
\left[
1
-
6\frac{K_u}{\mu_0M_s^2}
\right]^{-1/2}.
```

### Stoner-Wohlfarth Limit

The Stoner-Wohlfarth limit is recovered for $\nu=0$, where the magnetization is
spatially uniform and only the angular degree of freedom remains. In the
minimal model this is the relevant regime below the vortex-nucleation threshold,
roughly for radii $R<R_{\mathrm{nuc}}$, where the uniform state remains locally
stable against the hyperbolic-vortex mode.

For the Stoner-Wohlfarth limit with

```math
B_K=\frac{2K_u}{M_s},
```

the astroid switching field is

```math
B_{\mathrm{sw}}(\beta)
=
B_K
\left(
|\sin\beta|^{2/3}
+
|\cos\beta|^{2/3}
\right)^{-3/2}.
```

The projected coercive field is

```math
B_c(\beta)
=
\begin{cases}
B_{\mathrm{sw}}(\beta),
& 0\le \beta \le \pi/4,\\
\frac{B_K}{2}|\sin(2\beta)|,
& \pi/4 < \beta \le \pi/2.
\end{cases}
```

The reduced Stoner-Wohlfarth astroid is parametrized as

```math
h_\parallel(\psi)=-\cos^3\psi,
\qquad
h_\perp(\psi)=\sin^3\psi,
```

and satisfies

```math
|h_\parallel|^{2/3}+|h_\perp|^{2/3}=1.
```

## Installation

With Pixi:

```bash
pixi install
pixi run test
```

For an editable Python install:

```bash
python -m pip install -e .
```

## Basic Use

```python
from minimal_hyperbolic_micromagnetics import (
    HysteresisSettings,
    ModelParameters,
    ProfileComputation,
    compute_profiles,
    run_hysteresis,
)

profiles = compute_profiles(ProfileComputation(n_nu=2000, nu_max=20.0))

params = ModelParameters(
    Ku=4.8e4,
    Ms=1.7e6,
    A=1e-11,
    R=20e-9,
    beta_deg=0.0,
    gux_factor=0.0,
)

settings = HysteresisSettings(Bmax=1.0, n_half=250)
result = run_hysteresis(params, profiles, settings=settings)
```

The returned `HysteresisResult` contains the applied field, average projected
magnetization, selected vortex parameter, angular coordinate, reduced energy,
and optionally a runtime when using `compute_and_store_hysteresis`:

```python
result.B_T
result.mz_avg
result.nu_min
result.tau_rad
result.energy
result.elapsed_s
```

The closed formulas are available through:

```python
from minimal_hyperbolic_micromagnetics import (
    exchange_length,
    vortex_nucleation_field,
    critical_anisotropy_for_zero_nucleation_field,
    vortex_nucleation_radius,
    stoner_wohlfarth_astroid,
    stoner_wohlfarth_switching_field,
    stoner_wohlfarth_coercive_field,
)
```

## Examples

The example scripts can be run with:

```bash
pixi run compute
pixi run nucleation-field
pixi run sw-astroide-example
pixi run coercive-field
```

They generate CSV and PNG output in separate ignored output folders.

## Physics Checks

The package includes focused tests for:

- the uniform-profile limit,
- the optimized identity `g_u^x = 4/3 (1 - g_u^z)`,
- the analytic vortex-nucleation formulas,
- the Stoner-Wohlfarth astroid,
- the distinction between switching field and projected coercive field.
