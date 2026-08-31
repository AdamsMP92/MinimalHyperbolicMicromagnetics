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
- Polarized SANS framework: [https://doi.org/10.1103/PhysRevB.110.014420](https://doi.org/10.1103/PhysRevB.110.014420)
- Stoner-Wohlfarth model review: [https://doi.org/10.1088/0143-0807/29/3/008](https://doi.org/10.1088/0143-0807/29/3/008)
- Stoner-Wohlfarth static properties review: [https://doi.org/10.1016/j.physb.2008.05.031](https://doi.org/10.1016/j.physb.2008.05.031)

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

Here $\mathbf m$ is the normalized magnetization direction,

```math
\mathbf m=\frac{\mathbf M}{M_s}.
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
after inserting the local hyperbolic-vortex Ansatz $\mathbf M'(\mathbf r')$,
the rotation matrix $\mathbf R(\tau,\omega)$, and therefore the global
magnetization field $\mathbf M(\mathbf r)$ into the original Hamiltonian
$\mathcal H(\mathbf m)$. They collect the dependence on the vortex parameter
$\nu$, while the material parameters, particle radius, applied field, and
orientation angle remain as explicit prefactors in the reduced Hamiltonians. In
this notation, $g_{\mathrm{ex}}$ belongs to the exchange energy, $g_u^z$ and
$g_u^x$ to the longitudinal and transverse anisotropy contributions, $g_z^z$ to
the Zeeman term and average projected magnetization, and $g_{\mathrm{dem}}$ to
the magnetostatic self-energy.

A more detailed derivation is provided in
[docs/profile_derivation.md](docs/profile_derivation.md).

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

The Stoner-Wohlfarth limit is recovered by restricting the reduced Hamiltonian
to the uniform state $\nu=0$. In this limit

```math
g_{\mathrm{ex}}(0)=0,\qquad
g_u^z(0)=1,\qquad
g_u^x(0)=0,\qquad
g_z^z(0)=1,
```

and the demagnetizing contribution is an angularly constant sphere term. The
two reduced Hamiltonians $\mathcal H'$ and $\mathcal H''$ therefore give the
same angular Stoner-Wohlfarth energy:

```math
\mathcal H_{\mathrm{SW}}(\tau,B,\beta)
=
-
\frac{K_uR^2}{A}\cos^2(\tau-\beta)
-
\frac{M_sBR^2}{A}\cos\tau
+
C_0.
```

Here $C_0$ is independent of $\tau$.

Here $\tau$ is already the magnetization angle relative to the applied-field
axis. The angle relative to the easy axis is therefore only a shifted variable,

```math
\alpha=\tau-\beta.
```

Using

```math
B_K=\frac{2K_u}{M_s},
\qquad
b=\frac{B}{B_K},
```

the same angular energy can be rescaled by the anisotropy prefactor
$K_uR^2/A$ and written, up to an irrelevant additive constant, as

```math
e_{\mathrm{SW}}(\alpha)
=
\sin^2\alpha
-
2b\cos(\alpha+\beta).
```

This uses

```math
-\cos^2\alpha
=
\sin^2\alpha-1,
\qquad
\tau=\alpha+\beta.
```

The Zeeman part can then be decomposed with

```math
\cos(\alpha+\beta)
=
\cos\alpha\cos\beta-\sin\alpha\sin\beta,
```

so that a field applied at angle $\beta$ corresponds to reduced components
$h_\parallel=b\cos\beta$ and $h_\perp=-b\sin\beta$ in the easy-axis frame.

Thus, in the notation of this repository, one can work directly with $\tau$,
while $\alpha$ is only the conventional Stoner-Wohlfarth angle measured from
the easy axis. In the minimal model this uniform limit is the relevant regime
below the vortex-nucleation threshold, roughly for radii $R<R_{\mathrm{nuc}}$,
where the uniform state remains locally stable against the hyperbolic-vortex
mode.

For the standard astroid derivation one rewrites the Zeeman part in terms of
reduced field components parallel and perpendicular to the easy axis:

```math
e(\alpha)
=
\sin^2\alpha
-
2h_\parallel\cos\alpha
-
2h_\perp\sin\alpha,
\qquad
h_i=\frac{B_i}{B_K}.
```

The metastable magnetization directions are local minima of this one-dimensional
energy. The switching boundary is reached when a minimum and a saddle merge.
Therefore the astroid follows from solving the two conditions

```math
\frac{\partial e}{\partial\alpha}=0,
\qquad
\frac{\partial^2 e}{\partial\alpha^2}=0.
```

Eliminating $\alpha$ gives the Stoner-Wohlfarth astroid in reduced field space:

```math
h_\parallel(\alpha)=-\cos^3\alpha,
\qquad
h_\perp(\alpha)=\sin^3\alpha,
```

with

```math
|h_\parallel|^{2/3}+|h_\perp|^{2/3}=1.
```

Thus the astroid is not itself a hysteresis loop. It is the stability boundary
in the two-dimensional field plane. A hysteresis simulation at a fixed field
angle $\beta$ corresponds to moving along a straight line through this plane;
the switching field is the point where this line first intersects the astroid.

The astroid switching field is

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

This is the field magnitude at which the metastable branch loses stability.
The coercive field used in a hysteresis loop is instead the field where the
magnetization projected onto the field axis crosses zero. For field angles
below $45^\circ$, this happens at the switching event. For larger angles, the
projection can cross zero before the actual switching instability is reached.
The projected coercive field is

```math
B_c(\beta)
=
\begin{cases}
B_{\mathrm{sw}}(\beta),
& 0\le \beta \le \pi/4,\\
\frac{B_K}{2}\sin(2\beta),
& \pi/4 < \beta \le \pi/2.
\end{cases}
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

The runnable example scripts are located in the `examples/` folder and can be
started through Pixi tasks:

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
