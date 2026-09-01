# The Hysteresis Protocol, Step by Step

This chapter explains how a quasistatic magnetic hysteresis loop is calculated
in MinimalHyperbolicMicromagnetics. It is written as a linear account of the
algorithm: we first define the physical system, then its reduced state and
energy, then the applied-field path, and finally the continuation and analysis
steps.

The central idea is simple. The applied field is changed in small increments.
At every field value, the code searches for a stable state close to the state
found at the preceding field value. Following such local minima, rather than
independently selecting the global energy minimum at every field, produces
metastability and therefore hysteresis.

## 1. Physical system and coordinate convention

We consider a spherical magnetic particle of radius $R$. The applied magnetic
flux density is parallel to the laboratory $z$-axis,

```math
\mathbf{B}_{\mathrm{ext}} = B\,\mathbf{e}_z ,
```

where $B$ is signed and is measured in tesla. Thus $B>0$ points along
$+\mathbf{e}_z$, whereas $B<0$ points along $-\mathbf{e}_z$.
In much of the micromagnetic literature, the same externally controlled
quantity is labelled $\mu_0H_{\mathrm{ext}}$. The package uses $B$ and
tesla consistently, so its horizontal hysteresis axis should be read as
$B=\mu_0H_{\mathrm{ext}}$.

The uniaxial easy-axis direction lies in the laboratory $xz$-plane and is
written as

```math
\mathbf{e}_u(\beta)
=
\sin\beta\,\mathbf{e}_x
+
\cos\beta\,\mathbf{e}_z .
```

The angle $\beta$ therefore measures the inclination of the easy axis
relative to the applied-field axis. In the Python interface it is supplied as
beta_deg in degrees. Internally, trigonometric functions use radians.

The material and geometry are specified by four primary parameters:

- $K_u$: uniaxial anisotropy constant in $\mathrm{J\,m^{-3}}$;
- $M_s$: saturation magnetization in $\mathrm{A\,m^{-1}}$;
- $A$: exchange stiffness in $\mathrm{J\,m^{-1}}$;
- $R$: particle radius in metres.

The constant

```math
\mu_0=4\pi\times10^{-7}\,\mathrm{N\,A^{-2}}
```

is the vacuum permeability.

## 2. Reduced magnetic state

The full magnetization is a three-dimensional vector field. The hyperbolic
model replaces this field by a family of physically motivated profiles
described by two generalized coordinates:

```math
(\nu,\tau).
```

The coordinate $\nu\geq 0$ controls the strength of the vortex-like
inhomogeneity. In a body-fixed cylindrical coordinate system
$(\rho',\phi',z')$, the local profile is

```math
\mathbf{m}'(\rho')
=
\tanh\left(\nu\frac{\rho'}{R}\right)\mathbf{e}_{\phi'}
+
\mathrm{sech}\left(\nu\frac{\rho'}{R}\right)\mathbf{e}_{z'} .
```

At $\nu=0$,

```math
\tanh(0)=0,
\qquad
\mathrm{sech}(0)=1,
```

so every magnetic moment points along the body-fixed $z'$-axis. The state is
spatially uniform. Increasing $\nu$ strengthens the azimuthal component and
therefore produces a progressively more vortex-like state.

The angle $\tau$ gives the orientation of the body-fixed $z'$-axis relative
to the laboratory $z$-axis. It can consequently describe the rotation of a
uniform magnetic moment and the orientation of a vortex axis within the same
reduced model.

The volume-averaged magnetization component parallel to the applied field is

```math
\langle m_z\rangle
=
g_{zz}(\nu)\cos\tau.
```

This is the quantity plotted on the vertical axis of a standard hysteresis
loop. The profile function $g_{zz}$ accounts for the fact that an
inhomogeneous vortex state generally has a smaller net moment than a uniform
state. Because $\mathbf{m}=\mathbf{M}/M_s$ is a unit magnetization field,
$\langle m_z\rangle$ is dimensionless. The corresponding magnetization in
SI units is

```math
\langle M_z\rangle=M_s\langle m_z\rangle.
```

## 3. Reduced energy

The micromagnetic energy is reduced to a dimensionless function

```math
\mathcal{H}(\nu,\tau;B).
```

The profile functions

```math
g_{\mathrm{ex}},\quad
g_{uz},\quad
g_{ux},\quad
g_{zz},\quad
g_{\mathrm{dem}}
```

contain the spatial integrals of the exchange, anisotropy, Zeeman, and
magnetostatic contributions. Their derivation is discussed separately in
[Profile-function derivation](profile_derivation.md).

For compactness, define

```math
k=\frac{K_uR^2}{A},
\qquad
z=\frac{M_sBR^2}{A},
\qquad
d=\frac{\mu_0M_s^2R^2}{A},
\qquad
\delta=\tau-\beta,
```

and let $q$ denote the optional multiplicative factor gux_factor applied to
the transverse anisotropy profile. The choice $q=0$ gives the minimal
$\mathcal{H}''$ model, whereas $q=1$ retains the full transverse term of
$\mathcal{H}'$. Intermediate values are mathematically accepted, but are
best understood as interpolation or sensitivity tests. The reduced energy
used by the code is

```math
\begin{aligned}
\mathcal{H}(\nu,\tau;B)
=\;&g_{\mathrm{ex}}(\nu)
-k\left[
g_{uz}(\nu)\cos^2\delta
+q\,g_{ux}(\nu)\sin^2\delta
\right] \\
&-z\,g_{zz}(\nu)\cos\tau
-d\,g_{\mathrm{dem}}(\nu).
\end{aligned}
```

It is normalized by

```math
E_0=\frac{4\pi}{3}AR,
```

so $\mathcal{H}=E/E_0$. Multiplying all energy values by the same positive
normalization does not change the positions or stability of the minima.

At a prescribed field $B$, an equilibrium candidate must be stationary with
respect to both generalized coordinates. A stable state is a local minimum,
not merely a point of low energy.

## 4. The applied-field path

A hysteresis calculation requires an ordered sequence of field values,

```math
B_0,B_1,\ldots,B_{N-1}.
```

The order is physically important because the state at $B_i$ is used to
choose the state at $B_{i+1}$.

The default protocol is a complete major loop:

```math
+B_{\max}
\longrightarrow
-B_{\max}
\longrightarrow
+B_{\max}.
```

It is constructed from n_half evenly spaced values between
$-B_{\max}$ and $+B_{\max}$. The descending copy is reversed and followed
by the ascending copy. Consequently:

- the total number of field points is $2\,n_{\mathrm{half}}$;
- both endpoints are included;
- the negative turning point occurs twice, once at the end of the descending
  branch and once at the start of the ascending branch.

For example:

    settings = HysteresisSettings(Bmax=1.0, n_half=251)

describes a loop from $+1\,\mathrm{T}$ to $-1\,\mathrm{T}$ and back.

An arbitrary protocol can instead be passed explicitly:

    fields = np.concatenate(
        [
            np.linspace(0.8, -0.8, 321),
            np.linspace(-0.8, 0.8, 321)[1:],
        ]
    )
    settings = HysteresisSettings(fields=fields)

Here the second branch starts at index 1 so that the turning point is not
duplicated. Either convention is valid. The analysis routines identify
branches from the monotonicity of the actual field array and do not require a
particular duplication convention.

The present solver initializes the continuation at

```math
\nu_0=0,\qquad \tau_0=0.
```

This represents positive saturation along the laboratory $z$-axis. A custom
field path should therefore normally begin at a sufficiently large positive
field for this initial state to be physically appropriate.

## 5. One continuation step

Suppose that a stable state $(\nu_{i-1},\tau_{i-1})$ has already been found
at $B_{i-1}$. The following four operations are performed at $B_i$.

### 5.1 Find stable angular states

The coordinate $\nu$ is represented by a one-dimensional grid. For every
grid value $\nu_j$, the code solves

```math
\frac{\partial\mathcal{H}}{\partial\tau}
(\nu_j,\tau;B_i)=0.
```

Only angularly stable stationary points are retained:

```math
\frac{\partial^2\mathcal{H}}{\partial\tau^2}>0.
```

For $\beta=0$, the angular stationary states have a particularly simple
analytic form. For general $\beta$, the trigonometric stationarity condition
is converted to a polynomial problem and solved through its companion matrix.

Angles are periodic. The candidates $\tau$ and $\tau+2\pi$ represent the
same direction. Among stable angular candidates, the algorithm therefore
chooses the periodically closest one to the preceding angle
$\tau_{i-1}$. This suppresses artificial jumps caused only by angle wrapping.

### 5.2 Evaluate the energy along the profile grid

After selecting a stable angle for every $\nu_j$, the code evaluates

```math
\mathcal{H}_j
=
\mathcal{H}(\nu_j,\tau_j;B_i).
```

This produces a one-dimensional energy landscape along the profile coordinate.

### 5.3 Identify and continue a local minimum

The code detects local minima of the discrete sequence
$\mathcal{H}_j$. The two boundaries of the $\nu$-grid are included as
possible minima, which is essential because the uniform state lies at the
boundary $\nu=0$.

The selected minimum is the candidate whose grid index is closest to the
previously selected index. If two candidates are equally close, the one with
the lower energy is selected.

This rule deliberately follows a nearby local minimum. It does not replace the
state by the global minimum whenever another state becomes energetically
favourable.

### 5.4 Store the state and advance

For the selected state, the code stores at least

```math
B_i,\quad
\nu_i,\quad
\tau_i,\quad
\mathcal{H}_i,\quad
\langle m_z\rangle_i.
```

The pair $(\nu_i,\tau_i)$ then becomes the reference state for the next field
step.

## 6. Why this procedure produces hysteresis

At some field values, the reduced energy can contain more than one local
minimum. One minimum may correspond to the continuation of the state reached
from positive saturation, while another may already have a lower absolute
energy.

A quasistatically driven magnetic system can remain in its current local
minimum as long as that minimum is stable. It changes state when the minimum
loses stability or when the discrete continuation can no longer follow it.
The descending and ascending sweeps therefore need not visit the same state at
the same field. This history dependence is hysteresis.

The algorithm can be summarized as follows:

    choose the ordered field array
    initialize nu = 0 and tau = 0

    for every field value:
        find stable angular states along the nu grid
        evaluate the reduced energy along that grid
        detect all local minima in nu
        select the minimum closest to the preceding state
        store the state, magnetization, energy, and stability
        use this state as the starting point for the next field

If the global minimum were selected independently at every field, much of this
history dependence would be lost. The resulting curve would describe an
equilibrium envelope rather than the metastable hysteresis protocol used here.

## 7. Local stability and the Hessian

For a selected state, local stability in the two-dimensional reduced space is
described by the Hessian

```math
\mathbf{H}
=
\begin{pmatrix}
\mathcal{H}_{\nu\nu} & \mathcal{H}_{\nu\tau}\\
\mathcal{H}_{\nu\tau} & \mathcal{H}_{\tau\tau}
\end{pmatrix}.
```

With primes denoting derivatives with respect to $\nu$, its entries are

```math
\begin{aligned}
\mathcal{H}_{\nu\nu}
=\;&g_{\mathrm{ex}}''
-k\left(g_{uz}''\cos^2\delta
+qg_{ux}''\sin^2\delta\right)
-z g_{zz}''\cos\tau
-d g_{\mathrm{dem}}'',\\
\mathcal{H}_{\nu\tau}
=\;&k\left(g_{uz}'-qg_{ux}'\right)\sin(2\delta)
+z g_{zz}'\sin\tau,\\
\mathcal{H}_{\tau\tau}
=\;&2k\left(g_{uz}-qg_{ux}\right)\cos(2\delta)
+z g_{zz}\cos\tau.
\end{aligned}
```

The two eigenvalues of this symmetric $2\times2$ matrix are stored
analytically. A strict local minimum has two positive eigenvalues. If the
smallest eigenvalue approaches zero, the state becomes soft in one direction
of the reduced configuration space.

These eigenvalues are curvatures with respect to the chosen coordinates
$(\nu,\tau)$. Their numerical magnitudes depend on that parametrization.
Their signs and zero crossings are the most direct stability information.

## 8. The separately evaluated uniform reference state

The selected state can become vortex-like, with $\nu>0$, before or after the
uniform state loses its local stability. To distinguish these events, the code
also evaluates and stores a uniform reference state at

```math
\nu=0
```

throughout the field sweep. At each field point, its orientation is the stable
angular candidate selected at the $\nu=0$ grid point by the same angular
continuation rule described above. It is not a second, independently continued
hysteresis calculation. Its orientation and stability curvatures are stored
separately from those of the selected metastable state.

For the aligned case $\beta=\tau=0$, the curvature of the uniform state in
the vortex direction is

```math
\left.
\mathcal{H}_{\nu\nu}
\right|_{\nu=0,\,\tau=0}
=
4
+\frac{4}{5}\frac{K_uR^2}{A}
+\frac{2}{5}\frac{M_sBR^2}{A}
-\frac{2}{15}\frac{\mu_0M_s^2R^2}{A}.
```

Setting this expression to zero gives the analytic vortex-nucleation field

```math
B_{\mathrm{nuc}}
=
\frac{\mu_0M_s}{3}
-\frac{2K_u}{M_s}
-\frac{10A}{M_sR^2}.
```

This relation is useful both as a physical prediction and as a numerical
consistency check. The corresponding stored quantity is
uniform_vortex_curvature.

It is important to keep two statements separate:

1. The selected branch has entered a vortex-like state.
2. The uniform reference state has lost stability against a vortex-like
   perturbation.

They often occur nearby, but they are not definitions of the same event.

## 9. Descending and ascending branches

The analysis first divides the field protocol into monotonic branches.
A descending branch satisfies

```math
B_{i+1}<B_i,
```

whereas an ascending branch satisfies

```math
B_{i+1}>B_i.
```

Turning points separate the branches. This definition uses the actual order of
the field values, so it also works for custom field paths and does not assume
that the array consists of two equal halves.

When a reported quantity is branch dependent, the branch must always be named.
For example, the descending coercive field and ascending coercive field have
opposite signs in a symmetric loop.

## 10. Quantities extracted from a loop

Once the complete path has been calculated, the analysis module extracts
standard hysteresis observables.

### 10.1 Remanence

The signed remanence is the magnetization at zero applied field,

```math
m_r=\left.\langle m_z\rangle\right|_{B=0}.
```

If the field grid does not contain $B=0$ exactly, the value is obtained by
linear interpolation between the two adjacent field points that bracket zero.
The result remains signed; an absolute value can be taken later if a particular
comparison requires its magnitude.

### 10.2 Coercive field

The signed coercive field is the field at which the longitudinal average
magnetization crosses zero,

```math
\langle m_z\rangle(B_c)=0.
```

Again, linear interpolation is used between neighbouring data points with
opposite magnetization signs. A symmetric loop normally has a negative
descending $B_c$ and a positive ascending $B_c$.

### 10.3 Angular switching field

The angular switching estimator identifies the largest step in $\tau$ along
a branch. The angular difference is first wrapped periodically so that crossing
from $+\pi$ to $-\pi$ is not mistaken for a physical rotation of almost
$2\pi$. The reported switching field is the midpoint of the two field values
on either side of the largest wrapped jump.

This estimator identifies a discrete reorientation event. It is not
automatically identical to the coercive field.

### 10.4 Vortex-nucleation field

The preferred estimator uses a zero crossing of the uniform reference
curvature:

```math
\mathcal{H}^{\mathrm{uniform}}_{\nu\nu}(B_{\mathrm{nuc}})=0.
```

This is a stability-based definition and connects directly to the analytic
formula above. If the required stability data are unavailable, the analysis
can instead use a threshold crossing of the selected $\nu$ values. The
threshold method detects the appearance of a visibly nonuniform selected
state, whereas the curvature method detects the loss of stability of the
uniform reference state.

## 11. The Stoner--Wohlfarth restriction

Setting

    stoner_wohlfarth=True

restricts the profile coordinate to

```math
\nu=0.
```

Only coherent rotation remains, so the calculation becomes a
Stoner--Wohlfarth-type hysteresis protocol within the same angular continuation
framework. Profile functions are not required for the selected branch in this
mode.

The code still evaluates the hypothetical vortex curvature of the uniform
state when the necessary material parameters are available. This makes it
possible to ask whether a coherently rotating solution would be stable against
a vortex-like perturbation even when that perturbation is excluded from the
selected Stoner--Wohlfarth branch.

## 12. A complete calculation

The following example shows the main stages explicitly:

    import numpy as np

    from minimal_hyperbolic_micromagnetics import (
        HysteresisSettings,
        ModelParameters,
        ProfileComputation,
        analyze_hysteresis,
        compute_profiles,
        run_hysteresis,
    )

    params = ModelParameters(
        Ku=4.8e4,          # J/m^3
        Ms=1.4e6,          # A/m
        A=1.3e-11,         # J/m
        R=10.0e-9,         # m
        beta_deg=0.0,      # easy axis parallel to the field
    )

    settings = HysteresisSettings(
        Bmax=1.0,          # T
        n_half=251,
        stoner_wohlfarth=False,
    )

    profile_settings = ProfileComputation(
        nu_min=0.0,
        nu_max=20.0,
        n_nu=2000,
        n_quad=360,
        l_max_demag=161,
    )
    profiles = compute_profiles(profile_settings, verbose=True)

    result = run_hysteresis(
        params=params,
        settings=settings,
        profiles=profiles,
    )

    analysis = analyze_hysteresis(result)

    print(analysis.descending.remanence)
    print(analysis.descending.coercive_field_T)
    print(analysis.descending.vortex_nucleation_field_T)

The returned result contains the state at every field point. The analysis
object contains branch-resolved observables such as remanence, coercive field,
switching field, and vortex-nucleation field.

For a custom field path, replace Bmax and n_half by an explicit fields array:

    fields = np.linspace(1.0, -1.0, 401)
    settings = HysteresisSettings(fields=fields)

This example computes only the descending branch. It is sufficient when the
scientific question concerns positive saturation, reversal, and the remanent
state reached from that direction.

## 13. What the result arrays mean

The most important arrays in HysteresisResult are:

| Array | Meaning |
|---|---|
| B_T | Applied field $B_i$ in tesla |
| mz_avg | Volume-averaged longitudinal magnetization $\langle m_z\rangle_i$ |
| nu_min | Selected profile coordinate $\nu_i$ |
| tau_rad | Selected orientation angle $\tau_i$ in radians |
| energy | Selected reduced energy $\mathcal{H}_i$ |
| stability_nu_curvature | $\mathcal{H}_{\nu\nu}$ at the selected state |
| stability_mixed_curvature | $\mathcal{H}_{\nu\tau}$ at the selected state |
| stability_tau_curvature | $\mathcal{H}_{\tau\tau}$ at the selected state |
| stability_eigenvalue_min | Smaller Hessian eigenvalue at the selected state |
| stability_eigenvalue_max | Larger Hessian eigenvalue at the selected state |
| uniform_tau_rad | Orientation angle of the separately evaluated uniform state |
| uniform_vortex_curvature | Vortex-direction curvature of the uniform state |
| uniform_orientation_curvature | Angular curvature of the uniform state |
| uniform_stability_eigenvalue_min | Smaller Hessian eigenvalue of the uniform reference |
| uniform_stability_eigenvalue_max | Larger Hessian eigenvalue of the uniform reference |

All arrays follow the same field ordering as B_T.

## 14. Numerical resolution and convergence

A computed loop depends on two distinct discretizations.

First, the field resolution controls how accurately abrupt switching events and
interpolated crossing fields are located. Increasing n_half, or refining a
custom field array near a transition, improves the field resolution.

Second, the $\nu$-grid controls how accurately the reduced profile coordinate
and its local minima are resolved. A coarse $\nu$-grid can shift the apparent
transition or cause the continuation to jump between grid indices.

Profile precomputation introduces a third numerical setting: the quadrature
resolution used to calculate the spatial integrals. Its convergence should be
checked separately from the field and $\nu$ resolutions.

A practical convergence study therefore changes one resolution at a time:

1. keep the profile and $\nu$-grids fixed and refine the field path;
2. keep the field path fixed and refine the $\nu$-grid;
3. refine the profile quadrature and verify that the loop no longer changes
   materially.

The implemented analytic profile derivatives make the Hessian and its
eigenvalues more reliable than finite-difference derivatives would be,
especially near a stability zero crossing. They do not remove the need to
check the field and profile-coordinate resolutions.

## 15. Useful physical and numerical checks

Before interpreting a calculated loop, verify the following points:

1. The first field is large enough that $\nu=0,\tau=0$ is an appropriate
   positive-saturation state.
2. The final state on a closed symmetric field path returns close to the
   initial state.
3. For a symmetric aligned system, descending and ascending branches have the
   expected approximate point symmetry.
4. Remanence and coercivity change negligibly when the field grid is refined.
5. The selected $\nu$ history changes negligibly when the $\nu$-grid is
   refined.
6. The minimum Hessian eigenvalue remains positive while a tracked state is
   locally stable and approaches zero near a genuine loss of stability.
7. For the aligned uniform branch, the numerical zero of
   uniform_vortex_curvature agrees with the analytic $B_{\mathrm{nuc}}$ to
   the accuracy allowed by the field grid.

These checks separate physical switching from artefacts of a coarse continuation
grid.

## 16. Where the protocol is implemented

The main implementation is divided by responsibility:

- src/minimal_hyperbolic_micromagnetics/hysteresis.py defines the model
  parameters, field settings, continuation algorithm, state arrays, Hessian,
  and uniform reference tracking.
- src/minimal_hyperbolic_micromagnetics/analysis.py splits the field path into
  branches and extracts remanence, coercivity, switching, and nucleation
  fields.
- src/minimal_hyperbolic_micromagnetics/profiles.py constructs the reduced
  profile functions and their derivatives.
- examples/hysteresis_example.py demonstrates the complete hyperbolic-vortex
  profile and hysteresis calculation.
- examples/sw_coercive_field_example.py demonstrates coherent-rotation
  hysteresis and coercive-field analysis.
- examples/nucleation_field_example.py compares the stability-based numerical
  nucleation field with the analytic result.
- examples/uniform_vortex_crossover_example.py generates the radius-dependent
  remanence, vortex-to-uniform return field, texture-axis angle, and coercive
  field, while retaining the complete field histories of $\nu$ and $\tau$.
- examples/sw_ensemble_hysteresis_example.py demonstrates averaging over
  easy-axis orientations.

Reading these files in that order mirrors the logical order of this chapter:
define the path, follow the state, and only then analyze the completed loop.
