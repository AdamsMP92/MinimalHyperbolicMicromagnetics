# Derivation of the Profile Functions

This note collects the analytic reductions used for the profile functions in
`minimal_hyperbolic_micromagnetics.profiles`. The starting point is the
hyperbolic-vortex Ansatz in a sphere of radius `R`,

```math
\mathbf M(\rho)
=
M_s
\left[
\tanh\!\left(\nu\frac{\rho}{R}\right)\mathbf e_\phi
+
\mathrm{sech}\!\left(\nu\frac{\rho}{R}\right)\mathbf e_z
\right],
```

with normalized magnetization

```math
\mathbf m=\frac{\mathbf M}{M_s}.
```

The profile functions are dimensionless functions of the vortex parameter
`\nu`. They are obtained by inserting this Ansatz into the full micromagnetic
Hamiltonian and carrying out all angular integrations that can be done
analytically.

## Cylindrical Reduction

For any radial function `F(\rho/R)` in the sphere,

```math
\frac{1}{V_s}\int_V F(\rho/R)\,d^3r
=
3\int_0^1 x\sqrt{1-x^2}\,F(x)\,dx,
\qquad
x=\frac{\rho}{R},
```

because

```math
\int_V F(\rho/R)\,d^3r
=
4\pi R^3\int_0^1 x\sqrt{1-x^2}\,F(x)\,dx,
\qquad
V_s=\frac{4\pi R^3}{3}.
```

This gives the longitudinal anisotropy and Zeeman profiles directly:

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
\mathrm{sech}(\nu x)\,dx.
```

The transverse anisotropy profile is

```math
g_u^x(\nu)
=
4\int_0^1
x\sqrt{1-x^2}\,
\tanh^2(\nu x)\,dx.
```

Using

```math
\tanh^2 u=1-\mathrm{sech}^2 u
```

and

```math
\int_0^1 x\sqrt{1-x^2}\,dx=\frac{1}{3},
```

one obtains the useful identity

```math
g_u^x(\nu)
=
\frac{4}{3}\left[1-g_u^z(\nu)\right].
```

## Exchange Profile

For the hyperbolic vortex, the exchange term separates into the radial
variation of the profile and the azimuthal winding contribution. In reduced
form this yields

```math
g_{\mathrm{ex}}(\nu)
=
3\int_0^1
\sqrt{1-x^2}
\left[
\nu^2 x\,\mathrm{sech}^2(\nu x)
+
\frac{\tanh^2(\nu x)}{x}
\right]dx.
```

The apparent singularity at `x=0` is removable, since

```math
\frac{\tanh^2(\nu x)}{x}
=
\nu^2x+\mathcal O(x^3).
```

## Magnetostatic Profile

For the vortex Ansatz the volume charge vanishes:

```math
\rho_m=-\nabla\cdot\mathbf M=0.
```

The magnetostatic contribution is therefore determined by the surface charge

```math
\sigma_m=\mathbf M\cdot\mathbf n.
```

On the sphere, with `\mu=\cos\theta`, this becomes

```math
\frac{\sigma_m(\mu)}{M_s}
=
s_\nu(\mu)
=
\mu\,\mathrm{sech}\!\left(\nu\sqrt{1-\mu^2}\right).
```

The function `s_\nu(\mu)` is odd in `\mu`, so only odd Legendre moments
contribute:

```math
I_\ell(\nu)
=
\int_{-1}^{1}
\mu\,\mathrm{sech}\!\left(\nu\sqrt{1-\mu^2}\right)
P_\ell(\mu)\,d\mu,
\qquad
I_\ell(\nu)=0\quad\text{for even }\ell.
```

The positive demagnetizing-field energy coefficient is diagonal in these
Legendre moments,

```math
g_H(\nu)
=
\frac{3}{8}
\sum_{\substack{\ell=1\\ \ell\ \mathrm{odd}}}^{\infty}
I_\ell(\nu)^2.
```

Using the symmetry of the integrand and the substitution
`x=\sqrt{1-\mu^2}`, the moment form used in the implementation is

```math
J_\ell(\nu)
=
\int_0^1
x\,\mathrm{sech}(\nu x)\,
P_\ell\!\left(\sqrt{1-x^2}\right)\,dx,
\qquad
\ell\ \mathrm{odd},
```

with

```math
g_H(\nu)
=
\frac{3}{2}
\sum_{\substack{\ell=1\\ \ell\ \mathrm{odd}}}^{\infty}
J_\ell(\nu)^2.
```

The magnetostatic energy in the original `B`-field convention is

```math
E_s
=
-\frac{1}{2}\int_V \mathbf M\cdot\mathbf B_d\,d^3r,
\qquad
\mathbf B_d=\mu_0(\mathbf H_d+\mathbf M).
```

Since `|\mathbf M|=M_s`, this differs from the positive `H`-field
demagnetizing contribution by the constant `-\mu_0M_s^2V_s/2`. Therefore

```math
g_s(\nu)=g_H(\nu)-\frac{1}{2},
\qquad
g_{\mathrm{dem}}(\nu)=-g_s(\nu)=\frac{1}{2}-g_H(\nu).
```

This gives the final positive demagnetizing profile used in the reduced
Hamiltonian:

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
P_\ell\!\left(\sqrt{1-x^2}\right)\,dx
\right]^2.
```

## Uniform Limit

At `\nu=0`,

```math
g_{\mathrm{ex}}(0)=0,
\qquad
g_u^x(0)=0,
\qquad
g_u^z(0)=g_z^z(0)=1.
```

For the magnetostatic term, `s_0(\mu)=\mu=P_1(\mu)`, so

```math
I_1(0)=\int_{-1}^{1}\mu^2\,d\mu=\frac{2}{3},
```

and therefore

```math
g_H(0)=\frac{1}{6},
\qquad
g_s(0)=-\frac{1}{3},
\qquad
g_{\mathrm{dem}}(0)=\frac{1}{3}.
```

These values are used as numerical checks in the test suite.
