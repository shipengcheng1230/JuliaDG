# JuliaDG

[![CI](https://github.com/shipengcheng1230/JuliaDG/actions/workflows/ci.yml/badge.svg)](https://github.com/shipengcheng1230/JuliaDG/actions/workflows/ci.yml)

JuliaDG is a minimal Julia package for 2D Poisson and first-order 2D elastodynamics, not a generic PDE framework.

```text
-Delta u = f  in [0,1]^2
u = g         on the boundary
```

## Scope

V1 intentionally stays small:

- P1 discontinuous triangular elements.
- Structured unit-square meshes split into two triangles per rectangle.
- SIPG interior face terms.
- Weak Dirichlet boundary data through Nitsche/SIPG boundary terms.
- Sparse assembly and direct solution with `A \ b`.
- Mesh construction through Meshes.jl, with Makie plotting still optional.
- No mesh files, high-order elements, or generic PDE framework.

## API

```julia
using JuliaDG

mesh = JuliaDG.unit_square_mesh(8, 8)
A, b = JuliaDG.Poisson.assemble(mesh, f, g; penalty=20.0)
result = JuliaDG.Poisson.solve(f; nx=8, ny=8, g=(x, y) -> 0.0, penalty=20.0)
custom_result = JuliaDG.Poisson.solve(f; mesh=mesh, g=(x, y) -> 0.0, penalty=20.0)
u_xy = JuliaDG.Poisson.evaluate(result, x, y)
err = JuliaDG.Poisson.l2_error(result, exact)
```

Main containers:

```julia
JuliaDG.Poisson.Result(mesh, coeffs, A, b)
JuliaDG.Elastodynamics.Result(mesh, state, material, times, energy_history, boundary, state_history)
```

Both solvers accept `Meshes.Mesh`; `JuliaDG.unit_square_mesh` is only a convenience constructor and JuliaDG defines no mesh type of its own. Each triangle owns three DG degrees of freedom.

## SIPG Form

For interior facets, with normal `n` oriented from the left triangle to the right triangle:

```text
[u] = u_left - u_right
{grad u . n} = 0.5 * (grad u_left . n + grad u_right . n)
```

The bilinear form is:

```text
a(u, v) =
    sum_K int_K grad u . grad v
  - sum_F int_F {grad u . n} [v]
  - sum_F int_F {grad v . n} [u]
  + sum_F int_F (penalty / h_F) [u] [v]
```

On boundary facets, the same structure is used with `u_right = g` and `v_right = 0`, giving the Dirichlet contribution:

```text
int_boundary (-grad u . n v - grad v . n u + penalty / h_F u v)
= int_boundary (-grad v . n g + penalty / h_F g v)
```

## Manufactured Solution Example

```julia
using JuliaDG

exact(x, y) = sin(pi * x) * sin(pi * y)
f(x, y) = 2 * pi^2 * exact(x, y)

result = JuliaDG.Poisson.solve(f; nx=8, ny=8, g=(x, y) -> 0.0, penalty=20.0)
println("DOFs: ", length(result.coeffs))
println("L2 error: ", JuliaDG.Poisson.l2_error(result, exact))
```

Run it with:

```bash
julia --project=. examples/poisson2d_unit_square.jl
```

## 2D Elastodynamics Example

JuliaDG also includes a focused first-order velocity-stress solver for constant-material isotropic 2D elastodynamics on the same P1 triangular unit-square meshes:

```text
rho * dt(vx) = dx(sxx) + dy(sxy)
rho * dt(vy) = dx(sxy) + dy(syy)
dt(sxx) = (lambda + 2mu) * dx(vx) + lambda * dy(vy)
dt(syy) = lambda * dx(vx) + (lambda + 2mu) * dy(vy)
dt(sxy) = mu * (dy(vx) + dx(vy))
```

Minimal Gaussian pulse:

```julia
using JuliaDG

material = JuliaDG.Elastodynamics.Material(1.0, 1.0, 0.5)

initial_pulse(x, y) = (
    vx=0.0,
    vy=exp(-120 * ((x - 0.5)^2 + (y - 0.5)^2)),
    sxx=0.0,
    syy=0.0,
    sxy=0.0,
)

result = JuliaDG.Elastodynamics.solve(
    initial_pulse;
    nx=8,
    ny=8,
    material=material,
    tspan=(0.0, 0.02),
    boundary=:reflecting,
)

println("Final energy: ", JuliaDG.Elastodynamics.energy(result))
```

V1 uses constant `JuliaDG.Elastodynamics.Material(rho, lambda, mu)` values and supports `:reflecting` and `:traction_free` boundaries. Displacement output, spatially varying material, source terms, absorbing boundaries, mesh-file input, and higher-order elements are outside this first version.

## Optional Visualization

JuliaDG keeps Makie optional. Install and load a Makie backend such as CairoMakie or GLMakie before calling `JuliaDG.Poisson.plot`:

```julia
using JuliaDG
using CairoMakie

exact(x, y) = sin(pi * x) * sin(pi * y)
f(x, y) = 2 * pi^2 * exact(x, y)

result = JuliaDG.Poisson.solve(f; nx=16, ny=16)
fig = JuliaDG.Poisson.plot(result)
save("solution.png", fig)
```

The plotting helper duplicates cell vertices so discontinuous Galerkin jumps remain visible instead of being averaged across neighboring cells.
