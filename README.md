# JuliaDG

[![CI](https://github.com/shipengcheng1230/JuliaDG/actions/workflows/ci.yml/badge.svg)](https://github.com/shipengcheng1230/JuliaDG/actions/workflows/ci.yml)

JuliaDG is a minimal Julia package for solving the 2D Poisson equation on the unit square with a P1 discontinuous Galerkin SIPG method.

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
- No plotting, mesh files, external dependencies, high-order elements, or generic PDE framework.

## API

```julia
mesh = unit_square_mesh(nx, ny)
A, b = assemble_poisson_sipg(mesh, f, g; penalty=20.0)
result = solve_poisson(f; nx=8, ny=8, g=(x, y) -> 0.0, penalty=20.0)
u_xy = evaluate_solution(result, x, y)
err = l2_error(result, exact)
```

Main containers:

```julia
TriMesh(vertices, cells, faces)
DGResult(mesh, coeffs, A, b)
```

`TriMesh.vertices` is a `2 x nvertices` coordinate matrix. `TriMesh.cells` is a `3 x ncells` matrix of vertex indices. Each triangle owns three DG degrees of freedom.

## SIPG Form

For interior faces, with normal `n` oriented from the left cell to the right cell:

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

On boundary faces, the same structure is used with `u_right = g` and `v_right = 0`, giving the Dirichlet contribution:

```text
int_boundary (-grad u . n v - grad v . n u + penalty / h_F u v)
= int_boundary (-grad v . n g + penalty / h_F g v)
```

## Manufactured Solution Example

```julia
using JuliaDG

exact(x, y) = sin(pi * x) * sin(pi * y)
f(x, y) = 2 * pi^2 * exact(x, y)

result = solve_poisson(f; nx=8, ny=8, g=(x, y) -> 0.0, penalty=20.0)
println("DOFs: ", length(result.coeffs))
println("L2 error: ", l2_error(result, exact))
```

Run it with:

```bash
julia --project=. examples/poisson2d_unit_square.jl
```

## Optional Visualization

JuliaDG keeps Makie optional. Install and load a Makie backend such as CairoMakie or GLMakie before calling `plot_solution`:

```julia
using JuliaDG
using CairoMakie

exact(x, y) = sin(pi * x) * sin(pi * y)
f(x, y) = 2 * pi^2 * exact(x, y)

result = solve_poisson(f; nx=16, ny=16)
fig = plot_solution(result)
save("solution.png", fig)
```

The plotting helper duplicates cell vertices so discontinuous Galerkin jumps remain visible instead of being averaged across neighboring cells.
