# JuliaDG

JuliaDG is a focused Gridap.jl wrapper for SIPG 2D Poisson and conforming second-order displacement elastodynamics.

## Model contract

Every solver accepts a `Gridap.DiscreteModel`. Face labels on that model select every Dirichlet, Neumann, and traction boundary condition; the selected labels must identify physical boundary faces.

## Unit-square model

```julia
using JuliaDG

model = JuliaDG.unit_square_model(16, 16)
```

`unit_square_model` provides the Gridap face label `"boundary"` for the complete boundary of `[0, 1]^2`.

## Poisson

`Poisson.solve` uses the symmetric interior-penalty Galerkin method. It takes a scalar source field, Dirichlet labels and a Dirichlet callback, and accepts optional `neumann_tags` and `neumann` callbacks for labeled Neumann faces. The `penalty` keyword defaults to `10.0` and can be increased if a mesh requires stronger interface stabilization.

```julia
using JuliaDG
using Gridap

model = unit_square_model(16, 16)
exact(x) = sin(pi * x[1]) * sin(pi * x[2])
source(x) = 2 * pi^2 * exact(x)

result = Poisson.solve(
    model,
    source;
    dirichlet_tags = "boundary",
    dirichlet = x -> 0.0,
)

domain = Triangulation(model)
writevtk(domain, "poisson_solution", cellfields = ["u" => result.solution])
println("L2 error: ", Poisson.l2_error(result, exact))
```

Run the example with:

```bash
julia --project=. examples/poisson2d_unit_square.jl
```

## Plane-strain elastodynamics

JuliaDG solves the displacement form of plane-strain elastodynamics:

```text
rho * d2u/dt2 - div(sigma(u)) = body_force
sigma(u) = lambda * tr(epsilon(u)) * I + 2 * mu * epsilon(u)
```

The `displacement` callback and `dirichlet_tags` prescribe displacement on labeled faces. Optional `traction` and `traction_tags` callbacks prescribe traction on labeled faces. Supply `initial_displacement` and `initial_velocity` to define the initial state. Time integration uses average-acceleration Newmark.

```julia
using JuliaDG
using Gridap

model = unit_square_model(16, 16)
material = Elastodynamics.Material(1.0, 1.0, 0.5)
zero_vector(x) = VectorValue(0.0, 0.0)
zero_displacement(t, x) = VectorValue(0.0, 0.0)
initial_displacement(x) = VectorValue(
    sin(pi * x[1]) * sin(pi * x[2]),
    0.0,
)

result = Elastodynamics.solve(
    model;
    material = material,
    tspan = (0.0, 0.02),
    dt = 0.01,
    dirichlet_tags = "boundary",
    displacement = zero_displacement,
    initial_displacement = initial_displacement,
    initial_velocity = zero_vector,
)

domain = Triangulation(model)
for (step, (_, displacement)) in enumerate(result.solution)
    name = "elastodynamics_step_$(step)"
    writevtk(
        domain,
        name,
        cellfields = [
            "displacement" => displacement,
            "stress" => Elastodynamics.stress(displacement, material),
        ],
    )
end

println(
    "Initial energy: ",
    Elastodynamics.energy(
        result.initial_displacement,
        result.initial_velocity,
        material,
        model,
    ),
)
```

Run the example with:

```bash
julia --project=. examples/elastodynamics2d_unit_square.jl
```

## VTK output

Both examples use `Gridap.writevtk` on `Triangulation(model)`. The Poisson example writes the scalar solution, while the elastodynamics example writes displacement and derived stress for each time step. Run examples from a directory where their VTK output should be created.
