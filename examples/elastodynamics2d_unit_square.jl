using JuliaDG
using Gridap

model = unit_square_model(16, 16)
material = Elastodynamics.Material(1.0, 1.0, 0.5)
zero_vector(x) = VectorValue(0.0, 0.0)
zero_displacement(t, x) = VectorValue(0.0, 0.0)
initial_displacement(x) = VectorValue(sin(pi * x[1]) * sin(pi * x[2]), 0.0)

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
