using Test
using JuliaDG
using Gridap

const Poisson = JuliaDG.Poisson

function tagged_unit_square_model(nx::Integer = 8, ny::Integer = 8)
    model = unit_square_model(nx, ny)
    labels = Gridap.get_face_labeling(model)
    topology = Gridap.Geometry.get_grid_topology(model)

    left = Gridap.Geometry.face_labeling_from_vertex_filter(topology, "left", x -> isapprox(x[1], 0.0; atol = 1.0e-12))
    right =
        Gridap.Geometry.face_labeling_from_vertex_filter(topology, "right", x -> isapprox(x[1], 1.0; atol = 1.0e-12))
    middle =
        Gridap.Geometry.face_labeling_from_vertex_filter(topology, "middle", x -> isapprox(x[1], 0.5; atol = 1.0e-12))
    mixed = Gridap.Geometry.face_labeling_from_vertex_filter(
        topology,
        "mixed",
        x -> isapprox(x[1], 0.0; atol = 1.0e-12) || isapprox(x[1], 0.5; atol = 1.0e-12),
    )
    merge!(labels, left, right, middle, mixed)
    return model
end

@testset "Gridap model and tag contract" begin
    model = unit_square_model(2, 3)
    @test model isa Gridap.DiscreteModel
    @test Gridap.num_dims(model) == 2
    @test "boundary" in Gridap.Geometry.get_tag_name(Gridap.get_face_labeling(model))
    @test_throws ArgumentError unit_square_model(0, 3)
    @test_throws ArgumentError unit_square_model(2, 0)
end

@testset "conforming Poisson" begin
    model = unit_square_model(4, 4)
    affine(x) = 1.0 + x[1] + 2.0 * x[2]
    zero_source(x) = 0.0

    affine_result = Poisson.solve(model, zero_source; dirichlet_tags = "boundary", dirichlet = affine)
    @test affine_result.model === model
    @test affine_result.order == 1
    @test Poisson.l2_error(affine_result, affine) < 1.0e-10

    exact(x) = sin(pi * x[1]) * sin(pi * x[2])
    source(x) = 2 * pi^2 * exact(x)
    coarse = Poisson.solve(unit_square_model(8, 8), source; dirichlet_tags = "boundary", dirichlet = x -> 0.0)
    fine = Poisson.solve(unit_square_model(16, 16), source; dirichlet_tags = "boundary", dirichlet = x -> 0.0)
    @test Poisson.l2_error(fine, exact) < Poisson.l2_error(coarse, exact)

    tagged_model = tagged_unit_square_model()
    loaded = Poisson.solve(
        tagged_model,
        zero_source;
        dirichlet_tags = "left",
        dirichlet = x -> 0.0,
        neumann_tags = "right",
        neumann = x -> 1.0,
    )
    @test Poisson.l2_error(loaded, x -> x[1]) < 1.0e-10

    interior_model = tagged_unit_square_model(2, 2)
    @test_throws ArgumentError Poisson.solve(
        interior_model,
        zero_source;
        dirichlet_tags = "middle",
        dirichlet = x -> 0.0,
    )
    @test_throws ArgumentError Poisson.solve(
        interior_model,
        zero_source;
        dirichlet_tags = "mixed",
        dirichlet = x -> 0.0,
    )

    @test_throws ArgumentError Poisson.solve(model, zero_source; dirichlet_tags = "missing", dirichlet = x -> 0.0)
    @test_throws ArgumentError Poisson.solve(
        model,
        zero_source;
        dirichlet_tags = "boundary",
        dirichlet = x -> 0.0,
        neumann_tags = "boundary",
        neumann = x -> 0.0,
    )
    @test !isdefined(Poisson, :assemble)
    @test !isdefined(Poisson, :evaluate)
    @test !isdefined(Poisson, :plot)
end

const Elastodynamics = JuliaDG.Elastodynamics
zero_vector(x) = VectorValue(0.0, 0.0)
zero_time_vector(t, x) = VectorValue(0.0, 0.0)

@testset "transient conforming elastodynamics" begin
    material = Elastodynamics.Material(1.0, 1.0, 0.5)
    model = unit_square_model(4, 4)

    zero_result = Elastodynamics.solve(
        model;
        material = material,
        tspan = (0.0, 0.02),
        dt = 0.01,
        dirichlet_tags = "boundary",
        displacement = zero_time_vector,
        initial_displacement = zero_vector,
        initial_velocity = zero_vector,
    )
    @test zero_result.model === model
    @test zero_result.material === material
    @test zero_result.order == 1
    for (_, displacement) in zero_result.solution
        @test all(iszero, Gridap.get_free_dof_values(displacement))
    end

    pulse(x) = VectorValue(sin(pi * x[1]) * sin(pi * x[2]), 0.0)
    pulse_result = Elastodynamics.solve(
        model;
        material = material,
        tspan = (0.0, 0.02),
        dt = 0.01,
        dirichlet_tags = "boundary",
        displacement = zero_time_vector,
        initial_displacement = pulse,
        initial_velocity = zero_vector,
    )
    @test Elastodynamics.energy(pulse_result.initial_displacement, pulse_result.initial_velocity, material, model) > 0.0
    for (_, displacement) in pulse_result.solution
        @test all(isfinite, Gridap.get_free_dof_values(displacement))
    end

    loaded = Elastodynamics.solve(
        tagged_unit_square_model();
        material = material,
        tspan = (0.0, 0.02),
        dt = 0.01,
        dirichlet_tags = "left",
        displacement = zero_time_vector,
        traction_tags = "right",
        traction = (t, x) -> VectorValue(1.0, 0.0),
        initial_displacement = zero_vector,
        initial_velocity = zero_vector,
    )
    @test any(
        !iszero(value) for (_, displacement) in loaded.solution for value in Gridap.get_free_dof_values(displacement)
    )

    @test_throws ArgumentError Elastodynamics.Material(0.0, 1.0, 0.5)
    @test_throws ArgumentError Elastodynamics.Material(1.0, -1.0, 0.5)
    @test_throws ArgumentError Elastodynamics.Material(1.0, 1.0, 0.0)
    @test_throws ArgumentError Elastodynamics.solve(
        model;
        material = material,
        tspan = (0.0, 0.0),
        dt = 0.01,
        dirichlet_tags = "boundary",
        displacement = zero_time_vector,
        initial_displacement = zero_vector,
        initial_velocity = zero_vector,
    )
    @test_throws ArgumentError Elastodynamics.solve(
        model;
        material = material,
        tspan = (0.0, 0.02),
        dt = 0.0,
        dirichlet_tags = "boundary",
        displacement = zero_time_vector,
        initial_displacement = zero_vector,
        initial_velocity = zero_vector,
    )
    @test_throws ArgumentError Elastodynamics.solve(
        model;
        material = material,
        tspan = (0.0, 0.02),
        dt = 0.01,
        dirichlet_tags = "boundary",
        displacement = zero_time_vector,
        traction_tags = "boundary",
        traction = zero_time_vector,
        initial_displacement = zero_vector,
        initial_velocity = zero_vector,
    )

    three_dimensional_model = Gridap.CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (1, 1, 1))
    @test_throws ArgumentError Elastodynamics.solve(
        three_dimensional_model;
        material = material,
        tspan = (0.0, 0.02),
        dt = 0.01,
        dirichlet_tags = "boundary",
        displacement = zero_time_vector,
        initial_displacement = zero_vector,
        initial_velocity = zero_vector,
    )
    @test_throws ArgumentError Elastodynamics.energy(nothing, nothing, material, three_dimensional_model)
    @test !isdefined(Elastodynamics, :rhs)
    @test !isdefined(Elastodynamics, :evaluate)
    @test !isdefined(Elastodynamics, :plot)
    @test !isdefined(Elastodynamics, :record)
end
