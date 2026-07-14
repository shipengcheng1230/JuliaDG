using Test
using JuliaDG
using Gridap

const Poisson = JuliaDG.Poisson

function tagged_unit_square_model(nx::Integer = 8, ny::Integer = 8)
    model = unit_square_model(nx, ny)
    labels = Gridap.get_face_labeling(model)
    topology = Gridap.Geometry.get_grid_topology(model)

    left = Gridap.Geometry.face_labeling_from_vertex_filter(
        topology,
        "left",
        x -> isapprox(x[1], 0.0; atol = 1.0e-12),
    )
    right = Gridap.Geometry.face_labeling_from_vertex_filter(
        topology,
        "right",
        x -> isapprox(x[1], 1.0; atol = 1.0e-12),
    )
    merge!(labels, left, right)
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

    affine_result =
        Poisson.solve(model, zero_source; dirichlet_tags = "boundary", dirichlet = affine)
    @test affine_result.model === model
    @test affine_result.order == 1
    @test Poisson.l2_error(affine_result, affine) < 1.0e-10

    exact(x) = sin(pi * x[1]) * sin(pi * x[2])
    source(x) = 2 * pi^2 * exact(x)
    coarse = Poisson.solve(
        unit_square_model(8, 8),
        source;
        dirichlet_tags = "boundary",
        dirichlet = x -> 0.0,
    )
    fine = Poisson.solve(
        unit_square_model(16, 16),
        source;
        dirichlet_tags = "boundary",
        dirichlet = x -> 0.0,
    )
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

    @test_throws ArgumentError Poisson.solve(
        model,
        zero_source;
        dirichlet_tags = "missing",
        dirichlet = x -> 0.0,
    )
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
    @test !isdefined(Poisson, :plot_data)
end
