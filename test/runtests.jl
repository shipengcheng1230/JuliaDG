using LinearAlgebra
using SparseArrays
using Test
using JuliaDG

@testset "mesh" begin
    mesh = unit_square_mesh(1, 1)
    @test size(mesh.vertices) == (2, 4)
    @test size(mesh.cells) == (3, 2)
    @test length(mesh.faces) == 5
    @test count(face -> face.cells[2] == 0, mesh.faces) == 4
    @test count(face -> face.cells[2] != 0, mesh.faces) == 1

    mesh2 = unit_square_mesh(2, 1)
    @test size(mesh2.vertices) == (2, 6)
    @test size(mesh2.cells) == (3, 4)
    @test length(mesh2.faces) == 9
    @test count(face -> face.cells[2] == 0, mesh2.faces) == 6
    @test count(face -> face.cells[2] != 0, mesh2.faces) == 3

    @test_throws ArgumentError unit_square_mesh(0, 1)
    @test_throws ArgumentError unit_square_mesh(1, 0)
end

@testset "elastic state layout" begin
    material = ElasticMaterial(1.0, 2.0, 3.0)
    @test material.rho == 1.0
    @test material.lambda == 2.0
    @test material.mu == 3.0

    for args in ((0.0, 1.0, 1.0), (1.0, -0.1, 1.0), (1.0, 1.0, 0.0))
        try
            ElasticMaterial(args...)
            @test false
        catch err
            @test err isa ArgumentError
        end
    end

    mesh = unit_square_mesh(1, 1)
    ncells = size(mesh.cells, 2)
    @test JuliaDG.elastic_dof(1, 1, 1, ncells) == 1
    @test JuliaDG.elastic_dof(1, 3, 1, ncells) == 3
    @test JuliaDG.elastic_dof(2, 1, 1, ncells) == 4
    @test JuliaDG.elastic_dof(1, 1, 2, ncells) == 7
    @test JuliaDG.elastic_dof(2, 3, 5, ncells) == 30

    named_initial = (x, y) -> (vx=x, vy=y, sxx=x + y, syy=x - y, sxy=2 * x - y)
    tuple_initial = (x, y) -> (x, y, x + y, x - y, 2 * x - y)
    named_state = JuliaDG.interpolate_elastic_state(named_initial, mesh)
    tuple_state = JuliaDG.interpolate_elastic_state(tuple_initial, mesh)

    @test length(named_state) == 5 * 3 * ncells
    @test tuple_state == named_state
    @test JuliaDG.validate_elastic_boundary(:reflecting) == :reflecting
    @test JuliaDG.validate_elastic_boundary(:traction_free) == :traction_free

    try
        JuliaDG.validate_elastic_boundary(:periodic)
        @test false
    catch err
        @test err isa ArgumentError
        @test err.msg == "boundary must be :reflecting or :traction_free"
    end
end

@testset "elastic residual" begin
    mesh = unit_square_mesh(2, 2)
    material = ElasticMaterial(1.0, 1.0, 0.5)
    ndofs = 5 * 3 * size(mesh.cells, 2)
    state = zeros(ndofs)
    normal = (1.0, 0.0)
    interior = (1.0, 2.0, 3.0, 4.0, 5.0)

    @test JuliaDG.pressure_wave_speed(material) ≈ sqrt(2.0)
    @test JuliaDG.elastic_rhs(state, mesh, material, :reflecting) ≈ zeros(ndofs)
    @test JuliaDG.elastic_rhs(state, mesh, material, :traction_free) ≈ zeros(ndofs)
    @test JuliaDG.boundary_state(interior, normal, :reflecting) == (-1.0, 2.0, 3.0, 4.0, -5.0)
    @test JuliaDG.boundary_state(interior, normal, :traction_free) == (1.0, 2.0, -3.0, 4.0, -5.0)
    @test collect(
        JuliaDG.normal_flux(
            interior,
            JuliaDG.boundary_state(interior, normal, :reflecting),
            normal,
            material,
        ),
    ) ≈ [3 + sqrt(2.0), 0.0, 0.0, 0.0, 1 + 5 * sqrt(2.0)]

    mass_mesh = unit_square_mesh(1, 1)
    mass_ncells = size(mass_mesh.cells, 2)
    residual = zeros(5 * 3 * mass_ncells)
    residual[JuliaDG.elastic_dof(1, 1, 1, mass_ncells)] = 1.0
    residual[JuliaDG.elastic_dof(1, 2, 1, mass_ncells)] = 2.0
    residual[JuliaDG.elastic_dof(1, 3, 1, mass_ncells)] = 3.0
    mass_rhs = JuliaDG.apply_elastic_mass_inverse(residual, mass_mesh, mass_ncells)
    @test mass_rhs[JuliaDG.elastic_dof(1, 1, 1, mass_ncells)] ≈ -12.0
    @test mass_rhs[JuliaDG.elastic_dof(1, 2, 1, mass_ncells)] ≈ 12.0
    @test mass_rhs[JuliaDG.elastic_dof(1, 3, 1, mass_ncells)] ≈ 36.0
    @test all(iszero, mass_rhs[(JuliaDG.elastic_dof(1, 3, 1, mass_ncells) + 1):end])

    try
        JuliaDG.elastic_rhs(state[1:(end - 1)], mesh, material, :reflecting)
        @test false
    catch err
        @test err isa ArgumentError
        @test err.msg == "elastic state length does not match mesh"
    end
end

@testset "elastic postprocess" begin
    mesh = unit_square_mesh(1, 1)
    material = ElasticMaterial(1.0, 1.0, 0.5)
    initial = (x, y) -> (vx=x, vy=y, sxx=0.1 + x, syy=0.2 + y, sxy=0.3)
    state = JuliaDG.interpolate_elastic_state(initial, mesh)
    energy = JuliaDG.elastic_energy(mesh, state, material)
    result = ElasticResult(mesh, state, material, [0.0], [energy], :reflecting)

    value = evaluate_elastic_state(result, 0.25, 0.25)
    @test propertynames(value) == (:vx, :vy, :sxx, :syy, :sxy)
    @test all(isfinite, Tuple(value))
    @test value.vx ≈ 0.25
    @test value.vy ≈ 0.25

    @test isfinite(energy)
    @test energy >= 0.0
    @test elastic_energy(result) ≈ energy

    zero_state = zeros(length(state))
    zero_result = ElasticResult(mesh, zero_state, material, [0.0], [0.0], :reflecting)
    @test elastic_energy(zero_result) ≈ 0.0

    try
        evaluate_elastic_state(result, -0.1, 0.2)
        @test false
    catch err
        @test err isa ArgumentError
        @test err.msg == "point is outside the mesh"
    end
end

@testset "elastic solve" begin
    zero_initial = (x, y) -> (vx=0.0, vy=0.0, sxx=0.0, syy=0.0, sxy=0.0)

    for boundary in (:reflecting, :traction_free)
        result = solve_elastodynamics(
            zero_initial;
            nx=2,
            ny=2,
            tspan=(0.0, 0.03),
            dt=0.02,
            boundary=boundary,
        )

        @test result.boundary == boundary
        @test result.times[1] == 0.0
        @test result.times[end] == 0.03
        @test length(result.energy_history) == length(result.times)
        @test norm(result.state) ≈ 0.0 atol = 1.0e-12
        @test result.energy_history[end] ≈ 0.0 atol = 1.0e-12
    end

    pulse = (x, y) -> begin
        amplitude = exp(-80 * ((x - 0.5)^2 + (y - 0.5)^2))
        (vx=amplitude, vy=0.0, sxx=0.0, syy=0.0, sxy=0.0)
    end
    pulse_result = solve_elastodynamics(
        pulse;
        nx=2,
        ny=2,
        tspan=(0.0, 0.005),
        dt=0.002,
        boundary=:reflecting,
    )
    value = evaluate_elastic_state(pulse_result, 0.5, 0.5)

    @test pulse_result.times[end] == 0.005
    @test all(isfinite, Tuple(value))
    @test isfinite(elastic_energy(pulse_result))
    @test elastic_energy(pulse_result) >= 0.0

    try
        solve_elastodynamics(zero_initial; nx=1, ny=1, tspan=(0.0, 0.0), boundary=:periodic)
        @test false
    catch err
        @test err isa ArgumentError
        @test err.msg == "boundary must be :reflecting or :traction_free"
    end
end

@testset "SIPG assembly" begin
    mesh = unit_square_mesh(2, 2)
    f = (x, y) -> 1.0
    g = (x, y) -> 0.0

    A, b = assemble_poisson_sipg(mesh, f, g; penalty=20.0)
    ndofs = 3 * size(mesh.cells, 2)

    @test size(A) == (ndofs, ndofs)
    @test A isa SparseMatrixCSC{Float64,Int}
    @test length(b) == ndofs
    @test isapprox(norm(Matrix(A - transpose(A))), 0.0; atol=1.0e-10)

    solution = A \ b
    @test all(isfinite, solution)

    zero_f = (x, y) -> 0.0
    zero_g = (x, y) -> 0.0
    A0, b0 = assemble_poisson_sipg(mesh, zero_f, zero_g; penalty=20.0)
    @test isapprox(norm(A0 \ b0), 0.0; atol=1.0e-12)
end

@testset "basis geometry" begin
    mesh = unit_square_mesh(1, 1)
    coords = JuliaDG.cell_coordinates(mesh, 1)
    area, grads = JuliaDG.triangle_geometry(coords)

    @test area ≈ 0.5
    @test grads[:, 1] + grads[:, 2] + grads[:, 3] ≈ zeros(2)

    centroid = (
        sum(coords[1, i] for i in 1:3) / 3,
        sum(coords[2, i] for i in 1:3) / 3,
    )
    lambdas = JuliaDG.barycentric_coordinates(coords, centroid[1], centroid[2])
    @test collect(lambdas) ≈ fill(1 / 3, 3)

    vertex_values = JuliaDG.basis_values_at_point(coords, coords[1, 1], coords[2, 1])
    @test vertex_values ≈ [1.0, 0.0, 0.0]

    normal, edge_len = JuliaDG.edge_normal(mesh, 1, 1)
    @test collect(normal) ≈ [0.0, -1.0]
    @test edge_len ≈ 1.0

    x, y = JuliaDG.edge_point(mesh, 1, 1, 0.25)
    @test x ≈ 0.25
    @test y ≈ 0.0
end

@testset "solve and postprocess" begin
    exact = (x, y) -> sin(pi * x) * sin(pi * y)
    f = (x, y) -> 2 * pi^2 * exact(x, y)

    coarse = solve_poisson(f; nx=3, ny=3, g=(x, y) -> 0.0, penalty=30.0)
    fine = solve_poisson(f; nx=6, ny=6, g=(x, y) -> 0.0, penalty=30.0)

    coarse_error = l2_error(coarse, exact)
    fine_error = l2_error(fine, exact)

    @test fine_error < coarse_error
    @test evaluate_solution(fine, 0.5, 0.5) isa Float64

    affine = (x, y) -> 1.0 + x + 2.0 * y
    zero_f = (x, y) -> 0.0
    affine_result = solve_poisson(zero_f; nx=3, ny=3, g=affine, penalty=30.0)

    @test l2_error(affine_result, affine) < 1.0e-8
    @test_throws ArgumentError evaluate_solution(affine_result, -0.1, 0.2)
end

@testset "plot data" begin
    mesh = unit_square_mesh(1, 1)
    ndofs = 3 * size(mesh.cells, 2)
    coeffs = Float64.(1:ndofs)
    result = DGResult(mesh, coeffs, spzeros(ndofs, ndofs), zeros(ndofs))

    data = dg_plot_data(result)

    @test length(data.xs) == ndofs
    @test length(data.ys) == ndofs
    @test length(data.values) == ndofs
    @test length(data.faces) == size(mesh.cells, 2)
    @test data.faces == [(1, 2, 3), (4, 5, 6)]
    @test isempty(intersect(collect(data.faces[1]), collect(data.faces[2])))
    @test data.values == coeffs
    @test data.xs == [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    @test data.ys == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]

    bad_result = DGResult(mesh, coeffs[1:(end - 1)], spzeros(ndofs - 1, ndofs - 1), zeros(ndofs - 1))
    @test_throws ArgumentError dg_plot_data(bad_result)

    try
        plot_solution(result)
        @test false
    catch err
        @test err isa ArgumentError
        @test err.msg ==
              "plot_solution requires Makie; load CairoMakie or GLMakie before calling it"
    end
end

@testset "example script" begin
    package_root = dirname(@__DIR__)
    example_path = joinpath(package_root, "examples", "poisson2d_unit_square.jl")
    output = read(`$(Base.julia_cmd()) --project=$package_root $example_path`, String)

    @test occursin(r"DOFs:\s+384", output)
    @test occursin(r"L2 error:\s+[0-9]", output)
end
