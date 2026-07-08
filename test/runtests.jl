using LinearAlgebra
using SparseArrays
using Test
using JuliaDG
import Meshes

@testset "mesh" begin
    mesh = unit_square_mesh(1, 1)
    @test mesh isa Meshes.Mesh
    @test length(mesh) == JuliaDG.triangle_count(mesh)
    @test getproperty(Meshes.topology(mesh), :connec)[1].indices == (1, 2, 4)
    @test JuliaDG.triangle_connectivities(mesh) == [(1, 2, 4), (1, 4, 3)]
    @test JuliaDG.triangle_count(mesh) == 2
    @test length(JuliaDG.facet_adjacencies(mesh)) == 5
    @test count(facet -> facet.triangles[2] == 0, JuliaDG.facet_adjacencies(mesh)) == 4
    @test count(facet -> facet.triangles[2] != 0, JuliaDG.facet_adjacencies(mesh)) == 1
    @test map(facet -> (facet.point_ids, facet.triangles, facet.local_edges), JuliaDG.facet_adjacencies(mesh)) == [
        ((1, 2), (1, 0), (1, 0)),
        ((2, 4), (1, 0), (2, 0)),
        ((1, 4), (1, 2), (3, 1)),
        ((3, 4), (2, 0), (2, 0)),
        ((1, 3), (2, 0), (3, 0)),
    ]
    @test resolve_mesh(mesh, 9, 9) === mesh
    @test !isdefined(JuliaDG, Symbol("Tri", "Mesh"))
    @test !isdefined(JuliaDG, Symbol("Tri", "Face"))
    @test !isdefined(JuliaDG, Symbol("mesh", "_backend"))

    clockwise_points = [
        Meshes.Point(0.0, 0.0),
        Meshes.Point(1.0, 0.0),
        Meshes.Point(0.0, 1.0),
    ]
    clockwise_connectivities = [Meshes.connect((1, 3, 2), Meshes.Triangle)]
    clockwise = Meshes.SimpleMesh(clockwise_points, clockwise_connectivities)

    @test JuliaDG.triangle_geometry(clockwise, 1)[1] ≈ 0.5

    mesh2 = unit_square_mesh(2, 1)
    @test length(Meshes.vertices(mesh2)) == 6
    @test JuliaDG.triangle_count(mesh2) == 4
    @test length(JuliaDG.facet_adjacencies(mesh2)) == 9
    @test count(facet -> facet.triangles[2] == 0, JuliaDG.facet_adjacencies(mesh2)) == 6
    @test count(facet -> facet.triangles[2] != 0, JuliaDG.facet_adjacencies(mesh2)) == 3

    try
        resolve_mesh("not a mesh", 1, 1)
        @test false
    catch err
        @test err isa ArgumentError
        @test err.msg == "mesh must be nothing or Meshes.Mesh"
    end

    @test_throws ArgumentError unit_square_mesh(0, 1)
    @test_throws ArgumentError unit_square_mesh(1, 0)
end

@testset "Meshes triangle queries" begin
    raw = Meshes.simplexify(Meshes.CartesianGrid((0.0, 0.0), (1.0, 1.0), dims=(1, 1)))

    @test raw isa Meshes.Mesh
    @test !isdefined(JuliaDG, :orient_triangle_points)
    @test !isdefined(JuliaDG, :local_edge_for_points)
    @test !applicable(JuliaDG.facet_adjacencies, [(1, 2, 3)])
    @test JuliaDG.point_xy(raw, 1) == (0.0, 0.0)
    @test JuliaDG.point_xy(raw, 4) == (1.0, 1.0)
    @test JuliaDG.triangle_connectivities(raw) == [(1, 2, 4), (1, 4, 3)]
    @test JuliaDG.oriented_triangle_connectivities(raw) == [(1, 2, 4), (1, 4, 3)]
    @test JuliaDG.triangle_count(raw) == 2
    @test JuliaDG.triangle_points(raw, 1) == (1, 2, 4)
    @test JuliaDG.triangle_points(raw, 2) == (1, 4, 3)

    struct IteratorOnlyConnectivity
        indices::NTuple{3,Int}
    end

    struct IteratorOnlyConnectivities
        items::Vector{IteratorOnlyConnectivity}
    end

    Base.iterate(connectivities::IteratorOnlyConnectivities, state::Int=1) =
        state > length(connectivities.items) ? nothing : (connectivities.items[state], state + 1)
    Base.length(::IteratorOnlyConnectivities) = error("length is not supported")
    Base.getindex(::IteratorOnlyConnectivities, ::Int) = error("indexing is not supported")

    struct IteratorOnlyTopology <: Meshes.Topology
        connec::IteratorOnlyConnectivities
    end

    IteratorOnlyPoint = typeof(Meshes.Point(0.0, 0.0))
    IteratorOnlyCRS = typeof(Meshes.coords(Meshes.Point(0.0, 0.0)))

    struct IteratorOnlyMesh <: Meshes.Mesh{Meshes.𝔼{2}, IteratorOnlyCRS, IteratorOnlyTopology}
        points::Vector{IteratorOnlyPoint}
        topology::IteratorOnlyTopology
    end

    Meshes.vertices(mesh::IteratorOnlyMesh) = mesh.points
    Meshes.topology(mesh::IteratorOnlyMesh) = mesh.topology

    iterator_only = IteratorOnlyMesh(
        [
            Meshes.Point(0.0, 0.0),
            Meshes.Point(1.0, 0.0),
            Meshes.Point(0.0, 1.0),
        ],
        IteratorOnlyTopology(IteratorOnlyConnectivities([IteratorOnlyConnectivity((1, 3, 2))])),
    )

    @test JuliaDG.triangle_points(iterator_only, 1) == (1, 2, 3)

    facets = JuliaDG.facet_adjacencies(raw)
    @test length(facets) == 5
    @test count(facet -> facet.triangles[2] == 0, facets) == 4
    @test count(facet -> facet.triangles[2] != 0, facets) == 1
    @test map(facet -> (facet.point_ids, facet.triangles, facet.local_edges), facets) == [
        ((1, 2), (1, 0), (1, 0)),
        ((2, 4), (1, 0), (2, 0)),
        ((1, 4), (1, 2), (3, 1)),
        ((3, 4), (2, 0), (2, 0)),
        ((1, 3), (2, 0), (3, 0)),
    ]

    clockwise_points = [
        Meshes.Point(0.0, 0.0),
        Meshes.Point(1.0, 0.0),
        Meshes.Point(0.0, 1.0),
    ]
    clockwise_connectivities = [Meshes.connect((1, 3, 2), Meshes.Triangle)]
    clockwise = Meshes.SimpleMesh(clockwise_points, clockwise_connectivities)

    @test JuliaDG.triangle_connectivities(clockwise) == [(1, 3, 2)]
    @test JuliaDG.triangle_points(clockwise, 1) == (1, 2, 3)
    @test JuliaDG.oriented_triangle_connectivities(clockwise) == [(1, 2, 3)]
    @test map(facet -> (facet.point_ids, facet.triangles, facet.local_edges), JuliaDG.facet_adjacencies(clockwise)) == [
        ((1, 2), (1, 0), (1, 0)),
        ((2, 3), (1, 0), (2, 0)),
        ((1, 3), (1, 0), (3, 0)),
    ]
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
    triangle_total = JuliaDG.triangle_count(mesh)
    @test JuliaDG.elastic_dof(1, 1, 1, triangle_total) == 1
    @test JuliaDG.elastic_dof(1, 3, 1, triangle_total) == 3
    @test JuliaDG.elastic_dof(2, 1, 1, triangle_total) == 4
    @test JuliaDG.elastic_dof(1, 1, 2, triangle_total) == 7
    @test JuliaDG.elastic_dof(2, 3, 5, triangle_total) == 30

    named_initial = (x, y) -> (vx=x, vy=y, sxx=x + y, syy=x - y, sxy=2 * x - y)
    tuple_initial = (x, y) -> (x, y, x + y, x - y, 2 * x - y)
    named_state = JuliaDG.interpolate_elastic_state(named_initial, mesh)
    tuple_state = JuliaDG.interpolate_elastic_state(tuple_initial, mesh)

    @test length(named_state) == 5 * 3 * triangle_total
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
    ndofs = 5 * 3 * JuliaDG.triangle_count(mesh)
    state = zeros(ndofs)
    normal = (1.0, 0.0)
    interior = (1.0, 2.0, 3.0, 4.0, 5.0)

    @test JuliaDG.pressure_wave_speed(material) ≈ sqrt(2.0)
    @test JuliaDG.elastic_rhs(state, mesh, material, :reflecting) ≈ zeros(ndofs)
    @test JuliaDG.elastic_rhs(state, mesh, material, :traction_free) ≈ zeros(ndofs)
    triangles = JuliaDG.oriented_triangle_connectivities(mesh)
    facets = JuliaDG.facet_adjacencies(mesh)
    @test !applicable(JuliaDG.interpolate_elastic_state, (x, y) -> (0.0, 0.0, 0.0, 0.0, 0.0), mesh, triangles)
    @test !applicable(JuliaDG.elastic_rhs, state, mesh, triangles, facets, material, :reflecting)
    @test !applicable(JuliaDG.minimum_edge_length, mesh, triangles, facets)
    @test !applicable(JuliaDG.default_elastic_dt, mesh, material, 0.1, triangles, facets)
    @test !applicable(JuliaDG.ssprk3_step, state, 0.01, mesh, triangles, facets, material, :reflecting)
    @test !isdefined(JuliaDG, :add_elastic_volume_terms!)
    @test !isdefined(JuliaDG, :add_elastic_interior_face!)
    @test !isdefined(JuliaDG, :add_elastic_boundary_face!)
    @test !isdefined(JuliaDG, :elastic_triangle_energy)
    @test JuliaDG.boundary_state(interior, normal, :reflecting) == (-1.0, 2.0, 3.0, 4.0, -5.0)
    @test JuliaDG.boundary_state(interior, normal, :traction_free) == (1.0, 2.0, -3.0, 4.0, -5.0)
    reflected = JuliaDG.boundary_state(interior, normal, :reflecting)
    # LF/Rusanov dissipation must oppose the state jump for this residual convention.
    @test collect(JuliaDG.normal_flux(interior, reflected, normal, material)) ≈
          [3 - sqrt(2.0), 0.0, 0.0, 0.0, 1 - 5 * sqrt(2.0)]

    mass_mesh = unit_square_mesh(1, 1)
    mass_triangle_total = JuliaDG.triangle_count(mass_mesh)
    residual = zeros(5 * 3 * mass_triangle_total)
    residual[JuliaDG.elastic_dof(1, 1, 1, mass_triangle_total)] = 1.0
    residual[JuliaDG.elastic_dof(1, 2, 1, mass_triangle_total)] = 2.0
    residual[JuliaDG.elastic_dof(1, 3, 1, mass_triangle_total)] = 3.0
    mass_rhs = JuliaDG.apply_elastic_mass_inverse(residual, mass_mesh, mass_triangle_total)
    @test mass_rhs[JuliaDG.elastic_dof(1, 1, 1, mass_triangle_total)] ≈ -12.0
    @test mass_rhs[JuliaDG.elastic_dof(1, 2, 1, mass_triangle_total)] ≈ 12.0
    @test mass_rhs[JuliaDG.elastic_dof(1, 3, 1, mass_triangle_total)] ≈ 36.0
    @test all(iszero, mass_rhs[(JuliaDG.elastic_dof(1, 3, 1, mass_triangle_total) + 1):end])

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
    result = ElasticResult(mesh, state, material, [0.0], [energy], :reflecting, nothing)

    value = evaluate_elastic_state(result, 0.25, 0.25)
    @test propertynames(value) == (:vx, :vy, :sxx, :syy, :sxy)
    @test all(isfinite, Tuple(value))
    @test value.vx ≈ 0.25
    @test value.vy ≈ 0.25

    @test isfinite(energy)
    @test energy >= 0.0
    @test elastic_energy(result) ≈ energy

    zero_state = zeros(length(state))
    zero_result = ElasticResult(mesh, zero_state, material, [0.0], [0.0], :reflecting, nothing)
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
    tuple_zero_initial = (x, y) -> (0.0, 0.0, 0.0, 0.0, 0.0)

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

    cfl_result = solve_elastodynamics(
        tuple_zero_initial;
        nx=2,
        ny=2,
        tspan=(0.0, 0.03),
        boundary=:reflecting,
    )

    @test cfl_result.times[1] == 0.0
    @test cfl_result.times[end] == 0.03
    @test length(cfl_result.times) > 1
    @test length(cfl_result.energy_history) == length(cfl_result.times)
    @test all(isfinite, cfl_result.energy_history)
    @test norm(cfl_result.state) ≈ 0.0 atol = 1.0e-12
    @test JuliaDG.default_elastic_dt(cfl_result.mesh, cfl_result.material, 0.1) > 0.0
    @test JuliaDG.minimum_edge_length(cfl_result.mesh) > 0.0

    history_result = solve_elastodynamics(
        tuple_zero_initial;
        nx=2,
        ny=2,
        tspan=(0.0, 0.03),
        dt=0.02,
        boundary=:reflecting,
        save_history=true,
    )

    @test history_result.state_history !== nothing
    @test length(history_result.state_history) == length(history_result.times)
    @test history_result.state_history[1] ≈ zeros(length(history_result.state))
    @test history_result.state_history[end] ≈ history_result.state
    @test all(state -> length(state) == length(history_result.state), history_result.state_history)

    default_history_result = solve_elastodynamics(
        tuple_zero_initial;
        nx=2,
        ny=2,
        tspan=(0.0, 0.03),
        dt=0.02,
        boundary=:reflecting,
    )
    @test default_history_result.state_history === nothing

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

    elastic_custom_points = [
        Meshes.Point(0.0, 0.0),
        Meshes.Point(1.0, 0.0),
        Meshes.Point(0.0, 1.0),
    ]
    elastic_custom_connectivities = [Meshes.connect((1, 2, 3), Meshes.Triangle)]
    elastic_custom_mesh = Meshes.SimpleMesh(elastic_custom_points, elastic_custom_connectivities)
    elastic_custom_result = solve_elastodynamics(
        tuple_zero_initial;
        mesh=elastic_custom_mesh,
        tspan=(0.0, 0.01),
        dt=0.01,
        boundary=:reflecting,
    )

    @test elastic_custom_result.mesh === elastic_custom_mesh
    @test length(elastic_custom_result.state) == 15
    @test norm(elastic_custom_result.state) ≈ 0.0 atol = 1.0e-12

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
    ndofs = 3 * JuliaDG.triangle_count(mesh)

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

    raw = Meshes.simplexify(Meshes.CartesianGrid((0.0, 0.0), (1.0, 1.0), dims=(2, 2)))
    A_raw, b_raw = assemble_poisson_sipg(raw, f, g; penalty=20.0)
    raw_ndofs = 3 * JuliaDG.triangle_count(raw)

    @test size(A_raw) == (raw_ndofs, raw_ndofs)
    @test length(b_raw) == raw_ndofs
    @test isapprox(norm(Matrix(A_raw - transpose(A_raw))), 0.0; atol=1.0e-10)

    @test !isdefined(JuliaDG, :assemble_triangle_terms!)
    @test !isdefined(JuliaDG, :assemble_face_terms!)
    @test !isdefined(JuliaDG, :assemble_interior_face!)
    @test !isdefined(JuliaDG, :assemble_boundary_face!)
end

@testset "basis geometry" begin
    mesh = unit_square_mesh(1, 1)
    coords = JuliaDG.triangle_coordinates(mesh, 1)
    points = JuliaDG.triangle_points(mesh, 1)
    area, grads = JuliaDG.triangle_geometry(coords)

    @test area ≈ 0.5
    @test grads[:, 1] + grads[:, 2] + grads[:, 3] ≈ zeros(2)
    @test !applicable(JuliaDG.triangle_coordinates, mesh, points)

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
    @test !applicable(JuliaDG.edge_normal, mesh, points, 1)

    x, y = JuliaDG.edge_point(mesh, 1, 1, 0.25)
    @test x ≈ 0.25
    @test y ≈ 0.0
    @test !applicable(JuliaDG.edge_point, mesh, points, 1, 0.25)
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

    custom_points = [
        Meshes.Point(0.0, 0.0),
        Meshes.Point(1.0, 0.0),
        Meshes.Point(0.0, 1.0),
    ]
    custom_connectivities = [Meshes.connect((1, 2, 3), Meshes.Triangle)]
    custom_mesh = Meshes.SimpleMesh(custom_points, custom_connectivities)
    custom_result = solve_poisson(
        zero_f;
        mesh=custom_mesh,
        g=affine,
        penalty=30.0,
    )

    @test custom_result.mesh === custom_mesh
    @test length(custom_result.coeffs) == 3
    @test l2_error(custom_result, affine) < 1.0e-8
end

@testset "plot data" begin
    mesh = unit_square_mesh(1, 1)
    ndofs = 3 * JuliaDG.triangle_count(mesh)
    coeffs = Float64.(1:ndofs)
    result = DGResult(mesh, coeffs, spzeros(ndofs, ndofs), zeros(ndofs))

    data = dg_plot_data(result)

    @test length(data.xs) == ndofs
    @test length(data.ys) == ndofs
    @test length(data.values) == ndofs
    @test length(data.triangles) == JuliaDG.triangle_count(mesh)
    @test data.triangles == [(1, 2, 3), (4, 5, 6)]
    @test isempty(intersect(collect(data.triangles[1]), collect(data.triangles[2])))
    @test data.values == coeffs
    @test data.xs == [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    @test data.ys == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]

    raw = Meshes.simplexify(Meshes.CartesianGrid((0.0, 0.0), (1.0, 1.0), dims=(1, 1)))
    raw_ndofs = 3 * JuliaDG.triangle_count(raw)
    raw_coeffs = Float64.(1:raw_ndofs)
    raw_result = DGResult(raw, raw_coeffs, spzeros(raw_ndofs, raw_ndofs), zeros(raw_ndofs))
    raw_data = dg_plot_data(raw_result)

    @test raw_data.triangles == [(1, 2, 3), (4, 5, 6)]
    @test raw_data.values == raw_coeffs

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

@testset "elastic plot data" begin
    mesh = unit_square_mesh(1, 1)
    material = ElasticMaterial(1.0, 1.0, 0.5)
    initial = (x, y) -> (vx=x, vy=2 * y, sxx=x + y, syy=x - y, sxy=10 + x - y)
    state = JuliaDG.interpolate_elastic_state(initial, mesh)
    energy = JuliaDG.elastic_energy(mesh, state, material)
    result = ElasticResult(mesh, state, material, [0.0], [energy], :reflecting, nothing)

    data = elastic_plot_data(result)

    @test length(data.xs) == length(state) ÷ 5
    @test length(data.ys) == length(state) ÷ 5
    @test length(data.values) == length(state) ÷ 5
    @test data.triangles == [(1, 2, 3), (4, 5, 6)]
    @test data.values ≈ [sqrt(x^2 + (2 * y)^2) for (x, y) in zip(data.xs, data.ys)]

    sxy_data = elastic_plot_data(result; field=:sxy)
    @test sxy_data.xs == data.xs
    @test sxy_data.ys == data.ys
    @test sxy_data.triangles == data.triangles
    @test sxy_data.values ≈ [10 + x - y for (x, y) in zip(data.xs, data.ys)]

    @test_throws ArgumentError elastic_plot_data(result; field=:pressure)

    state2 = copy(state)
    triangle_total = JuliaDG.triangle_count(mesh)
    for triangle in 1:triangle_total
        for local_index in 1:JuliaDG.ELASTIC_LOCAL_DOF_COUNT
            state2[JuliaDG.elastic_dof(triangle, local_index, 1, triangle_total)] = 3.0
            state2[JuliaDG.elastic_dof(triangle, local_index, 2, triangle_total)] = 4.0
        end
    end
    history_result = ElasticResult(mesh, state2, material, [0.0, 0.1], [energy, energy], :reflecting, [state, state2])
    frame_data = elastic_plot_data(history_result, 2)

    @test frame_data.values ≈ fill(5.0, length(frame_data.values))
    @test_throws ArgumentError elastic_plot_data(result, 1)
    @test_throws ArgumentError elastic_plot_data(history_result, 0)
    @test_throws ArgumentError elastic_plot_data(history_result, 3)

    try
        record_solution(result, "unused.gif")
        @test false
    catch err
        @test err isa ArgumentError
        @test err.msg ==
              "record_solution requires Makie; load CairoMakie or GLMakie before calling it"
    end
end

@testset "example script" begin
    package_root = dirname(@__DIR__)
    example_path = joinpath(package_root, "examples", "poisson2d_unit_square.jl")
    output = read(`$(Base.julia_cmd()) --project=$package_root $example_path`, String)

    @test occursin(r"DOFs:\s+384", output)
    @test occursin(r"L2 error:\s+[0-9]", output)
end

@testset "elastic example script" begin
    package_root = dirname(@__DIR__)
    example_path = joinpath(package_root, "examples", "elastodynamics2d_unit_square.jl")
    output = read(`$(Base.julia_cmd()) --project=$package_root $example_path`, String)

    @test occursin(r"Elastic DOFs:\s+[0-9]+", output)
    @test occursin(r"Final time:\s+0\.02", output)
    @test occursin(r"Final energy:\s+[0-9]", output)
end
