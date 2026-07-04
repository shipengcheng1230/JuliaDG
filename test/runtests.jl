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
