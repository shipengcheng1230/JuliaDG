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
