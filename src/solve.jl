struct DGResult
    mesh::TriMesh
    coeffs::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
end

function solve_poisson(f; nx::Integer=8, ny::Integer=8, g=(x, y) -> 0.0, penalty::Real=20.0)
    mesh = unit_square_mesh(nx, ny)
    A, b = assemble_poisson_sipg(mesh, f, g; penalty=penalty)
    coeffs = A \ b
    return DGResult(mesh, Vector{Float64}(coeffs), A, b)
end

function evaluate_solution(result::DGResult, x::Real, y::Real)
    for cell in 1:size(result.mesh.cells, 2)
        coords = cell_coordinates(result.mesh, cell)
        lambdas = barycentric_coordinates(coords, x, y)

        if all(lambda -> lambda >= -1.0e-10 && lambda <= 1.0 + 1.0e-10, lambdas)
            value = 0.0
            for local_index in 1:3
                value += result.coeffs[global_dof(cell, local_index)] * lambdas[local_index]
            end
            return value
        end
    end

    throw(ArgumentError("point is outside the mesh"))
end

function l2_error(result::DGResult, exact)
    error_squared = 0.0

    for cell in 1:size(result.mesh.cells, 2)
        coords = cell_coordinates(result.mesh, cell)
        area, _ = triangle_geometry(coords)

        for (lambdas, weight) in TRIANGLE_QUADRATURE
            x, y = physical_point(coords, lambdas)
            numerical = sum(
                result.coeffs[global_dof(cell, local_index)] * lambdas[local_index] for
                local_index in 1:3
            )
            diff = numerical - exact(x, y)
            error_squared += area * weight * diff^2
        end
    end

    return sqrt(error_squared)
end
