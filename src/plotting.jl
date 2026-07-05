function dg_plot_data(result::DGResult)
    ncells = size(result.mesh.cells, 2)
    expected_coeffs = 3 * ncells
    length(result.coeffs) == expected_coeffs ||
        throw(ArgumentError("DGResult coefficient vector must contain three values per cell"))

    xs = Vector{Float64}(undef, expected_coeffs)
    ys = Vector{Float64}(undef, expected_coeffs)
    values = Vector{Float64}(undef, expected_coeffs)
    faces = Vector{NTuple{3,Int}}(undef, ncells)

    point_index = 1
    for cell in 1:ncells
        coords = cell_coordinates(result.mesh, cell)
        first_point = point_index

        for local_index in 1:3
            xs[point_index] = coords[1, local_index]
            ys[point_index] = coords[2, local_index]
            values[point_index] = result.coeffs[global_dof(cell, local_index)]
            point_index += 1
        end

        faces[cell] = (first_point, first_point + 1, first_point + 2)
    end

    return (xs=xs, ys=ys, values=values, faces=faces)
end

function plot_solution(args...; kwargs...)
    throw(ArgumentError("plot_solution requires Makie; load CairoMakie or GLMakie before calling it"))
end
