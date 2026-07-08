function dg_plot_data(result::DGResult)
    cell_points = oriented_triangle_connectivities(result.mesh)
    triangle_total = length(cell_points)
    expected_coeffs = 3 * triangle_total
    length(result.coeffs) == expected_coeffs ||
        throw(ArgumentError("DGResult coefficient vector must contain three values per triangle"))

    xs = Vector{Float64}(undef, expected_coeffs)
    ys = Vector{Float64}(undef, expected_coeffs)
    values = Vector{Float64}(undef, expected_coeffs)
    triangles = Vector{NTuple{3,Int}}(undef, triangle_total)
    coordinates_for(points) = begin
        coords = Matrix{Float64}(undef, 2, 3)
        for local_index in 1:3
            x, y = point_xy(result.mesh, points[local_index])
            coords[1, local_index] = x
            coords[2, local_index] = y
        end
        coords
    end

    point_index = 1
    for (triangle, points) in pairs(cell_points)
        coords = coordinates_for(points)
        first_point = point_index

        for local_index in 1:3
            xs[point_index] = coords[1, local_index]
            ys[point_index] = coords[2, local_index]
            values[point_index] = result.coeffs[global_dof(triangle, local_index)]
            point_index += 1
        end

        triangles[triangle] = (first_point, first_point + 1, first_point + 2)
    end

    return (xs=xs, ys=ys, values=values, triangles=triangles)
end

function elastic_plot_data(result::ElasticResult; field::Symbol=:velocity_magnitude)
    return elastic_plot_data(result.mesh, result.state; field=field)
end

function elastic_plot_data(result::ElasticResult, frame::Integer; field::Symbol=:velocity_magnitude)
    result.state_history === nothing &&
        throw(ArgumentError("ElasticResult does not contain state history; solve with save_history=true"))
    1 <= frame <= length(result.state_history) || throw(ArgumentError("frame out of range"))
    return elastic_plot_data(result.mesh, result.state_history[frame]; field=field)
end

function elastic_plot_data(mesh, state::AbstractVector{<:Real}; field::Symbol=:velocity_magnitude)
    cell_points = oriented_triangle_connectivities(mesh)
    triangle_total = length(cell_points)
    expected_state = ELASTIC_FIELD_COUNT * ELASTIC_LOCAL_DOF_COUNT * triangle_total
    length(state) == expected_state ||
        throw(ArgumentError("elastic state length does not match mesh"))

    npoints = ELASTIC_LOCAL_DOF_COUNT * triangle_total
    xs = Vector{Float64}(undef, npoints)
    ys = Vector{Float64}(undef, npoints)
    values = Vector{Float64}(undef, npoints)
    triangles = Vector{NTuple{3,Int}}(undef, triangle_total)
    field_index = elastic_plot_field_index(field)
    coordinates_for(points) = begin
        coords = Matrix{Float64}(undef, 2, 3)
        for local_index in 1:3
            x, y = point_xy(mesh, points[local_index])
            coords[1, local_index] = x
            coords[2, local_index] = y
        end
        coords
    end

    point_index = 1
    for (triangle, points) in pairs(cell_points)
        coords = coordinates_for(points)
        first_point = point_index

        for local_index in 1:ELASTIC_LOCAL_DOF_COUNT
            xs[point_index] = coords[1, local_index]
            ys[point_index] = coords[2, local_index]
            values[point_index] = elastic_plot_value(state, triangle, local_index, field_index, triangle_total)
            point_index += 1
        end

        triangles[triangle] = (first_point, first_point + 1, first_point + 2)
    end

    return (xs=xs, ys=ys, values=values, triangles=triangles)
end

function elastic_plot_field_index(field::Symbol)
    field === :velocity_magnitude && return 0

    index = findfirst(==(field), ELASTIC_FIELD_NAMES)
    index === nothing &&
        throw(ArgumentError("field must be :velocity_magnitude, :vx, :vy, :sxx, :syy, or :sxy"))
    return index
end

function elastic_plot_value(state, cell::Integer, local_index::Integer, field_index::Integer, ncells::Integer)
    if field_index == 0
        vx = state[elastic_dof(cell, local_index, 1, ncells)]
        vy = state[elastic_dof(cell, local_index, 2, ncells)]
        return sqrt(vx^2 + vy^2)
    end

    return state[elastic_dof(cell, local_index, field_index, ncells)]
end

function plot_solution(args...; kwargs...)
    throw(ArgumentError("plot_solution requires Makie; load CairoMakie or GLMakie before calling it"))
end

function record_solution(args...; kwargs...)
    throw(ArgumentError("record_solution requires Makie; load CairoMakie or GLMakie before calling it"))
end
