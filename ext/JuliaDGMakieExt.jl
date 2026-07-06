module JuliaDGMakieExt

using JuliaDG
using Makie

function JuliaDG.plot_solution(result::JuliaDG.DGResult; colormap=:viridis, show_mesh::Bool=true)
    data = JuliaDG.dg_plot_data(result)
    return plot_cell_data(data; colormap=colormap, show_mesh=show_mesh, colorbar_label="u")
end

function JuliaDG.plot_solution(
    result::JuliaDG.ElasticResult;
    field::Symbol=:velocity_magnitude,
    colormap=:viridis,
    show_mesh::Bool=true,
)
    data = JuliaDG.elastic_plot_data(result; field=field)
    return plot_cell_data(data; colormap=colormap, show_mesh=show_mesh, colorbar_label=elastic_colorbar_label(field))
end

function plot_cell_data(data; colormap, show_mesh::Bool, colorbar_label::AbstractString)
    vertices, faces = triangulation(data)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; xlabel="x", ylabel="y", aspect=Makie.DataAspect())
    mesh_plot = Makie.mesh!(
        ax,
        vertices,
        faces;
        color=data.values,
        colormap=colormap,
        shading=Makie.NoShading,
    )

    if show_mesh
        Makie.linesegments!(
            ax,
            edge_segments(data);
            color=(:black, 0.45),
            linewidth=0.75,
        )
    end

    Makie.Colorbar(fig[1, 2], mesh_plot; label=colorbar_label)
    Makie.autolimits!(ax)
    return fig
end

function triangulation(data)
    vertices = hcat(data.xs, data.ys)
    faces = Matrix{Int}(undef, length(data.faces), 3)

    for (row, face) in enumerate(data.faces)
        faces[row, 1] = face[1]
        faces[row, 2] = face[2]
        faces[row, 3] = face[3]
    end

    return vertices, faces
end

elastic_colorbar_label(field::Symbol) = field === :velocity_magnitude ? "|v|" : String(field)

function edge_segments(data)
    points = Makie.Point2f[]

    for (a, b, c) in data.faces
        pa = Makie.Point2f(data.xs[a], data.ys[a])
        pb = Makie.Point2f(data.xs[b], data.ys[b])
        pc = Makie.Point2f(data.xs[c], data.ys[c])

        push!(points, pa, pb, pb, pc, pc, pa)
    end

    return points
end

function JuliaDG.record_solution(
    result::JuliaDG.ElasticResult,
    path::AbstractString;
    field::Symbol=:velocity_magnitude,
    framerate::Real=20,
    colormap=:viridis,
    show_mesh::Bool=true,
)
    result.state_history === nothing &&
        throw(ArgumentError("ElasticResult does not contain state history; solve with save_history=true"))

    first_data = JuliaDG.elastic_plot_data(result, 1; field=field)
    vertices, faces = triangulation(first_data)

    color_values = Makie.Observable(first_data.values)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; xlabel="x", ylabel="y", aspect=Makie.DataAspect())
    mesh_plot = Makie.mesh!(
        ax,
        vertices,
        faces;
        color=color_values,
        colormap=colormap,
        shading=Makie.NoShading,
    )

    if show_mesh
        Makie.linesegments!(
            ax,
            edge_segments(first_data);
            color=(:black, 0.45),
            linewidth=0.75,
        )
    end

    Makie.Colorbar(fig[1, 2], mesh_plot; label=elastic_colorbar_label(field))
    Makie.autolimits!(ax)

    Makie.record(fig, path, eachindex(result.state_history); framerate=framerate) do frame
        color_values[] = JuliaDG.elastic_plot_data(result, frame; field=field).values
    end

    return path
end

end
