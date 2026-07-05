module JuliaDGMakieExt

using JuliaDG
using Makie

function JuliaDG.plot_solution(result::JuliaDG.DGResult; colormap=:viridis, show_mesh::Bool=true)
    data = JuliaDG.dg_plot_data(result)
    vertices = hcat(data.xs, data.ys)
    faces = Matrix{Int}(undef, length(data.faces), 3)

    for (row, face) in enumerate(data.faces)
        faces[row, 1] = face[1]
        faces[row, 2] = face[2]
        faces[row, 3] = face[3]
    end

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

    Makie.Colorbar(fig[1, 2], mesh_plot; label="u")
    Makie.autolimits!(ax)
    return fig
end

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

end
