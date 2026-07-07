struct TriFace
    vertices::NTuple{2,Int}
    cells::NTuple{2,Int}
    local_edges::NTuple{2,Int}
end

struct TriMesh{T<:Real}
    vertices::Matrix{T}
    cells::Matrix{Int}
    faces::Vector{TriFace}
    backend
end

function TriMesh(
    vertices::AbstractMatrix{T},
    cells::AbstractMatrix{<:Integer},
    faces::Vector{TriFace},
) where {T<:Real}
    vertices_matrix = Matrix{T}(vertices)
    cells_matrix = Matrix{Int}(cells)
    return TriMesh(vertices_matrix, cells_matrix, faces, meshes_simple_mesh(vertices_matrix, cells_matrix))
end

function TriMesh(
    vertices::AbstractMatrix{T},
    cells::AbstractMatrix{<:Integer},
    faces::Vector{TriFace},
    backend,
) where {T<:Real}
    return TriMesh{T}(Matrix{T}(vertices), Matrix{Int}(cells), faces, backend)
end

const LOCAL_EDGES = ((1, 2), (2, 3), (3, 1))

mesh_backend(mesh::TriMesh) = mesh.backend

function meshes_points(vertices::AbstractMatrix{<:Real})
    return [Meshes.Point(vertices[1, vertex], vertices[2, vertex]) for vertex in axes(vertices, 2)]
end

function meshes_connectivities(cells::AbstractMatrix{<:Integer})
    return [Meshes.connect(Tuple(Int.(cells[:, cell])), Meshes.Triangle) for cell in axes(cells, 2)]
end

function meshes_simple_mesh(vertices::AbstractMatrix{<:Real}, cells::AbstractMatrix{<:Integer})
    return Meshes.SimpleMesh(meshes_points(vertices), meshes_connectivities(cells))
end

function TriMesh(mesh::Meshes.Mesh)
    vertices = vertices_matrix(mesh)
    cells = cells_matrix(mesh)
    return TriMesh(vertices, cells, build_faces(cells), mesh)
end

plain_coordinate(value::Real) = Float64(value)
plain_coordinate(value) = Float64(getproperty(value, :val))

function vertices_matrix(mesh::Meshes.Mesh)
    points = Meshes.vertices(mesh)
    vertices = Matrix{Float64}(undef, 2, length(points))

    for (vertex, point) in enumerate(points)
        point_coords = Meshes.coords(point)
        vertices[1, vertex] = plain_coordinate(point_coords.x)
        vertices[2, vertex] = plain_coordinate(point_coords.y)
    end

    return vertices
end

function cells_matrix(mesh::Meshes.Mesh)
    mesh_topology = Meshes.topology(mesh)
    hasproperty(mesh_topology, :connec) ||
        throw(ArgumentError("only Meshes meshes with connectivity topology are supported"))

    connectivities = collect(getproperty(mesh_topology, :connec))
    cells = Matrix{Int}(undef, 3, length(connectivities))

    for (cell, connectivity) in enumerate(connectivities)
        indices = getproperty(connectivity, :indices)
        length(indices) == 3 || throw(ArgumentError("only triangular Meshes meshes are supported"))
        cells[:, cell] .= indices
    end

    return cells
end

function unit_square_mesh(nx::Integer, ny::Integer)
    nx > 0 || throw(ArgumentError("nx must be positive"))
    ny > 0 || throw(ArgumentError("ny must be positive"))

    grid = Meshes.CartesianGrid((0.0, 0.0), (1.0, 1.0), dims=(nx, ny))
    return TriMesh(Meshes.simplexify(grid))
end

ordered_edge(a::Int, b::Int) = a < b ? (a, b) : (b, a)

function build_faces(cells::Matrix{Int})
    connectivities = meshes_connectivities(cells)
    topology = Meshes.HalfEdgeTopology(connectivities; sort=false)
    edge_cells = Meshes.Coboundary{1,2}(topology)
    edge4pair = getproperty(topology, :edge4pair)
    keys_by_edge = Dict(edge => key for (key, edge) in edge4pair)

    faces = TriFace[]
    sizehint!(faces, length(edge4pair))

    for edge in sort!(collect(values(edge4pair)))
        vertices = keys_by_edge[edge]
        incident_cells = Tuple(edge_cells(edge))

        if length(incident_cells) == 1
            cell = incident_cells[1]
            local_edge = local_edge_for_vertices(Tuple(cells[:, cell]), vertices)
            push!(faces, TriFace(vertices, (cell, 0), (local_edge, 0)))
        elseif length(incident_cells) == 2
            left_cell, right_cell = incident_cells
            left_edge = local_edge_for_vertices(Tuple(cells[:, left_cell]), vertices)
            right_edge = local_edge_for_vertices(Tuple(cells[:, right_cell]), vertices)
            push!(faces, TriFace(vertices, (left_cell, right_cell), (left_edge, right_edge)))
        else
            throw(ArgumentError("triangular mesh edge must have one or two incident cells"))
        end
    end

    return faces
end

function local_edge_for_vertices(cell_vertices::NTuple{3,Int}, edge_vertices::NTuple{2,Int})
    for local_edge in 1:3
        a, b = LOCAL_EDGES[local_edge]
        ordered_edge(cell_vertices[a], cell_vertices[b]) == edge_vertices && return local_edge
    end

    throw(ArgumentError("edge is not part of cell"))
end
