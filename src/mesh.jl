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
    open_faces = Dict{NTuple{2,Int},NTuple{2,Int}}()
    faces = TriFace[]

    for cell in 1:size(cells, 2)
        for local_edge in 1:3
            a, b = LOCAL_EDGES[local_edge]
            key = ordered_edge(cells[a, cell], cells[b, cell])
            if haskey(open_faces, key)
                other_cell, other_edge = open_faces[key]
                push!(faces, TriFace(key, (other_cell, cell), (other_edge, local_edge)))
                delete!(open_faces, key)
            else
                open_faces[key] = (cell, local_edge)
            end
        end
    end

    for (key, owner) in open_faces
        cell, local_edge = owner
        push!(faces, TriFace(key, (cell, 0), (local_edge, 0)))
    end

    return faces
end
