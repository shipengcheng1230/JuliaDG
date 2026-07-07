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

function unit_square_mesh(nx::Integer, ny::Integer)
    nx > 0 || throw(ArgumentError("nx must be positive"))
    ny > 0 || throw(ArgumentError("ny must be positive"))

    nvertices = (nx + 1) * (ny + 1)
    vertices = Matrix{Float64}(undef, 2, nvertices)

    vertex_id(i, j) = i + 1 + (nx + 1) * j

    for j in 0:ny
        y = j / ny
        for i in 0:nx
            x = i / nx
            id = vertex_id(i, j)
            vertices[1, id] = x
            vertices[2, id] = y
        end
    end

    cells = Matrix{Int}(undef, 3, 2 * nx * ny)
    cell = 1
    for j in 0:(ny - 1)
        for i in 0:(nx - 1)
            v00 = vertex_id(i, j)
            v10 = vertex_id(i + 1, j)
            v01 = vertex_id(i, j + 1)
            v11 = vertex_id(i + 1, j + 1)

            cells[:, cell] .= (v00, v10, v11)
            cell += 1
            cells[:, cell] .= (v00, v11, v01)
            cell += 1
        end
    end

    return TriMesh(vertices, cells, build_faces(cells))
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
