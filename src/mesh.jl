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

function resolve_mesh(mesh, nx::Integer, ny::Integer)
    if mesh === nothing
        return unit_square_mesh(nx, ny)
    elseif mesh isa TriMesh
        return mesh
    elseif mesh isa Meshes.Mesh
        return TriMesh(mesh)
    end

    throw(ArgumentError("mesh must be nothing, TriMesh, or Meshes.Mesh"))
end

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
    cells = orient_counterclockwise_cells(vertices, cells_matrix(mesh))
    return TriMesh(vertices, cells, build_faces(cells), mesh)
end

struct FacetAdjacency
    point_ids::NTuple{2,Int}
    triangles::NTuple{2,Int}
    local_edges::NTuple{2,Int}
end

plain_coordinate(value::Real) = Float64(value)
plain_coordinate(value) = Float64(getproperty(value, :val))

function point_xy(mesh::Meshes.Mesh, point_id::Integer)
    point = Meshes.vertices(mesh)[Int(point_id)]
    coords = Meshes.coords(point)
    return (plain_coordinate(coords.x), plain_coordinate(coords.y))
end

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

function triangle_connectivities(mesh::Meshes.Mesh)
    topology = Meshes.topology(mesh)
    hasproperty(topology, :connec) ||
        throw(ArgumentError("only Meshes meshes with connectivity topology are supported"))

    triangles = NTuple{3,Int}[]
    for connectivity in getproperty(topology, :connec)
        ids = getproperty(connectivity, :indices)
        length(ids) == 3 || throw(ArgumentError("only triangular Meshes meshes are supported"))
        push!(triangles, (Int(ids[1]), Int(ids[2]), Int(ids[3])))
    end

    return triangles
end

function orient_triangle_points(mesh::Meshes.Mesh, points::NTuple{3,Int})
    x1, y1 = point_xy(mesh, points[1])
    x2, y2 = point_xy(mesh, points[2])
    x3, y3 = point_xy(mesh, points[3])
    twice_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

    twice_area > 0 && return points
    twice_area < 0 && return (points[1], points[3], points[2])
    throw(ArgumentError("Meshes triangle elements must have positive area"))
end

function oriented_triangle_connectivities(mesh::Meshes.Mesh)
    return [orient_triangle_points(mesh, points) for points in triangle_connectivities(mesh)]
end

function triangle_count(mesh::Meshes.Mesh)
    return length(oriented_triangle_connectivities(mesh))
end

function triangle_points(mesh::Meshes.Mesh, triangle::Integer)
    triangles = oriented_triangle_connectivities(mesh)
    index = Int(triangle)
    1 <= index <= length(triangles) || throw(ArgumentError("triangle out of range"))
    return triangles[index]
end

function orient_counterclockwise_cells(vertices::AbstractMatrix{<:Real}, cells::Matrix{Int})
    oriented_cells = copy(cells)

    for cell in axes(oriented_cells, 2)
        v1 = oriented_cells[1, cell]
        v2 = oriented_cells[2, cell]
        v3 = oriented_cells[3, cell]
        det_j =
            (vertices[1, v2] - vertices[1, v1]) * (vertices[2, v3] - vertices[2, v1]) -
            (vertices[1, v3] - vertices[1, v1]) * (vertices[2, v2] - vertices[2, v1])

        if det_j < 0
            oriented_cells[2, cell], oriented_cells[3, cell] =
                oriented_cells[3, cell], oriented_cells[2, cell]
        elseif det_j == 0
            throw(ArgumentError("Meshes triangle cells must have positive area"))
        end
    end

    return oriented_cells
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

function facet_adjacencies(mesh::Meshes.Mesh)
    triangles = oriented_triangle_connectivities(mesh)
    connectivities = [Meshes.connect(points, Meshes.Triangle) for points in triangles]
    topology = Meshes.HalfEdgeTopology(connectivities; sort=false)
    edge_triangles = Meshes.Coboundary{1,2}(topology)
    edge4pair = getproperty(topology, :edge4pair)
    points_by_edge = Dict(edge => point_ids for (point_ids, edge) in edge4pair)

    facets = FacetAdjacency[]
    sizehint!(facets, length(edge4pair))

    for edge in sort!(collect(values(edge4pair)))
        point_ids = points_by_edge[edge]
        incident_triangles = Tuple(edge_triangles(edge))

        if length(incident_triangles) == 1
            triangle = incident_triangles[1]
            local_edge = local_edge_for_points(triangles[triangle], point_ids)
            push!(facets, FacetAdjacency(point_ids, (triangle, 0), (local_edge, 0)))
        elseif length(incident_triangles) == 2
            left, right = incident_triangles
            left_edge = local_edge_for_points(triangles[left], point_ids)
            right_edge = local_edge_for_points(triangles[right], point_ids)
            push!(facets, FacetAdjacency(point_ids, (left, right), (left_edge, right_edge)))
        else
            throw(ArgumentError("triangular mesh edge must have one or two incident triangles"))
        end
    end

    return facets
end

function local_edge_for_points(triangle_points::NTuple{3,Int}, edge_points::NTuple{2,Int})
    for local_edge in 1:3
        a, b = LOCAL_EDGES[local_edge]
        ordered_edge(triangle_points[a], triangle_points[b]) == edge_points && return local_edge
    end

    throw(ArgumentError("edge is not part of triangle"))
end
