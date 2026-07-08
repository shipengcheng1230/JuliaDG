struct FacetAdjacency
    point_ids::NTuple{2,Int}
    triangles::NTuple{2,Int}
    local_edges::NTuple{2,Int}
end

const LOCAL_EDGES = ((1, 2), (2, 3), (3, 1))

function resolve_mesh(mesh, nx::Integer, ny::Integer)
    if mesh === nothing
        return unit_square_mesh(nx, ny)
    elseif mesh isa Meshes.Mesh
        return mesh
    end

    throw(ArgumentError("mesh must be nothing or Meshes.Mesh"))
end

function unit_square_mesh(nx::Integer, ny::Integer)
    nx > 0 || throw(ArgumentError("nx must be positive"))
    ny > 0 || throw(ArgumentError("ny must be positive"))

    grid = Meshes.CartesianGrid((0.0, 0.0), (1.0, 1.0), dims=(nx, ny))
    return Meshes.simplexify(grid)
end

plain_coordinate(value::Real) = Float64(value)
plain_coordinate(value) = Float64(getproperty(value, :val))

function point_xy(mesh::Meshes.Mesh, point_id::Integer)
    point = Meshes.vertices(mesh)[Int(point_id)]
    coords = Meshes.coords(point)
    return (plain_coordinate(coords.x), plain_coordinate(coords.y))
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

function oriented_triangle_connectivities(mesh::Meshes.Mesh)
    return [orient_triangle_points(mesh, points) for points in triangle_connectivities(mesh)]
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

function triangle_count(mesh::Meshes.Mesh)
    return length(oriented_triangle_connectivities(mesh))
end

function triangle_points(mesh::Meshes.Mesh, triangle::Integer)
    triangles = oriented_triangle_connectivities(mesh)
    index = Int(triangle)
    1 <= index <= length(triangles) || throw(ArgumentError("triangle out of range"))
    return triangles[index]
end

function facet_adjacencies(mesh::Meshes.Mesh)
    return facet_adjacencies(oriented_triangle_connectivities(mesh))
end

function facet_adjacencies(triangles::Vector{NTuple{3,Int}})
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

ordered_edge(a::Int, b::Int) = a < b ? (a, b) : (b, a)

function local_edge_for_points(triangle_points::NTuple{3,Int}, edge_points::NTuple{2,Int})
    for local_edge in 1:3
        a, b = LOCAL_EDGES[local_edge]
        ordered_edge(triangle_points[a], triangle_points[b]) == edge_points && return local_edge
    end

    throw(ArgumentError("edge is not part of triangle"))
end
