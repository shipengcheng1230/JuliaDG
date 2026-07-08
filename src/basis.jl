const TRIANGLE_QUADRATURE = (
    ((2 / 3, 1 / 6, 1 / 6), 1 / 3),
    ((1 / 6, 2 / 3, 1 / 6), 1 / 3),
    ((1 / 6, 1 / 6, 2 / 3), 1 / 3),
)

const EDGE_QUADRATURE = (
    (0.5 - 0.5 / sqrt(3.0), 0.5),
    (0.5 + 0.5 / sqrt(3.0), 0.5),
)

function cell_coordinates(mesh::TriMesh, cell::Integer)
    coords = Matrix{Float64}(undef, 2, 3)
    for local_index in 1:3
        vertex = mesh.cells[local_index, cell]
        coords[1, local_index] = mesh.vertices[1, vertex]
        coords[2, local_index] = mesh.vertices[2, vertex]
    end
    return coords
end

function triangle_geometry(coords::AbstractMatrix{<:Real})
    x1, y1 = coords[1, 1], coords[2, 1]
    x2, y2 = coords[1, 2], coords[2, 2]
    x3, y3 = coords[1, 3], coords[2, 3]

    det_j = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    det_j > 0 || throw(ArgumentError("triangle vertices must be counterclockwise"))

    grads = Matrix{Float64}(undef, 2, 3)
    grads[:, 1] .= ((y2 - y3) / det_j, (x3 - x2) / det_j)
    grads[:, 2] .= ((y3 - y1) / det_j, (x1 - x3) / det_j)
    grads[:, 3] .= ((y1 - y2) / det_j, (x2 - x1) / det_j)

    return 0.5 * det_j, grads
end

function triangle_coordinates(mesh, triangle::Integer)
    backend = mesh isa TriMesh ? mesh_backend(mesh) : mesh
    coords = Matrix{Float64}(undef, 2, 3)
    points = triangle_points(backend, triangle)

    for local_index in 1:3
        x, y = point_xy(backend, points[local_index])
        coords[1, local_index] = x
        coords[2, local_index] = y
    end

    return coords
end

triangle_geometry(mesh, triangle::Integer) = triangle_geometry(triangle_coordinates(mesh, triangle))

function barycentric_coordinates(coords::AbstractMatrix{<:Real}, x::Real, y::Real)
    x1, y1 = coords[1, 1], coords[2, 1]
    x2, y2 = coords[1, 2], coords[2, 2]
    x3, y3 = coords[1, 3], coords[2, 3]

    det_t = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det_t
    l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det_t
    l3 = 1.0 - l1 - l2

    return (Float64(l1), Float64(l2), Float64(l3))
end

basis_values_at_point(coords::AbstractMatrix{<:Real}, x::Real, y::Real) =
    collect(barycentric_coordinates(coords, x, y))

function physical_point(coords::AbstractMatrix{<:Real}, lambdas)
    x = sum(lambdas[i] * coords[1, i] for i in 1:3)
    y = sum(lambdas[i] * coords[2, i] for i in 1:3)
    return (Float64(x), Float64(y))
end

function point_in_triangle(coords::AbstractMatrix{<:Real}, x::Real, y::Real; tol::Real=1.0e-10)
    lambdas = barycentric_coordinates(coords, x, y)
    return all(lambda -> lambda >= -tol && lambda <= 1 + tol, lambdas)
end

function edge_normal(mesh, triangle::Integer, local_edge::Integer)
    backend = mesh isa TriMesh ? mesh_backend(mesh) : mesh
    a, b = LOCAL_EDGES[local_edge]
    points = triangle_points(backend, triangle)
    x1, y1 = point_xy(backend, points[a])
    x2, y2 = point_xy(backend, points[b])

    dx = x2 - x1
    dy = y2 - y1
    edge_len = hypot(dx, dy)

    return ((dy / edge_len, -dx / edge_len), edge_len)
end

function edge_point(mesh, triangle::Integer, local_edge::Integer, s::Real)
    backend = mesh isa TriMesh ? mesh_backend(mesh) : mesh
    a, b = LOCAL_EDGES[local_edge]
    points = triangle_points(backend, triangle)
    x1, y1 = point_xy(backend, points[a])
    x2, y2 = point_xy(backend, points[b])

    return ((1 - s) * x1 + s * x2, (1 - s) * y1 + s * y2)
end
