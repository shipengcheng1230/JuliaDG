module Elastodynamics

import Meshes
using ..JuliaDG:
    EDGE_QUADRATURE,
    LOCAL_EDGES,
    TRIANGLE_QUADRATURE,
    barycentric_coordinates,
    basis_values_at_point,
    facet_adjacencies,
    oriented_triangle_connectivities,
    point_xy,
    resolve_mesh,
    triangle_geometry

export Material, Result, solve, evaluate, energy, plot_data, plot, record

struct Material
    rho::Float64
    lambda::Float64
    mu::Float64

    function Material(rho::Real, lambda::Real, mu::Real)
        rho > 0 || throw(ArgumentError("rho must be positive"))
        lambda >= 0 || throw(ArgumentError("lambda must be nonnegative"))
        mu > 0 || throw(ArgumentError("mu must be positive"))
        return new(Float64(rho), Float64(lambda), Float64(mu))
    end
end

struct Result
    mesh::Meshes.Mesh
    state::Vector{Float64}
    material::Material
    times::Vector{Float64}
    energy_history::Vector{Float64}
    boundary::Symbol
    state_history::Union{Nothing,Vector{Vector{Float64}}}
end

const FIELD_NAMES = (:vx, :vy, :sxx, :syy, :sxy)
const FIELD_COUNT = 5
const LOCAL_DOF_COUNT = 3

function dof(cell::Integer, local_index::Integer, field::Integer, ncells::Integer)
    1 <= cell <= ncells || throw(ArgumentError("cell out of range"))
    1 <= local_index <= LOCAL_DOF_COUNT || throw(ArgumentError("local_index out of range"))
    1 <= field <= FIELD_COUNT || throw(ArgumentError("field out of range"))
    return LOCAL_DOF_COUNT * ncells * (field - 1) +
           LOCAL_DOF_COUNT * (cell - 1) +
           local_index
end

function validate_boundary(boundary::Symbol)
    boundary in (:reflecting, :traction_free) ||
        throw(ArgumentError("boundary must be :reflecting or :traction_free"))
    return boundary
end

function components(value)
    if value isa NamedTuple
        all(field -> hasproperty(value, field), FIELD_NAMES) || throw(
            ArgumentError("initial named tuple must contain vx, vy, sxx, syy, and sxy"),
        )
        return (
            Float64(value.vx),
            Float64(value.vy),
            Float64(value.sxx),
            Float64(value.syy),
            Float64(value.sxy),
        )
    elseif value isa Tuple && length(value) == FIELD_COUNT
        return ntuple(index -> Float64(value[index]), FIELD_COUNT)
    end

    throw(ArgumentError("initial condition must return a named tuple or 5-tuple"))
end

function interpolate_state(initial, mesh)
    triangles = oriented_triangle_connectivities(mesh)
    triangle_total = length(triangles)
    state = zeros(Float64, FIELD_COUNT * LOCAL_DOF_COUNT * triangle_total)
    coordinates_for(points) = begin
        coords = Matrix{Float64}(undef, 2, 3)
        for local_index = 1:3
            x, y = point_xy(mesh, points[local_index])
            coords[1, local_index] = x
            coords[2, local_index] = y
        end
        coords
    end

    for (triangle, points) in pairs(triangles)
        coords = coordinates_for(points)
        for local_index = 1:LOCAL_DOF_COUNT
            values = components(initial(coords[1, local_index], coords[2, local_index]))
            for field = 1:FIELD_COUNT
                state[dof(triangle, local_index, field, triangle_total)] = values[field]
            end
        end
    end

    return state
end

function pressure_wave_speed(material::Material)
    return sqrt((material.lambda + 2 * material.mu) / material.rho)
end

function flux_x(q, material::Material)
    vx, vy, sxx, syy, sxy = q
    return (
        sxx / material.rho,
        sxy / material.rho,
        (material.lambda + 2 * material.mu) * vx,
        material.lambda * vx,
        material.mu * vy,
    )
end

function flux_y(q, material::Material)
    vx, vy, sxx, syy, sxy = q
    return (
        sxy / material.rho,
        syy / material.rho,
        material.lambda * vy,
        (material.lambda + 2 * material.mu) * vy,
        material.mu * vx,
    )
end

function physical_normal_flux(q, normal, material::Material)
    x_flux = flux_x(q, material)
    y_flux = flux_y(q, material)
    return ntuple(
        field -> normal[1] * x_flux[field] + normal[2] * y_flux[field],
        FIELD_COUNT,
    )
end

function normal_flux(left, right, normal, material::Material)
    left_flux = physical_normal_flux(left, normal, material)
    right_flux = physical_normal_flux(right, normal, material)
    alpha = pressure_wave_speed(material)

    return ntuple(
        field ->
            0.5 * (left_flux[field] + right_flux[field]) +
            -0.5 * alpha * (left[field] - right[field]),
        FIELD_COUNT,
    )
end

function state_at_point(state::AbstractVector{<:Real}, cell::Integer, phi, ncells::Integer)
    return ntuple(
        field -> sum(
            state[dof(cell, local_index, field, ncells)] * phi[local_index] for
            local_index = 1:LOCAL_DOF_COUNT
        ),
        FIELD_COUNT,
    )
end

function boundary_state(q, normal, boundary::Symbol)
    if boundary === :reflecting
        return reflecting_state(q, normal)
    elseif boundary === :traction_free
        return traction_free_state(q, normal)
    end

    validate_boundary(boundary)
end

function reflecting_state(q, normal)
    vx, vy, sxx, syy, sxy = q
    nx, ny = normal
    normal_velocity = vx * nx + vy * ny
    reflected_vx = vx - 2 * normal_velocity * nx
    reflected_vy = vy - 2 * normal_velocity * ny
    reflected_sxx, reflected_syy, reflected_sxy = reflected_stress(sxx, syy, sxy, normal)
    return (reflected_vx, reflected_vy, reflected_sxx, reflected_syy, reflected_sxy)
end

function reflected_stress(sxx::Real, syy::Real, sxy::Real, normal)
    nx, ny = normal
    r11 = 1 - 2 * nx^2
    r12 = -2 * nx * ny
    r21 = r12
    r22 = 1 - 2 * ny^2

    reflected_sxx = r11^2 * sxx + 2 * r11 * r12 * sxy + r12^2 * syy
    reflected_syy = r21^2 * sxx + 2 * r21 * r22 * sxy + r22^2 * syy
    reflected_sxy = r11 * r21 * sxx + (r11 * r22 + r12 * r21) * sxy + r12 * r22 * syy
    return (reflected_sxx, reflected_syy, reflected_sxy)
end

function traction_free_state(q, normal)
    vx, vy, sxx, syy, sxy = q
    traction_sxx, traction_syy, traction_sxy = traction_free_stress(sxx, syy, sxy, normal)
    return (vx, vy, traction_sxx, traction_syy, traction_sxy)
end

function traction_free_stress(sxx::Real, syy::Real, sxy::Real, normal)
    nx, ny = normal
    traction_x = sxx * nx + sxy * ny
    traction_y = sxy * nx + syy * ny
    normal_traction = nx * traction_x + ny * traction_y

    ghost_sxx = sxx - 4 * traction_x * nx + 2 * normal_traction * nx^2
    ghost_syy = syy - 4 * traction_y * ny + 2 * normal_traction * ny^2
    ghost_sxy =
        sxy - 2 * (traction_x * ny + nx * traction_y) + 2 * normal_traction * nx * ny
    return (ghost_sxx, ghost_syy, ghost_sxy)
end

function rhs(state::AbstractVector{<:Real}, mesh, material::Material, boundary::Symbol)
    validate_boundary(boundary)
    triangles = oriented_triangle_connectivities(mesh)
    facets = facet_adjacencies(mesh)
    triangle_total = length(triangles)
    expected_length = FIELD_COUNT * LOCAL_DOF_COUNT * triangle_total
    length(state) == expected_length ||
        throw(ArgumentError("elastic state length does not match mesh"))

    coordinates_for(points) = begin
        coords = Matrix{Float64}(undef, 2, 3)
        for local_index = 1:3
            x, y = point_xy(mesh, points[local_index])
            coords[1, local_index] = x
            coords[2, local_index] = y
        end
        coords
    end
    edge_normal_for(points, local_edge) = begin
        a, b = LOCAL_EDGES[local_edge]
        x1, y1 = point_xy(mesh, points[a])
        x2, y2 = point_xy(mesh, points[b])
        dx = x2 - x1
        dy = y2 - y1
        edge_len = hypot(dx, dy)
        ((dy / edge_len, -dx / edge_len), edge_len)
    end
    edge_point_for(points, local_edge, s) = begin
        a, b = LOCAL_EDGES[local_edge]
        x1, y1 = point_xy(mesh, points[a])
        x2, y2 = point_xy(mesh, points[b])
        ((1 - s) * x1 + s * x2, (1 - s) * y1 + s * y2)
    end

    function add_volume_terms_local!(residual, triangle, points)
        coords = coordinates_for(points)
        area, grads = triangle_geometry(coords)

        for (lambdas, weight) in TRIANGLE_QUADRATURE
            q = state_at_point(state, triangle, lambdas, triangle_total)
            x_flux = flux_x(q, material)
            y_flux = flux_y(q, material)

            for local_index = 1:LOCAL_DOF_COUNT
                scale_x = -area * weight * grads[1, local_index]
                scale_y = -area * weight * grads[2, local_index]
                for field = 1:FIELD_COUNT
                    row = dof(triangle, local_index, field, triangle_total)
                    residual[row] += scale_x * x_flux[field] + scale_y * y_flux[field]
                end
            end
        end

        return nothing
    end

    function add_interior_face_local!(residual, facet)
        left_triangle, right_triangle = facet.triangles
        left_points = triangles[left_triangle]
        right_points = triangles[right_triangle]
        left_edge = facet.local_edges[1]
        normal, edge_length = edge_normal_for(left_points, left_edge)
        left_coords = coordinates_for(left_points)
        right_coords = coordinates_for(right_points)

        for (s, weight_1d) in EDGE_QUADRATURE
            x, y = edge_point_for(left_points, left_edge, s)
            left_phi = basis_values_at_point(left_coords, x, y)
            right_phi = basis_values_at_point(right_coords, x, y)
            left_state = state_at_point(state, left_triangle, left_phi, triangle_total)
            right_state = state_at_point(state, right_triangle, right_phi, triangle_total)
            flux = normal_flux(left_state, right_state, normal, material)
            weight = edge_length * weight_1d

            for local_index = 1:LOCAL_DOF_COUNT
                for field = 1:FIELD_COUNT
                    residual[dof(left_triangle, local_index, field, triangle_total)] +=
                        weight * left_phi[local_index] * flux[field]
                    residual[dof(right_triangle, local_index, field, triangle_total)] -=
                        weight * right_phi[local_index] * flux[field]
                end
            end
        end

        return nothing
    end

    function add_boundary_face_local!(residual, facet)
        triangle = facet.triangles[1]
        points = triangles[triangle]
        local_edge = facet.local_edges[1]
        normal, edge_length = edge_normal_for(points, local_edge)
        coords = coordinates_for(points)

        for (s, weight_1d) in EDGE_QUADRATURE
            x, y = edge_point_for(points, local_edge, s)
            phi = basis_values_at_point(coords, x, y)
            interior_state = state_at_point(state, triangle, phi, triangle_total)
            ghost_state = boundary_state(interior_state, normal, boundary)
            flux = normal_flux(interior_state, ghost_state, normal, material)
            weight = edge_length * weight_1d

            for local_index = 1:LOCAL_DOF_COUNT
                for field = 1:FIELD_COUNT
                    residual[dof(triangle, local_index, field, triangle_total)] +=
                        weight * phi[local_index] * flux[field]
                end
            end
        end

        return nothing
    end

    residual = zeros(Float64, expected_length)
    for (triangle, points) in pairs(triangles)
        add_volume_terms_local!(residual, triangle, points)
    end
    for facet in facets
        if facet.triangles[2] == 0
            add_boundary_face_local!(residual, facet)
        else
            add_interior_face_local!(residual, facet)
        end
    end
    return apply_mass_inverse(residual, mesh, triangle_total)
end

function apply_mass_inverse(residual::AbstractVector{<:Real}, mesh, triangle_total::Integer)
    triangles = oriented_triangle_connectivities(mesh)
    rhs = similar(Vector{Float64}(residual))
    coordinates_for(points) = begin
        coords = Matrix{Float64}(undef, 2, 3)
        for local_index = 1:3
            x, y = point_xy(mesh, points[local_index])
            coords[1, local_index] = x
            coords[2, local_index] = y
        end
        coords
    end

    for (triangle, points) in pairs(triangles)
        area, _ = triangle_geometry(coordinates_for(points))
        apply_local_mass_inverse!(rhs, residual, triangle, area, triangle_total)
    end

    return rhs
end

function apply_local_mass_inverse!(
    rhs,
    residual::AbstractVector{<:Real},
    triangle::Integer,
    area::Real,
    triangle_total::Integer,
)
    for field = 1:FIELD_COUNT
        dof1 = dof(triangle, 1, field, triangle_total)
        dof2 = dof(triangle, 2, field, triangle_total)
        dof3 = dof(triangle, 3, field, triangle_total)
        r1 = residual[dof1]
        r2 = residual[dof2]
        r3 = residual[dof3]

        rhs[dof1] = (9 * r1 - 3 * r2 - 3 * r3) / area
        rhs[dof2] = (-3 * r1 + 9 * r2 - 3 * r3) / area
        rhs[dof3] = (-3 * r1 - 3 * r2 + 9 * r3) / area
    end

    return nothing
end

function named_state(values)
    return (
        vx = Float64(values[1]),
        vy = Float64(values[2]),
        sxx = Float64(values[3]),
        syy = Float64(values[4]),
        sxy = Float64(values[5]),
    )
end

function evaluate(result::Result, x::Real, y::Real)
    triangles = oriented_triangle_connectivities(result.mesh)
    triangle_total = length(triangles)
    coordinates_for(points) = begin
        coords = Matrix{Float64}(undef, 2, 3)
        for local_index = 1:3
            px, py = point_xy(result.mesh, points[local_index])
            coords[1, local_index] = px
            coords[2, local_index] = py
        end
        coords
    end

    for (triangle, points) in pairs(triangles)
        coords = coordinates_for(points)
        lambdas = barycentric_coordinates(coords, x, y)

        if all(lambda -> lambda >= -1.0e-10 && lambda <= 1.0 + 1.0e-10, lambdas)
            return named_state(
                state_at_point(result.state, triangle, lambdas, triangle_total),
            )
        end
    end

    throw(ArgumentError("point is outside the mesh"))
end

function energy(result::Result)
    return energy(result.mesh, result.state, result.material)
end

function energy(mesh, state::AbstractVector{<:Real}, material::Material)
    triangles = oriented_triangle_connectivities(mesh)
    triangle_total = length(triangles)
    expected_length = FIELD_COUNT * LOCAL_DOF_COUNT * triangle_total
    length(state) == expected_length ||
        throw(ArgumentError("elastic state length does not match mesh"))

    coordinates_for(points) = begin
        coords = Matrix{Float64}(undef, 2, 3)
        for local_index = 1:3
            x, y = point_xy(mesh, points[local_index])
            coords[1, local_index] = x
            coords[2, local_index] = y
        end
        coords
    end
    triangle_energy_local(triangle, points) = begin
        coords = coordinates_for(points)
        area, _ = triangle_geometry(coords)
        total = 0.0

        for (lambdas, weight) in TRIANGLE_QUADRATURE
            q = state_at_point(state, triangle, lambdas, triangle_total)
            total += area * weight * energy_density(q, material)
        end

        total
    end

    total = 0.0
    for (triangle, points) in pairs(triangles)
        total += triangle_energy_local(triangle, points)
    end

    return total
end

function energy_density(q, material::Material)
    vx, vy, sxx, syy, sxy = q
    lambda = material.lambda
    mu = material.mu

    exx = ((lambda + 2 * mu) * sxx - lambda * syy) / (4 * mu * (lambda + mu))
    eyy = ((lambda + 2 * mu) * syy - lambda * sxx) / (4 * mu * (lambda + mu))
    exy = sxy / (2 * mu)

    kinetic = 0.5 * material.rho * (vx^2 + vy^2)
    strain = 0.5 * (sxx * exx + syy * eyy + 2 * sxy * exy)
    return kinetic + strain
end

function minimum_edge_length(mesh)
    triangles = oriented_triangle_connectivities(mesh)
    facets = facet_adjacencies(mesh)
    hmin = Inf
    edge_normal_for(points, local_edge) = begin
        a, b = LOCAL_EDGES[local_edge]
        x1, y1 = point_xy(mesh, points[a])
        x2, y2 = point_xy(mesh, points[b])
        dx = x2 - x1
        dy = y2 - y1
        edge_len = hypot(dx, dy)
        ((dy / edge_len, -dx / edge_len), edge_len)
    end

    for facet in facets
        triangle = facet.triangles[1]
        local_edge = facet.local_edges[1]
        _, edge_length = edge_normal_for(triangles[triangle], local_edge)
        hmin = min(hmin, edge_length)
    end

    return hmin
end

function default_dt(mesh, material::Material, cfl::Real)
    cfl > 0 || throw(ArgumentError("cfl must be positive"))
    return Float64(cfl) * minimum_edge_length(mesh) / pressure_wave_speed(material)
end

function ssprk3_step(
    state::Vector{Float64},
    dt::Float64,
    mesh,
    material::Material,
    boundary::Symbol,
)
    rhs0 = rhs(state, mesh, material, boundary)
    u1 = state .+ dt .* rhs0

    rhs1 = rhs(u1, mesh, material, boundary)
    u2 = 0.75 .* state .+ 0.25 .* (u1 .+ dt .* rhs1)

    rhs2 = rhs(u2, mesh, material, boundary)
    return (1 / 3) .* state .+ (2 / 3) .* (u2 .+ dt .* rhs2)
end

function solve(
    initial;
    nx::Integer = 20,
    ny::Integer = 20,
    mesh = nothing,
    material::Material = Material(1.0, 1.0, 0.5),
    tspan = (0.0, 0.1),
    dt = nothing,
    cfl::Real = 0.1,
    boundary::Symbol = :reflecting,
    save_history::Bool = false,
)
    boundary = validate_boundary(boundary)
    t0 = Float64(tspan[1])
    tend = Float64(tspan[2])
    tend >= t0 || throw(ArgumentError("tspan end must be greater than or equal to start"))

    dg_mesh = resolve_mesh(mesh, nx, ny)
    triangles = oriented_triangle_connectivities(dg_mesh)
    facets = facet_adjacencies(dg_mesh)
    triangle_total = length(triangles)
    expected_length = FIELD_COUNT * LOCAL_DOF_COUNT * triangle_total
    coordinates_for(points) = begin
        coords = Matrix{Float64}(undef, 2, 3)
        for local_index = 1:3
            x, y = point_xy(dg_mesh, points[local_index])
            coords[1, local_index] = x
            coords[2, local_index] = y
        end
        coords
    end
    edge_normal_for(points, local_edge) = begin
        a, b = LOCAL_EDGES[local_edge]
        x1, y1 = point_xy(dg_mesh, points[a])
        x2, y2 = point_xy(dg_mesh, points[b])
        dx = x2 - x1
        dy = y2 - y1
        edge_len = hypot(dx, dy)
        ((dy / edge_len, -dx / edge_len), edge_len)
    end
    edge_point_for(points, local_edge, s) = begin
        a, b = LOCAL_EDGES[local_edge]
        x1, y1 = point_xy(dg_mesh, points[a])
        x2, y2 = point_xy(dg_mesh, points[b])
        ((1 - s) * x1 + s * x2, (1 - s) * y1 + s * y2)
    end

    function add_volume_terms_cached!(residual, current_state, triangle, points)
        coords = coordinates_for(points)
        area, grads = triangle_geometry(coords)

        for (lambdas, weight) in TRIANGLE_QUADRATURE
            q = state_at_point(current_state, triangle, lambdas, triangle_total)
            x_flux = flux_x(q, material)
            y_flux = flux_y(q, material)

            for local_index = 1:LOCAL_DOF_COUNT
                scale_x = -area * weight * grads[1, local_index]
                scale_y = -area * weight * grads[2, local_index]
                for field = 1:FIELD_COUNT
                    row = dof(triangle, local_index, field, triangle_total)
                    residual[row] += scale_x * x_flux[field] + scale_y * y_flux[field]
                end
            end
        end

        return nothing
    end

    function add_interior_face_cached!(residual, current_state, facet)
        left_triangle, right_triangle = facet.triangles
        left_points = triangles[left_triangle]
        right_points = triangles[right_triangle]
        left_edge = facet.local_edges[1]
        normal, edge_length = edge_normal_for(left_points, left_edge)
        left_coords = coordinates_for(left_points)
        right_coords = coordinates_for(right_points)

        for (s, weight_1d) in EDGE_QUADRATURE
            x, y = edge_point_for(left_points, left_edge, s)
            left_phi = basis_values_at_point(left_coords, x, y)
            right_phi = basis_values_at_point(right_coords, x, y)
            left_state =
                state_at_point(current_state, left_triangle, left_phi, triangle_total)
            right_state =
                state_at_point(current_state, right_triangle, right_phi, triangle_total)
            flux = normal_flux(left_state, right_state, normal, material)
            weight = edge_length * weight_1d

            for local_index = 1:LOCAL_DOF_COUNT
                for field = 1:FIELD_COUNT
                    residual[dof(left_triangle, local_index, field, triangle_total)] +=
                        weight * left_phi[local_index] * flux[field]
                    residual[dof(right_triangle, local_index, field, triangle_total)] -=
                        weight * right_phi[local_index] * flux[field]
                end
            end
        end

        return nothing
    end

    function add_boundary_face_cached!(residual, current_state, facet)
        triangle = facet.triangles[1]
        points = triangles[triangle]
        local_edge = facet.local_edges[1]
        normal, edge_length = edge_normal_for(points, local_edge)
        coords = coordinates_for(points)

        for (s, weight_1d) in EDGE_QUADRATURE
            x, y = edge_point_for(points, local_edge, s)
            phi = basis_values_at_point(coords, x, y)
            interior_state = state_at_point(current_state, triangle, phi, triangle_total)
            ghost_state = boundary_state(interior_state, normal, boundary)
            flux = normal_flux(interior_state, ghost_state, normal, material)
            weight = edge_length * weight_1d

            for local_index = 1:LOCAL_DOF_COUNT
                for field = 1:FIELD_COUNT
                    residual[dof(triangle, local_index, field, triangle_total)] +=
                        weight * phi[local_index] * flux[field]
                end
            end
        end

        return nothing
    end

    triangle_energy_cached(current_state, triangle, points) = begin
        coords = coordinates_for(points)
        area, _ = triangle_geometry(coords)
        total = 0.0

        for (lambdas, weight) in TRIANGLE_QUADRATURE
            q = state_at_point(current_state, triangle, lambdas, triangle_total)
            total += area * weight * energy_density(q, material)
        end

        total
    end

    function interpolate_cached(initial_condition)
        state = zeros(Float64, expected_length)
        for (triangle, points) in pairs(triangles)
            coords = coordinates_for(points)
            for local_index = 1:LOCAL_DOF_COUNT
                values = components(
                    initial_condition(coords[1, local_index], coords[2, local_index]),
                )
                for field = 1:FIELD_COUNT
                    state[dof(triangle, local_index, field, triangle_total)] = values[field]
                end
            end
        end
        return state
    end

    function mass_inverse_cached(residual)
        rhs = similar(Vector{Float64}(residual))
        for (triangle, points) in pairs(triangles)
            area, _ = triangle_geometry(coordinates_for(points))
            apply_local_mass_inverse!(rhs, residual, triangle, area, triangle_total)
        end
        return rhs
    end

    function energy_cached(current_state)
        length(current_state) == expected_length ||
            throw(ArgumentError("elastic state length does not match mesh"))

        total = 0.0
        for (triangle, points) in pairs(triangles)
            total += triangle_energy_cached(current_state, triangle, points)
        end
        return total
    end

    function minimum_edge_length_cached()
        hmin = Inf
        for facet in facets
            triangle = facet.triangles[1]
            _, edge_length = edge_normal_for(triangles[triangle], facet.local_edges[1])
            hmin = min(hmin, edge_length)
        end
        return hmin
    end

    function default_dt_cached(cfl_value::Real)
        cfl_value > 0 || throw(ArgumentError("cfl must be positive"))
        return Float64(cfl_value) * minimum_edge_length_cached() /
               pressure_wave_speed(material)
    end

    function rhs_cached(current_state)
        length(current_state) == expected_length ||
            throw(ArgumentError("elastic state length does not match mesh"))

        residual = zeros(Float64, expected_length)
        for (triangle, points) in pairs(triangles)
            add_volume_terms_cached!(residual, current_state, triangle, points)
        end
        for facet in facets
            if facet.triangles[2] == 0
                add_boundary_face_cached!(residual, current_state, facet)
            else
                add_interior_face_cached!(residual, current_state, facet)
            end
        end
        return mass_inverse_cached(residual)
    end

    function rk_step_cached(current_state, step::Float64)
        rhs0 = rhs_cached(current_state)
        u1 = current_state .+ step .* rhs0

        rhs1 = rhs_cached(u1)
        u2 = 0.75 .* current_state .+ 0.25 .* (u1 .+ step .* rhs1)

        rhs2 = rhs_cached(u2)
        return (1 / 3) .* current_state .+ (2 / 3) .* (u2 .+ step .* rhs2)
    end

    state = interpolate_cached(initial)
    step_dt = dt === nothing ? default_dt_cached(cfl) : Float64(dt)
    step_dt > 0 || throw(ArgumentError("dt must be positive"))

    times = [t0]
    energy_history = [energy_cached(state)]
    state_history = save_history ? [copy(state)] : nothing
    time = t0

    while time < tend
        step = min(step_dt, tend - time)
        state = rk_step_cached(state, Float64(step))
        time += step

        if tend - time <= 10 * eps(max(abs(tend), 1.0))
            time = tend
        end

        push!(times, time)
        push!(energy_history, energy_cached(state))
        if save_history
            push!(state_history, copy(state))
        end
    end

    return Result(dg_mesh, state, material, times, energy_history, boundary, state_history)
end
function plot_data(result::Result; field::Symbol = :velocity_magnitude)
    return plot_data(result.mesh, result.state; field = field)
end

function plot_data(result::Result, frame::Integer; field::Symbol = :velocity_magnitude)
    result.state_history === nothing && throw(
        ArgumentError(
            "Elastodynamics.Result does not contain state history; solve with save_history=true",
        ),
    )
    1 <= frame <= length(result.state_history) || throw(ArgumentError("frame out of range"))
    return plot_data(result.mesh, result.state_history[frame]; field = field)
end

function plot_data(mesh, state::AbstractVector{<:Real}; field::Symbol = :velocity_magnitude)
    cell_points = oriented_triangle_connectivities(mesh)
    triangle_total = length(cell_points)
    expected_state = FIELD_COUNT * LOCAL_DOF_COUNT * triangle_total
    length(state) == expected_state ||
        throw(ArgumentError("elastic state length does not match mesh"))

    npoints = LOCAL_DOF_COUNT * triangle_total
    xs = Vector{Float64}(undef, npoints)
    ys = Vector{Float64}(undef, npoints)
    values = Vector{Float64}(undef, npoints)
    triangles = Vector{NTuple{3,Int}}(undef, triangle_total)
    field_index = plot_field_index(field)
    coordinates_for(points) = begin
        coords = Matrix{Float64}(undef, 2, 3)
        for local_index = 1:3
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

        for local_index = 1:LOCAL_DOF_COUNT
            xs[point_index] = coords[1, local_index]
            ys[point_index] = coords[2, local_index]
            values[point_index] =
                plot_value(state, triangle, local_index, field_index, triangle_total)
            point_index += 1
        end

        triangles[triangle] = (first_point, first_point + 1, first_point + 2)
    end

    return (xs = xs, ys = ys, values = values, triangles = triangles)
end

function plot_field_index(field::Symbol)
    field === :velocity_magnitude && return 0

    index = findfirst(==(field), FIELD_NAMES)
    index === nothing && throw(
        ArgumentError("field must be :velocity_magnitude, :vx, :vy, :sxx, :syy, or :sxy"),
    )
    return index
end

function plot_value(
    state,
    cell::Integer,
    local_index::Integer,
    field_index::Integer,
    ncells::Integer,
)
    if field_index == 0
        vx = state[dof(cell, local_index, 1, ncells)]
        vy = state[dof(cell, local_index, 2, ncells)]
        return sqrt(vx^2 + vy^2)
    end

    return state[dof(cell, local_index, field_index, ncells)]
end

function plot(args...; kwargs...)
    throw(
        ArgumentError("plot requires Makie; load CairoMakie or GLMakie before calling it"),
    )
end

function record(args...; kwargs...)
    throw(
        ArgumentError(
            "record requires Makie; load CairoMakie or GLMakie before calling it",
        ),
    )
end

end
