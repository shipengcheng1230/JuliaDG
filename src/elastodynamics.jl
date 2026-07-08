struct ElasticMaterial
    rho::Float64
    lambda::Float64
    mu::Float64

    function ElasticMaterial(rho::Real, lambda::Real, mu::Real)
        rho > 0 || throw(ArgumentError("rho must be positive"))
        lambda >= 0 || throw(ArgumentError("lambda must be nonnegative"))
        mu > 0 || throw(ArgumentError("mu must be positive"))
        return new(Float64(rho), Float64(lambda), Float64(mu))
    end
end

struct ElasticResult
    mesh::Meshes.Mesh
    state::Vector{Float64}
    material::ElasticMaterial
    times::Vector{Float64}
    energy_history::Vector{Float64}
    boundary::Symbol
    state_history::Union{Nothing,Vector{Vector{Float64}}}
end

const ELASTIC_FIELD_NAMES = (:vx, :vy, :sxx, :syy, :sxy)
const ELASTIC_FIELD_COUNT = 5
const ELASTIC_LOCAL_DOF_COUNT = 3

function elastic_dof(cell::Integer, local_index::Integer, field::Integer, ncells::Integer)
    1 <= cell <= ncells || throw(ArgumentError("cell out of range"))
    1 <= local_index <= ELASTIC_LOCAL_DOF_COUNT || throw(ArgumentError("local_index out of range"))
    1 <= field <= ELASTIC_FIELD_COUNT || throw(ArgumentError("field out of range"))
    return ELASTIC_LOCAL_DOF_COUNT * ncells * (field - 1) +
           ELASTIC_LOCAL_DOF_COUNT * (cell - 1) +
           local_index
end

function validate_elastic_boundary(boundary::Symbol)
    boundary in (:reflecting, :traction_free) ||
        throw(ArgumentError("boundary must be :reflecting or :traction_free"))
    return boundary
end

function elastic_components(value)
    if value isa NamedTuple
        all(field -> hasproperty(value, field), ELASTIC_FIELD_NAMES) ||
            throw(ArgumentError("initial named tuple must contain vx, vy, sxx, syy, and sxy"))
        return (
            Float64(value.vx),
            Float64(value.vy),
            Float64(value.sxx),
            Float64(value.syy),
            Float64(value.sxy),
        )
    elseif value isa Tuple && length(value) == ELASTIC_FIELD_COUNT
        return ntuple(index -> Float64(value[index]), ELASTIC_FIELD_COUNT)
    end

    throw(ArgumentError("initial condition must return a named tuple or 5-tuple"))
end

function interpolate_elastic_state(initial, mesh)
    triangles = oriented_triangle_connectivities(mesh)
    triangle_total = length(triangles)
    state = zeros(Float64, ELASTIC_FIELD_COUNT * ELASTIC_LOCAL_DOF_COUNT * triangle_total)

    for (triangle, points) in pairs(triangles)
        coords = triangle_coordinates(mesh, points)
        for local_index in 1:ELASTIC_LOCAL_DOF_COUNT
            values = elastic_components(initial(coords[1, local_index], coords[2, local_index]))
            for field in 1:ELASTIC_FIELD_COUNT
                state[elastic_dof(triangle, local_index, field, triangle_total)] = values[field]
            end
        end
    end

    return state
end

function pressure_wave_speed(material::ElasticMaterial)
    return sqrt((material.lambda + 2 * material.mu) / material.rho)
end

function elastic_flux_x(q, material::ElasticMaterial)
    vx, vy, sxx, syy, sxy = q
    return (
        sxx / material.rho,
        sxy / material.rho,
        (material.lambda + 2 * material.mu) * vx,
        material.lambda * vx,
        material.mu * vy,
    )
end

function elastic_flux_y(q, material::ElasticMaterial)
    vx, vy, sxx, syy, sxy = q
    return (
        sxy / material.rho,
        syy / material.rho,
        material.lambda * vy,
        (material.lambda + 2 * material.mu) * vy,
        material.mu * vx,
    )
end

function physical_normal_flux(q, normal, material::ElasticMaterial)
    flux_x = elastic_flux_x(q, material)
    flux_y = elastic_flux_y(q, material)
    return ntuple(
        field -> normal[1] * flux_x[field] + normal[2] * flux_y[field],
        ELASTIC_FIELD_COUNT,
    )
end

function normal_flux(left, right, normal, material::ElasticMaterial)
    left_flux = physical_normal_flux(left, normal, material)
    right_flux = physical_normal_flux(right, normal, material)
    alpha = pressure_wave_speed(material)

    return ntuple(
        field -> 0.5 * (left_flux[field] + right_flux[field]) +
                 -0.5 * alpha * (left[field] - right[field]),
        ELASTIC_FIELD_COUNT,
    )
end

function state_at_point(state::AbstractVector{<:Real}, cell::Integer, phi, ncells::Integer)
    return ntuple(
        field -> sum(
            state[elastic_dof(cell, local_index, field, ncells)] * phi[local_index] for
            local_index in 1:ELASTIC_LOCAL_DOF_COUNT
        ),
        ELASTIC_FIELD_COUNT,
    )
end

function boundary_state(q, normal, boundary::Symbol)
    if boundary === :reflecting
        return reflecting_state(q, normal)
    elseif boundary === :traction_free
        return traction_free_state(q, normal)
    end

    validate_elastic_boundary(boundary)
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
    ghost_sxy = sxy - 2 * (traction_x * ny + nx * traction_y) + 2 * normal_traction * nx * ny
    return (ghost_sxx, ghost_syy, ghost_sxy)
end

function elastic_rhs(state::AbstractVector{<:Real}, mesh, material::ElasticMaterial, boundary::Symbol)
    validate_elastic_boundary(boundary)

    triangles = oriented_triangle_connectivities(mesh)
    triangle_total = length(triangles)
    expected_length = ELASTIC_FIELD_COUNT * ELASTIC_LOCAL_DOF_COUNT * triangle_total
    length(state) == expected_length ||
        throw(ArgumentError("elastic state length does not match mesh"))

    residual = zeros(Float64, expected_length)
    add_elastic_volume_terms!(residual, state, mesh, triangles, material, triangle_total)
    add_elastic_face_terms!(residual, state, mesh, triangles, material, boundary, triangle_total)
    return apply_elastic_mass_inverse(residual, mesh, triangles, triangle_total)
end

function add_elastic_volume_terms!(residual, state, mesh, triangles, material::ElasticMaterial, triangle_total::Integer)
    for (triangle, points) in pairs(triangles)
        coords = triangle_coordinates(mesh, points)
        area, grads = triangle_geometry(coords)

        for (lambdas, weight) in TRIANGLE_QUADRATURE
            q = state_at_point(state, triangle, lambdas, triangle_total)
            flux_x = elastic_flux_x(q, material)
            flux_y = elastic_flux_y(q, material)

            for local_index in 1:ELASTIC_LOCAL_DOF_COUNT
                scale_x = -area * weight * grads[1, local_index]
                scale_y = -area * weight * grads[2, local_index]
                for field in 1:ELASTIC_FIELD_COUNT
                    row = elastic_dof(triangle, local_index, field, triangle_total)
                    residual[row] += scale_x * flux_x[field] + scale_y * flux_y[field]
                end
            end
        end
    end

    return nothing
end

function add_elastic_face_terms!(
    residual,
    state,
    mesh,
    triangles,
    material::ElasticMaterial,
    boundary::Symbol,
    triangle_total::Integer,
)
    for facet in facet_adjacencies(mesh)
        if facet.triangles[2] == 0
            add_elastic_boundary_face!(
                residual,
                state,
                mesh,
                triangles,
                facet,
                material,
                boundary,
                triangle_total,
            )
        else
            add_elastic_interior_face!(residual, state, mesh, triangles, facet, material, triangle_total)
        end
    end

    return nothing
end

function add_elastic_interior_face!(
    residual,
    state,
    mesh,
    triangles,
    facet::FacetAdjacency,
    material::ElasticMaterial,
    triangle_total::Integer,
)
    left_triangle, right_triangle = facet.triangles
    left_edge, right_edge = facet.local_edges
    left_points = triangles[left_triangle]
    right_points = triangles[right_triangle]
    normal, edge_length = edge_normal(mesh, left_points, left_edge)
    left_coords = triangle_coordinates(mesh, left_points)
    right_coords = triangle_coordinates(mesh, right_points)

    for (s, weight_1d) in EDGE_QUADRATURE
        x, y = edge_point(mesh, left_points, left_edge, s)
        left_phi = basis_values_at_point(left_coords, x, y)
        right_phi = basis_values_at_point(right_coords, x, y)
        left_state = state_at_point(state, left_triangle, left_phi, triangle_total)
        right_state = state_at_point(state, right_triangle, right_phi, triangle_total)
        flux = normal_flux(left_state, right_state, normal, material)
        weight = edge_length * weight_1d

        for local_index in 1:ELASTIC_LOCAL_DOF_COUNT
            for field in 1:ELASTIC_FIELD_COUNT
                residual[elastic_dof(left_triangle, local_index, field, triangle_total)] +=
                    weight * left_phi[local_index] * flux[field]
                residual[elastic_dof(right_triangle, local_index, field, triangle_total)] -=
                    weight * right_phi[local_index] * flux[field]
            end
        end
    end

    return nothing
end

function add_elastic_boundary_face!(
    residual,
    state,
    mesh,
    triangles,
    facet::FacetAdjacency,
    material::ElasticMaterial,
    boundary::Symbol,
    triangle_total::Integer,
)
    triangle = facet.triangles[1]
    local_edge = facet.local_edges[1]
    points = triangles[triangle]
    normal, edge_length = edge_normal(mesh, points, local_edge)
    coords = triangle_coordinates(mesh, points)

    for (s, weight_1d) in EDGE_QUADRATURE
        x, y = edge_point(mesh, points, local_edge, s)
        phi = basis_values_at_point(coords, x, y)
        interior_state = state_at_point(state, triangle, phi, triangle_total)
        ghost_state = boundary_state(interior_state, normal, boundary)
        flux = normal_flux(interior_state, ghost_state, normal, material)
        weight = edge_length * weight_1d

        for local_index in 1:ELASTIC_LOCAL_DOF_COUNT
            for field in 1:ELASTIC_FIELD_COUNT
                residual[elastic_dof(triangle, local_index, field, triangle_total)] +=
                    weight * phi[local_index] * flux[field]
            end
        end
    end

    return nothing
end

function apply_elastic_mass_inverse(residual::AbstractVector{<:Real}, mesh, triangle_total::Integer)
    triangles = oriented_triangle_connectivities(mesh)
    return apply_elastic_mass_inverse(residual, mesh, triangles, triangle_total)
end

function apply_elastic_mass_inverse(
    residual::AbstractVector{<:Real},
    mesh,
    triangles,
    triangle_total::Integer,
)
    rhs = similar(Vector{Float64}(residual))

    for (triangle, points) in pairs(triangles)
        area, _ = triangle_geometry(triangle_coordinates(mesh, points))
        for field in 1:ELASTIC_FIELD_COUNT
            dof1 = elastic_dof(triangle, 1, field, triangle_total)
            dof2 = elastic_dof(triangle, 2, field, triangle_total)
            dof3 = elastic_dof(triangle, 3, field, triangle_total)
            r1 = residual[dof1]
            r2 = residual[dof2]
            r3 = residual[dof3]

            rhs[dof1] = (9 * r1 - 3 * r2 - 3 * r3) / area
            rhs[dof2] = (-3 * r1 + 9 * r2 - 3 * r3) / area
            rhs[dof3] = (-3 * r1 - 3 * r2 + 9 * r3) / area
        end
    end

    return rhs
end

function named_elastic_state(values)
    return (
        vx=Float64(values[1]),
        vy=Float64(values[2]),
        sxx=Float64(values[3]),
        syy=Float64(values[4]),
        sxy=Float64(values[5]),
    )
end

function evaluate_elastic_state(result::ElasticResult, x::Real, y::Real)
    triangles = oriented_triangle_connectivities(result.mesh)
    triangle_total = length(triangles)

    for (triangle, points) in pairs(triangles)
        coords = triangle_coordinates(result.mesh, points)
        lambdas = barycentric_coordinates(coords, x, y)

        if all(lambda -> lambda >= -1.0e-10 && lambda <= 1.0 + 1.0e-10, lambdas)
            return named_elastic_state(state_at_point(result.state, triangle, lambdas, triangle_total))
        end
    end

    throw(ArgumentError("point is outside the mesh"))
end

function elastic_energy(result::ElasticResult)
    return elastic_energy(result.mesh, result.state, result.material)
end

function elastic_energy(mesh, state::AbstractVector{<:Real}, material::ElasticMaterial)
    triangles = oriented_triangle_connectivities(mesh)
    triangle_total = length(triangles)
    expected_length = ELASTIC_FIELD_COUNT * ELASTIC_LOCAL_DOF_COUNT * triangle_total
    length(state) == expected_length ||
        throw(ArgumentError("elastic state length does not match mesh"))

    total = 0.0
    for (triangle, points) in pairs(triangles)
        coords = triangle_coordinates(mesh, points)
        area, _ = triangle_geometry(coords)

        for (lambdas, weight) in TRIANGLE_QUADRATURE
            q = state_at_point(state, triangle, lambdas, triangle_total)
            total += area * weight * elastic_energy_density(q, material)
        end
    end

    return total
end

function elastic_energy_density(q, material::ElasticMaterial)
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
    hmin = Inf
    triangles = oriented_triangle_connectivities(mesh)

    for facet in facet_adjacencies(mesh)
        triangle = facet.triangles[1]
        local_edge = facet.local_edges[1]
        _, edge_length = edge_normal(mesh, triangles[triangle], local_edge)
        hmin = min(hmin, edge_length)
    end

    return hmin
end

function default_elastic_dt(mesh, material::ElasticMaterial, cfl::Real)
    cfl > 0 || throw(ArgumentError("cfl must be positive"))
    return Float64(cfl) * minimum_edge_length(mesh) / pressure_wave_speed(material)
end

function ssprk3_step(
    state::Vector{Float64},
    dt::Float64,
    mesh,
    material::ElasticMaterial,
    boundary::Symbol,
)
    rhs0 = elastic_rhs(state, mesh, material, boundary)
    u1 = state .+ dt .* rhs0

    rhs1 = elastic_rhs(u1, mesh, material, boundary)
    u2 = 0.75 .* state .+ 0.25 .* (u1 .+ dt .* rhs1)

    rhs2 = elastic_rhs(u2, mesh, material, boundary)
    return (1 / 3) .* state .+ (2 / 3) .* (u2 .+ dt .* rhs2)
end

function solve_elastodynamics(
    initial;
    nx::Integer=20,
    ny::Integer=20,
    mesh=nothing,
    material::ElasticMaterial=ElasticMaterial(1.0, 1.0, 0.5),
    tspan=(0.0, 0.1),
    dt=nothing,
    cfl::Real=0.1,
    boundary::Symbol=:reflecting,
    save_history::Bool=false,
)
    boundary = validate_elastic_boundary(boundary)
    t0 = Float64(tspan[1])
    tend = Float64(tspan[2])
    tend >= t0 || throw(ArgumentError("tspan end must be greater than or equal to start"))

    dg_mesh = resolve_mesh(mesh, nx, ny)
    state = interpolate_elastic_state(initial, dg_mesh)
    step_dt = dt === nothing ? default_elastic_dt(dg_mesh, material, cfl) : Float64(dt)
    step_dt > 0 || throw(ArgumentError("dt must be positive"))

    times = [t0]
    energy_history = [elastic_energy(dg_mesh, state, material)]
    state_history = save_history ? [copy(state)] : nothing
    time = t0

    while time < tend
        step = min(step_dt, tend - time)
        state = ssprk3_step(state, Float64(step), dg_mesh, material, boundary)
        time += step

        if tend - time <= 10 * eps(max(abs(tend), 1.0))
            time = tend
        end

        push!(times, time)
        push!(energy_history, elastic_energy(dg_mesh, state, material))
        if save_history
            push!(state_history, copy(state))
        end
    end

    return ElasticResult(dg_mesh, state, material, times, energy_history, boundary, state_history)
end
