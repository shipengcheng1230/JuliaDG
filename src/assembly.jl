function assemble_poisson_sipg(mesh, f, g; penalty::Real=20.0)
    ndofs = 3 * triangle_count(mesh)
    rows = Int[]
    cols = Int[]
    values = Float64[]
    b = zeros(Float64, ndofs)

    assemble_triangle_terms!(rows, cols, values, b, mesh, f)
    assemble_face_terms!(rows, cols, values, b, mesh, g, Float64(penalty))

    return sparse(rows, cols, values, ndofs, ndofs), b
end

global_dof(cell::Integer, local_index::Integer) = 3 * (cell - 1) + local_index

function add_entry!(rows, cols, values, row::Integer, col::Integer, value::Real)
    push!(rows, row)
    push!(cols, col)
    push!(values, Float64(value))
    return nothing
end

normal_dot(grads::AbstractMatrix{<:Real}, local_index::Integer, normal) =
    grads[1, local_index] * normal[1] + grads[2, local_index] * normal[2]

function assemble_triangle_terms!(rows, cols, values, b, mesh, f)
    for triangle in 1:triangle_count(mesh)
        coords = triangle_coordinates(mesh, triangle)
        area, grads = triangle_geometry(coords)

        for test_local in 1:3
            row = global_dof(triangle, test_local)
            for trial_local in 1:3
                col = global_dof(triangle, trial_local)
                stiffness = area * dot(grads[:, test_local], grads[:, trial_local])
                add_entry!(rows, cols, values, row, col, stiffness)
            end
        end

        for (lambdas, weight) in TRIANGLE_QUADRATURE
            x, y = physical_point(coords, lambdas)
            f_value = f(x, y)
            for test_local in 1:3
                b[global_dof(triangle, test_local)] += area * weight * f_value * lambdas[test_local]
            end
        end
    end

    return nothing
end

function assemble_face_terms!(rows, cols, values, b, mesh, g, penalty::Float64)
    for facet in facet_adjacencies(mesh)
        if facet.triangles[2] == 0
            assemble_boundary_face!(rows, cols, values, b, mesh, facet, g, penalty)
        else
            assemble_interior_face!(rows, cols, values, mesh, facet, penalty)
        end
    end

    return nothing
end

function assemble_interior_face!(rows, cols, values, mesh, facet::FacetAdjacency, penalty::Float64)
    left_triangle, right_triangle = facet.triangles
    left_edge, right_edge = facet.local_edges

    normal, h_face = edge_normal(mesh, left_triangle, left_edge)
    left_coords = triangle_coordinates(mesh, left_triangle)
    right_coords = triangle_coordinates(mesh, right_triangle)
    _, left_grads = triangle_geometry(left_coords)
    _, right_grads = triangle_geometry(right_coords)

    side_triangles = (left_triangle, right_triangle)
    side_grads = (left_grads, right_grads)
    jump_signs = (1.0, -1.0)

    for (s, weight_1d) in EDGE_QUADRATURE
        x, y = edge_point(mesh, left_triangle, left_edge, s)
        left_phi = basis_values_at_point(left_coords, x, y)
        right_phi = basis_values_at_point(right_coords, x, y)
        side_phi = (left_phi, right_phi)
        weight = h_face * weight_1d

        for test_side in 1:2
            for trial_side in 1:2
                for test_local in 1:3
                    row = global_dof(side_triangles[test_side], test_local)
                    jump_test = jump_signs[test_side] * side_phi[test_side][test_local]
                    avg_flux_test =
                        0.5 * normal_dot(side_grads[test_side], test_local, normal)

                    for trial_local in 1:3
                        col = global_dof(side_triangles[trial_side], trial_local)
                        jump_trial = jump_signs[trial_side] * side_phi[trial_side][trial_local]
                        avg_flux_trial =
                            0.5 * normal_dot(side_grads[trial_side], trial_local, normal)

                        value = weight * (
                            -avg_flux_trial * jump_test -
                            avg_flux_test * jump_trial +
                            penalty / h_face * jump_trial * jump_test
                        )
                        add_entry!(rows, cols, values, row, col, value)
                    end
                end
            end
        end
    end

    return nothing
end

function assemble_boundary_face!(rows, cols, values, b, mesh, facet::FacetAdjacency, g, penalty::Float64)
    triangle = facet.triangles[1]
    local_edge = facet.local_edges[1]

    normal, h_face = edge_normal(mesh, triangle, local_edge)
    coords = triangle_coordinates(mesh, triangle)
    _, grads = triangle_geometry(coords)

    for (s, weight_1d) in EDGE_QUADRATURE
        x, y = edge_point(mesh, triangle, local_edge, s)
        phi = basis_values_at_point(coords, x, y)
        g_value = g(x, y)
        weight = h_face * weight_1d

        for test_local in 1:3
            row = global_dof(triangle, test_local)
            flux_test = normal_dot(grads, test_local, normal)

            for trial_local in 1:3
                col = global_dof(triangle, trial_local)
                flux_trial = normal_dot(grads, trial_local, normal)
                value = weight * (
                    -flux_trial * phi[test_local] -
                    flux_test * phi[trial_local] +
                    penalty / h_face * phi[trial_local] * phi[test_local]
                )
                add_entry!(rows, cols, values, row, col, value)
            end

            b[row] += weight * (-flux_test + penalty / h_face * phi[test_local]) * g_value
        end
    end

    return nothing
end
