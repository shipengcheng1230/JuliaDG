function assemble_triangle_terms! end
function assemble_face_terms! end
function assemble_interior_face! end
function assemble_boundary_face! end

function assemble_poisson_sipg(mesh, f, g; penalty::Real=20.0)
    triangles = oriented_triangle_connectivities(mesh)
    facets = facet_adjacencies(mesh)
    ndofs = 3 * length(triangles)
    rows = Int[]
    cols = Int[]
    values = Float64[]
    b = zeros(Float64, ndofs)

    coordinates_for(points) = begin
        coords = Matrix{Float64}(undef, 2, 3)
        for local_index in 1:3
            x, y = point_xy(mesh, points[local_index])
            coords[1, local_index] = x
            coords[2, local_index] = y
        end
        coords
    end
    edge_normal_for(points, local_edge) = begin
        start_index, end_index = LOCAL_EDGES[local_edge]
        x1, y1 = point_xy(mesh, points[start_index])
        x2, y2 = point_xy(mesh, points[end_index])
        dx = x2 - x1
        dy = y2 - y1
        edge_len = hypot(dx, dy)
        ((dy / edge_len, -dx / edge_len), edge_len)
    end
    edge_point_for(points, local_edge, s) = begin
        start_index, end_index = LOCAL_EDGES[local_edge]
        x1, y1 = point_xy(mesh, points[start_index])
        x2, y2 = point_xy(mesh, points[end_index])
        ((1 - s) * x1 + s * x2, (1 - s) * y1 + s * y2)
    end

    function assemble_triangle_terms_local!()
        for (triangle, points) in pairs(triangles)
            coords = coordinates_for(points)
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

    function assemble_interior_face_local!(facet)
        left_triangle, right_triangle = facet.triangles
        left_edge, _ = facet.local_edges
        left_points = triangles[left_triangle]
        right_points = triangles[right_triangle]

        normal, h_face = edge_normal_for(left_points, left_edge)
        left_coords = coordinates_for(left_points)
        right_coords = coordinates_for(right_points)
        _, left_grads = triangle_geometry(left_coords)
        _, right_grads = triangle_geometry(right_coords)

        side_triangles = (left_triangle, right_triangle)
        side_grads = (left_grads, right_grads)
        jump_signs = (1.0, -1.0)

        for (s, weight_1d) in EDGE_QUADRATURE
            x, y = edge_point_for(left_points, left_edge, s)
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

    function assemble_boundary_face_local!(facet)
        triangle = facet.triangles[1]
        local_edge = facet.local_edges[1]
        points = triangles[triangle]

        normal, h_face = edge_normal_for(points, local_edge)
        coords = coordinates_for(points)
        _, grads = triangle_geometry(coords)

        for (s, weight_1d) in EDGE_QUADRATURE
            x, y = edge_point_for(points, local_edge, s)
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

    function assemble_face_terms_local!()
        for facet in facets
            if facet.triangles[2] == 0
                assemble_boundary_face_local!(facet)
            else
                assemble_interior_face_local!(facet)
            end
        end

        return nothing
    end

    assemble_triangle_terms_local!()
    assemble_face_terms_local!()

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
