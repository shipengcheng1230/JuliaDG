module Poisson

import Gridap
using Gridap:
    AffineFEOperator,
    BoundaryTriangulation,
    Measure,
    ReferenceFE,
    TestFESpace,
    TrialFESpace,
    Triangulation,
    lagrangian,
    ∫,
    ∇,
    ⋅
using ..JuliaDG: optional_tags, required_tags, validate_boundary_roles

export Result, solve, l2_error

struct Result{M,U}
    model::M
    solution::U
    order::Int
end

function solve(
    model::Gridap.DiscreteModel,
    source;
    order::Integer = 1,
    dirichlet_tags,
    dirichlet,
    neumann_tags = nothing,
    neumann = x -> 0.0,
)
    order > 0 || throw(ArgumentError("order must be positive"))
    essential = required_tags(model, dirichlet_tags; keyword = "dirichlet_tags")
    natural = optional_tags(model, neumann_tags; keyword = "neumann_tags")
    validate_boundary_roles(model, essential, natural)

    reffe = ReferenceFE(lagrangian, Float64, Int(order))
    test = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = essential)
    trial = TrialFESpace(test, dirichlet)
    domain = Triangulation(model)
    dΩ = Measure(domain, 2 * Int(order))

    a(u, v) = ∫(∇(v) ⋅ ∇(u)) * dΩ

    linear_form = if isempty(natural)
        v -> ∫(v * source) * dΩ
    else
        boundary = BoundaryTriangulation(model; tags = natural)
        dΓ = Measure(boundary, 2 * Int(order))
        v -> ∫(v * source) * dΩ + ∫(v * neumann) * dΓ
    end

    operator = AffineFEOperator(a, linear_form, trial, test)
    return Result(model, Gridap.solve(operator), Int(order))
end

function l2_error(result::Result, exact; degree::Integer = 2 * result.order)
    degree > 0 || throw(ArgumentError("degree must be positive"))
    domain = Triangulation(result.model)
    dΩ = Measure(domain, Int(degree))
    error_field = result.solution - exact
    return sqrt(sum(∫(error_field * error_field) * dΩ))
end

end
