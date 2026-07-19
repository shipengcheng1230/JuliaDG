module Poisson

import Gridap
using Gridap:
    AffineFEOperator,
    BoundaryTriangulation,
    CellField,
    Measure,
    ReferenceFE,
    SkeletonTriangulation,
    TestFESpace,
    TrialFESpace,
    Triangulation,
    get_array,
    get_normal_vector,
    jump,
    lagrangian,
    mean,
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
    penalty::Real = 10.0,
    dirichlet_tags,
    dirichlet,
    neumann_tags = nothing,
    neumann = x -> 0.0,
)
    order > 0 || throw(ArgumentError("order must be positive"))
    penalty > 0 || throw(ArgumentError("penalty must be positive"))
    essential = required_tags(model, dirichlet_tags; keyword = "dirichlet_tags")
    natural = optional_tags(model, neumann_tags; keyword = "neumann_tags")
    validate_boundary_roles(model, essential, natural)

    p = Int(order)
    reffe = ReferenceFE(lagrangian, Float64, p)
    test = TestFESpace(model, reffe; conformity = :L2)
    trial = TrialFESpace(test)
    domain = Triangulation(model)
    skeleton = SkeletonTriangulation(model)
    dirichlet_boundary = BoundaryTriangulation(model; tags = essential)
    dΩ = Measure(domain, 2 * p)
    dΛ = Measure(skeleton, 2 * p)
    dΓD = Measure(dirichlet_boundary, 2 * p)

    nΛ = get_normal_vector(skeleton)
    nD = get_normal_vector(dirichlet_boundary)
    γ = Float64(penalty) * p^2
    βΛ = CellField(γ ./ get_array(∫(1) * dΛ), skeleton)
    βD = CellField(γ ./ get_array(∫(1) * dΓD), dirichlet_boundary)

    a(u, v) =
        ∫(∇(v) ⋅ ∇(u)) * dΩ +
        ∫(βΛ * (jump(v * nΛ) ⋅ jump(u * nΛ)) - jump(v * nΛ) ⋅ mean(∇(u)) - mean(∇(v)) ⋅ jump(u * nΛ)) * dΛ +
        ∫(βD * v * u - v * (nD ⋅ ∇(u)) - (nD ⋅ ∇(v)) * u) * dΓD

    dirichlet_load(v) = ∫((βD * v - nD ⋅ ∇(v)) * dirichlet) * dΓD

    linear_form = if isempty(natural)
        v -> ∫(v * source) * dΩ + dirichlet_load(v)
    else
        boundary = BoundaryTriangulation(model; tags = natural)
        dΓ = Measure(boundary, 2 * p)
        v -> ∫(v * source) * dΩ + dirichlet_load(v) + ∫(v * neumann) * dΓ
    end

    operator = AffineFEOperator(a, linear_form, trial, test)
    return Result(model, Gridap.solve(operator), p)
end

function l2_error(result::Result, exact; degree::Integer = 2 * result.order)
    degree > 0 || throw(ArgumentError("degree must be positive"))
    domain = Triangulation(result.model)
    dΩ = Measure(domain, Int(degree))
    error_field = result.solution - exact
    return sqrt(sum(∫(error_field * error_field) * dΩ))
end

end
