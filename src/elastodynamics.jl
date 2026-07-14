module Elastodynamics

import Gridap
using Gridap:
    BoundaryTriangulation,
    FESpace,
    Measure,
    ReferenceFE,
    TensorValue,
    Triangulation,
    VectorValue,
    ∫,
    ε,
    ⋅,
    ⊙,
    interpolate_everywhere,
    lagrangian,
    tr
using Gridap.Algebra: LUSolver, NLSolver
using Gridap.ODEs: Newmark, TransientLinearFEOperator, TransientTrialFESpace, ∂t
using ..JuliaDG: optional_tags, required_tags, validate_boundary_roles

export Material, Result, strain, stress, energy, solve

struct Material
    rho::Float64
    lambda::Float64
    mu::Float64

    function Material(rho::Real, lambda::Real, mu::Real)
        rho > 0 || throw(ArgumentError("rho must be positive"))
        mu > 0 || throw(ArgumentError("mu must be positive"))
        lambda + mu > 0 || throw(ArgumentError("lambda + mu must be positive"))
        return new(Float64(rho), Float64(lambda), Float64(mu))
    end
end

struct Result{M,U,V,S}
    model::M
    material::Material
    initial_displacement::U
    initial_velocity::V
    solution::S
    order::Int
end

strain(displacement) = ε(displacement)

function stress(displacement, material::Material)
    identity = TensorValue(1.0, 0.0, 0.0, 1.0)
    deformation = strain(displacement)
    return material.lambda * tr(deformation) * identity + 2 * material.mu * deformation
end

function energy(
    displacement,
    velocity,
    material::Material,
    model::Gridap.DiscreteModel;
    degree::Integer = 2,
)
    degree > 0 || throw(ArgumentError("degree must be positive"))
    domain = Triangulation(model)
    dΩ = Measure(domain, Int(degree))
    deformation = strain(displacement)
    kinetic = 0.5 * material.rho * (velocity ⋅ velocity)
    strain_density =
        material.mu * (deformation ⊙ deformation) +
        0.5 * material.lambda * tr(deformation) * tr(deformation)
    return sum(∫(kinetic + strain_density) * dΩ)
end

function validate_time_grid(tspan, dt::Real)
    tspan isa Tuple && length(tspan) == 2 ||
        throw(ArgumentError("tspan must be a two-element tuple"))
    t0, tF = Float64(tspan[1]), Float64(tspan[2])
    tF > t0 || throw(ArgumentError("tspan must have tF > t0"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    steps = (tF - t0) / dt
    isapprox(steps, round(steps); atol = 1.0e-12, rtol = 1.0e-12) ||
        throw(ArgumentError("tspan length must be an integer multiple of dt"))
    return t0, tF, Float64(dt)
end

function solve(
    model::Gridap.DiscreteModel;
    material::Material,
    tspan,
    dt::Real,
    order::Integer = 1,
    dirichlet_tags,
    displacement,
    traction_tags = nothing,
    traction = (t, x) -> VectorValue(0.0, 0.0),
    body_force = (t, x) -> VectorValue(0.0, 0.0),
    initial_displacement,
    initial_velocity,
)
    order > 0 || throw(ArgumentError("order must be positive"))
    t0, tF, Δt = validate_time_grid(tspan, dt)
    essential = required_tags(model, dirichlet_tags; keyword = "dirichlet_tags")
    natural = optional_tags(model, traction_tags; keyword = "traction_tags")
    validate_boundary_roles(model, essential, natural)

    reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, Int(order))
    test = FESpace(model, reffe; conformity = :H1, dirichlet_tags = essential)
    trial = TransientTrialFESpace(test, t -> x -> displacement(t, x))
    domain = Triangulation(model)
    dΩ = Measure(domain, 2 * Int(order))

    mass(t, acceleration, v) = ∫(material.rho * (v ⋅ acceleration)) * dΩ
    damping(t, velocity, v) = ∫(0.0 * (v ⋅ velocity)) * dΩ
    stiffness(t, u, v) =
        ∫(2 * material.mu * (ε(v) ⊙ ε(u)) + material.lambda * tr(ε(v)) * tr(ε(u))) * dΩ

    forcing = if isempty(natural)
        (t, v) -> ∫(v ⋅ (x -> body_force(t, x))) * dΩ
    else
        boundary = BoundaryTriangulation(model; tags = natural)
        dΓ = Measure(boundary, 2 * Int(order))
        (t, v) ->
            ∫(v ⋅ (x -> body_force(t, x))) * dΩ + ∫(v ⋅ (x -> traction(t, x))) * dΓ
    end

    operator = TransientLinearFEOperator(
        (stiffness, damping, mass),
        forcing,
        trial,
        test;
        constant_forms = (true, true, true),
    )
    displacement0 = interpolate_everywhere(initial_displacement, trial(t0))
    velocity0 = interpolate_everywhere(initial_velocity, ∂t(trial)(t0))
    linear_solver = LUSolver()
    nonlinear_solver =
        NLSolver(linear_solver; show_trace = false, method = :newton, iterations = 10)
    time_solver = Newmark(nonlinear_solver, Δt, 0.5, 0.25)
    solution =
        Gridap.Algebra.solve(time_solver, operator, t0, tF, (displacement0, velocity0))
    return Result(model, material, displacement0, velocity0, solution, Int(order))
end

end
