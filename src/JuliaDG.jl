module JuliaDG

import Meshes

export Poisson, ElasticMaterial, ElasticResult, unit_square_mesh, resolve_mesh,
    solve_elastodynamics, evaluate_elastic_state, elastic_energy,
    elastic_plot_data, plot_solution, record_solution

include("triangulation.jl")
include("basis.jl")
include("poisson.jl")
include("elastodynamics.jl")
include("plotting.jl")

end
