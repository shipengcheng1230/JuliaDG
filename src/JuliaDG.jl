module JuliaDG

using LinearAlgebra
using SparseArrays
import Meshes

export TriMesh,
    DGResult,
    ElasticMaterial,
    ElasticResult,
    unit_square_mesh,
    mesh_backend,
    assemble_poisson_sipg,
    solve_poisson,
    solve_elastodynamics,
    evaluate_solution,
    evaluate_elastic_state,
    l2_error,
    elastic_energy,
    dg_plot_data,
    elastic_plot_data,
    plot_solution,
    record_solution

include("mesh.jl")
include("basis.jl")
include("assembly.jl")
include("solve.jl")
include("elastodynamics.jl")
include("plotting.jl")

end
