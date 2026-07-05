module JuliaDG

using LinearAlgebra
using SparseArrays

export TriMesh,
    DGResult,
    unit_square_mesh,
    assemble_poisson_sipg,
    solve_poisson,
    evaluate_solution,
    l2_error,
    dg_plot_data,
    plot_solution

include("mesh.jl")
include("basis.jl")
include("assembly.jl")
include("solve.jl")
include("plotting.jl")

end
