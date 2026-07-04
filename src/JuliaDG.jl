module JuliaDG

using LinearAlgebra
using SparseArrays

export TriMesh,
    DGResult,
    unit_square_mesh,
    assemble_poisson_sipg,
    solve_poisson,
    evaluate_solution,
    l2_error

include("mesh.jl")
include("basis.jl")
include("assembly.jl")
include("solve.jl")

end
