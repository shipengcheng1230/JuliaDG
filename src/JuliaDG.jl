module JuliaDG

using LinearAlgebra
using SparseArrays

export TriMesh, unit_square_mesh, assemble_poisson_sipg

include("mesh.jl")
include("basis.jl")
include("assembly.jl")

end
