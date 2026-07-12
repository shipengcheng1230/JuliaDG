module JuliaDG

using Gridap
import Meshes

export Poisson, Elastodynamics, unit_square_mesh, unit_square_model

include("model.jl")
include("triangulation.jl")
include("basis.jl")
include("poisson.jl")
include("elastodynamics.jl")

end
