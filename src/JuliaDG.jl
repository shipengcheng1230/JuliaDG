module JuliaDG

import Meshes

export Poisson, Elastodynamics, unit_square_mesh

include("triangulation.jl")
include("basis.jl")
include("poisson.jl")
include("elastodynamics.jl")

end
