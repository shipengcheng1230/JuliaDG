module JuliaDG

using Gridap

export Poisson, Elastodynamics, unit_square_model

include("model.jl")
include("poisson.jl")
include("elastodynamics.jl")

end
