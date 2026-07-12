function unit_square_model(nx::Integer, ny::Integer)
    nx > 0 || throw(ArgumentError("nx must be positive"))
    ny > 0 || throw(ArgumentError("ny must be positive"))

    return Gridap.CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (Int(nx), Int(ny)))
end
