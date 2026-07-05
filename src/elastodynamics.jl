struct ElasticMaterial
    rho::Float64
    lambda::Float64
    mu::Float64

    function ElasticMaterial(rho::Real, lambda::Real, mu::Real)
        rho > 0 || throw(ArgumentError("rho must be positive"))
        lambda >= 0 || throw(ArgumentError("lambda must be nonnegative"))
        mu > 0 || throw(ArgumentError("mu must be positive"))
        return new(Float64(rho), Float64(lambda), Float64(mu))
    end
end

struct ElasticResult
    mesh::TriMesh
    state::Vector{Float64}
    material::ElasticMaterial
    times::Vector{Float64}
    energy_history::Vector{Float64}
    boundary::Symbol
end

const ELASTIC_FIELD_NAMES = (:vx, :vy, :sxx, :syy, :sxy)
const ELASTIC_FIELD_COUNT = 5
const ELASTIC_LOCAL_DOF_COUNT = 3

function elastic_dof(cell::Integer, local_index::Integer, field::Integer, ncells::Integer)
    1 <= cell <= ncells || throw(ArgumentError("cell out of range"))
    1 <= local_index <= ELASTIC_LOCAL_DOF_COUNT || throw(ArgumentError("local_index out of range"))
    1 <= field <= ELASTIC_FIELD_COUNT || throw(ArgumentError("field out of range"))
    return ELASTIC_LOCAL_DOF_COUNT * ncells * (field - 1) +
           ELASTIC_LOCAL_DOF_COUNT * (cell - 1) +
           local_index
end

function validate_elastic_boundary(boundary::Symbol)
    boundary in (:reflecting, :traction_free) ||
        throw(ArgumentError("boundary must be :reflecting or :traction_free"))
    return boundary
end

function elastic_components(value)
    if value isa NamedTuple
        all(field -> hasproperty(value, field), ELASTIC_FIELD_NAMES) ||
            throw(ArgumentError("initial named tuple must contain vx, vy, sxx, syy, and sxy"))
        return (
            Float64(value.vx),
            Float64(value.vy),
            Float64(value.sxx),
            Float64(value.syy),
            Float64(value.sxy),
        )
    elseif value isa Tuple && length(value) == ELASTIC_FIELD_COUNT
        return ntuple(index -> Float64(value[index]), ELASTIC_FIELD_COUNT)
    end

    throw(ArgumentError("initial condition must return a named tuple or 5-tuple"))
end

function interpolate_elastic_state(initial, mesh::TriMesh)
    ncells = size(mesh.cells, 2)
    state = zeros(Float64, ELASTIC_FIELD_COUNT * ELASTIC_LOCAL_DOF_COUNT * ncells)

    for cell in 1:ncells
        coords = cell_coordinates(mesh, cell)
        for local_index in 1:ELASTIC_LOCAL_DOF_COUNT
            values = elastic_components(initial(coords[1, local_index], coords[2, local_index]))
            for field in 1:ELASTIC_FIELD_COUNT
                state[elastic_dof(cell, local_index, field, ncells)] = values[field]
            end
        end
    end

    return state
end
