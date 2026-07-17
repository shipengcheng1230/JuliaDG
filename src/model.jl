function unit_square_model(nx::Integer, ny::Integer)
    nx > 0 || throw(ArgumentError("nx must be positive"))
    ny > 0 || throw(ArgumentError("ny must be positive"))

    return Gridap.CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (Int(nx), Int(ny)))
end

function normalized_tags(tags; keyword::AbstractString, required::Bool)
    if tags === nothing
        required && throw(ArgumentError("$keyword must contain at least one tag"))
        return String[]
    elseif tags isa AbstractString
        return [String(tags)]
    elseif tags isa AbstractVector{<:AbstractString}
        values = String.(tags)
        isempty(values) && required && throw(ArgumentError("$keyword must contain at least one tag"))
        return unique(values)
    end

    throw(ArgumentError("$keyword must be a tag name or a vector of tag names"))
end

function validate_known_tags(model::Gridap.DiscreteModel, tags::Vector{String}; keyword::AbstractString)
    names = Set(Gridap.Geometry.get_tag_name(Gridap.get_face_labeling(model)))
    missing = sort!(collect(setdiff(Set(tags), names)))
    isempty(missing) || throw(ArgumentError("$keyword contains unknown Gridap tag(s): $(join(missing, ", "))"))
    return tags
end

function required_tags(model::Gridap.DiscreteModel, tags; keyword::AbstractString)
    values = normalized_tags(tags; keyword = keyword, required = true)
    return validate_known_tags(model, values; keyword = keyword)
end

function optional_tags(model::Gridap.DiscreteModel, tags; keyword::AbstractString)
    values = normalized_tags(tags; keyword = keyword, required = false)
    return validate_known_tags(model, values; keyword = keyword)
end

function validate_boundary_roles(model::Gridap.DiscreteModel, essential::Vector{String}, natural::Vector{String})
    labels = Gridap.get_face_labeling(model)
    boundary_dimension = Gridap.num_dims(model) - 1
    topology = Gridap.Geometry.get_grid_topology(model)
    boundary_faces = Gridap.Geometry.get_isboundary_face(topology, boundary_dimension)
    essential_mask = Gridap.Geometry.get_face_mask(labels, essential, boundary_dimension)
    essential_faces = essential_mask .& boundary_faces
    any(essential_faces) && essential_faces == essential_mask ||
        throw(ArgumentError("Dirichlet tags must select physical boundary faces"))
    isempty(natural) && return nothing
    natural_mask = Gridap.Geometry.get_face_mask(labels, natural, boundary_dimension)
    natural_faces = natural_mask .& boundary_faces
    any(natural_faces) && natural_faces == natural_mask ||
        throw(ArgumentError("natural boundary tags must select physical boundary faces"))
    any(essential_faces .& natural_faces) &&
        throw(ArgumentError("Dirichlet and natural boundary tags select common faces"))
    return nothing
end
