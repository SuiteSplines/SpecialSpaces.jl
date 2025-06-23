"""
    setcoeffs!(f::M, v::Vector{T}, k::Int=1, slice::Base.UnitRange{Int}=Base.UnitRange(1,length(v))) where {M<:AbstractMapping,T<:Real}

Set coefficients of a mapping component to a slice of coefficients in a vector.

# Arguments:
- `f`: mapping in question
- `v`: vector with coeffs
- `slice`: range of indices of vector `v`
- `k`: component of field `f` 
"""
function setcoeffs!(f::M, v::Vector{T}, k::Int=1, slice::Base.UnitRange{Int}=Base.UnitRange(1,length(v))) where {M<:AbstractMapping,T<:Real}
    y = view(f[k].coeffs, :)
    y .= v[slice]
end

"""
    setcoeffs!(f::M, scalarspace::S, v::Vector{T}) where {M<:AbstractMapping,S<:ScalarSplineSpace,T<:Real}

Set coefficients of a mapping defined on a scalar space.

# Arguments:
- `f`: mapping in question
- `scalarspace`: space
- `v`: vector with coeffs with length `dim(scalarspace)`
"""
function setcoeffs!(f::M, scalarspace::S, v::Vector{T}) where {M<:AbstractMapping,S<:ScalarSplineSpace,T<:Real}
    ndofs = dimension(scalarspace)
    setcoeffs!(f, v, 1, 1:ndofs)
end

"""
    setcoeffs!(f::M, vectorspace::S, v::Vector{T}) where {M<:AbstractMapping,S<:VectorSplineSpace,T<:Real}

Set coefficients of a mapping defined on a vector space.

# Arguments:
- `f`: mapping in question
- `vectorspace`: space
- `v`: vector with coeffs with length `dim(vectorspace)`
"""
function setcoeffs!(f::M, vectorspace::S, v::Vector{T}) where {M<:AbstractMapping,S<:VectorSplineSpace,T<:Real}
    ncomp = length(vectorspace)
    for k = 1:ncomp
        setcoeffs!(f, v, k, indices(vectorspace, k))
    end
end

"""
    setcoeffs!(f::M, mixedspace::S, field::Symbol, v::Vector{T}) where {M<:AbstractMapping,S<:MixedSplineSpace,T<:Real}

Set coefficients of a mapping defined on a mixedspace.

# Arguments:
- `f`: mapping in question
- `mixedspace`: space
- `field`: space field
- `v`: vector with coeffs with length `dim(mixedspace)`
"""
function setcoeffs!(f::M, mixedspace::S, field::Symbol, v::Vector{T}) where {M<:AbstractMapping,S<:MixedSplineSpace,T<:Real}
    space = getfield(mixedspace, field)
    setcoeffs!(f, mixedspace, field, v, space)
end
function setcoeffs!(f::M, mixedspace::S, field::Symbol, v::Vector{T}, ::ScalarSplineSpace) where {M<:AbstractMapping,S<:MixedSplineSpace,T<:Real}
    setcoeffs!(f, v, 1, indices(mixedspace, field))
end
function setcoeffs!(f::M, mixedspace::S, field::Symbol, v::Vector{T}, V::VectorSplineSpace) where {M<:AbstractMapping,S<:MixedSplineSpace,T<:Real}
    ncomp = length(V)
    for k = 1:ncomp
        setcoeffs!(f, v, k, indices(mixedspace, field, k))
    end
end

"""
    getcoeffs(f::Field)
    getcoeffs(f::GeometricMapping)

Return a vector of vertically concatenated (tensor-product Bspline) mapping coefficients.
"""
getcoeffs(f::Field{<:Any,1}) = f[1].coeffs[:]
getcoeffs(f::Field{<:Any,Codim}) where {Codim} = vcat(ntuple(k -> view(f[k].coeffs, :), Codim)...)
getcoeffs(f::GeometricMapping{<:Any,1}) = f[1].coeffs[:]
getcoeffs(f::GeometricMapping{<:Any,Codim}) where {Codim} = vcat(ntuple(k -> view(f[k].coeffs, :), Codim)...)


"""
    Field(space::S) where {S<:ScalarSplineSpace}
    Field(space::S) where {S<:VectorSplineSpace}
    Field(space::S, s::Symbol) where {S<:MixedSplineSpace}

Construct field on some spline spaces.
"""
Field(space::S) where {S<:ScalarSplineSpace} = Field(TensorProductBspline(space))
Field(space::S) where {S<:VectorSplineSpace} = Field(TensorProductBspline(space)...)
Field(space::S, s::Symbol) where {S<:MixedSplineSpace} = Field(getfield(space, s))


import AbstractMappings: n_input_args, process_mapping_input, n_output_args

AbstractMappings.n_input_args(space::S) where {S<:ScalarSplineSpace} = length(space)
AbstractMappings.n_input_args(space::S) where {Dim,Codim,S<:VectorSplineSpace{Dim,Codim}} = Dim

function AbstractMappings.process_mapping_input(space::S) where {S<:ScalarSplineSpace}
    TensorProductBspline(space)
end

"""
    AbstractMappings.GeometricMapping(domain, args::ScalarScalarSpace; orientation::Int=1)
    AbstractMappings.GeometricMapping(domain, args::VectorSplineSpace; orientation::Int=1)
    AbstractMappings.GeometricMapping(domain, args::MixedSplineSpace, field::Symbol; orientation::Int=1)

Construct geometric mapping on some spline spaces.
"""
function AbstractMappings.GeometricMapping(domain, args::VectorSplineSpace; orientation::Int=1)
    AbstractMappings.GeometricMapping(domain, args...; orientation=orientation)
end

function AbstractMappings.GeometricMapping(domain, args::MixedSplineSpace, field::Symbol; orientation::Int=1)
    space = getfield(args, field)
    AbstractMappings.GeometricMapping(domain, space; orientation=orientation)
end