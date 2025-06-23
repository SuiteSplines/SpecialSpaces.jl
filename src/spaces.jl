# vector function spaces
"""
    abstract type VectorFunctionSpace{Dim,T}

Concrete vector spaces like [`VectorSplineSpace`](@ref) subtype this.

Vector function spaces are expected to collect components
as a tuple in field `:V`.
"""
abstract type VectorFunctionSpace{Dim,T} end

Base.parent(V::S) where {S<:VectorFunctionSpace} = getfield(V, :V)

"""
    Base.length(V::S) where {S<:VectorFunctionSpace}

Return the number of vector function space components.
"""
Base.length(V::S) where {S<:VectorFunctionSpace} = length(parent(V))

"""
    Base.getindex(V::S, i::Int64) where {S<:VectorFunctionSpace}

Return `i`th vector function space component.
"""
Base.getindex(V::S, i::Int64) where {S<:VectorFunctionSpace} = getindex(parent(V), i)

"""
    Base.iterate(V::S) where {S<:VectorFunctionSpace} = iterate(parent(V))
    Base.iterate(V::S, i::Int) where {S<:VectorFunctionSpace} = iterate(parent(V), i)

Iterate over vector function space components.
"""
Base.iterate(V::S) where {S<:VectorFunctionSpace} = iterate(parent(V))
Base.iterate(V::S, i::Int) where {S<:VectorFunctionSpace} = iterate(parent(V), i)

"""
    indices(space::S) where {S<:VectorFunctionSpace}

Return linear indices for a vector function space.
"""
function indices(space::S) where {S<:VectorFunctionSpace}
    lastind = dimension(space)
    Base.UnitRange(1, lastind)
end

"""
    indices(space::S, i::Int) where {S<:VectorFunctionSpace}

Return linear indices for `i`th component of a vector function space.
"""
function indices(space::S, i::Int) where {S<:VectorFunctionSpace}
    indices(space, Val(i))
end

function indices(space::S, ::Val{1}) where {S<:VectorFunctionSpace}
    lastind = dimension(space[1])
    return Base.UnitRange(1, lastind)
end

function indices(space::S, ::Val{k}) where {S<:VectorFunctionSpace,k}
    prev = indices(space, Val(k - 1))
    firstind = prev.stop + 1
    lastind = prev.stop + dimension(space[k])
    return Base.UnitRange(firstind, lastind)
end

"""
    IgaBase.dimension(V::T, k::Int) where {T<:VectorFunctionSpace}

Return the dimension of the `i`th component of a vector function space.
"""
function IgaBase.dimension(V::T, k::Int) where {T<:VectorFunctionSpace}
    dimension(V[k])
end

"""
    IgaBase.dimension(V::T) where {T<:VectorFunctionSpace}

Return the dimension of a vector function space.
"""
function IgaBase.dimension(V::T) where {T<:VectorFunctionSpace}
    sum(map(dimension, V))
end

"""
    dimensions(V::T, i::Int64) where {T<:VectorFunctionSpace}

Return the dimension in each tensor-product direction of
the `i`th component of a vector function space as a tuple.
"""
function dimensions(V::T, i::Int64) where {T<:VectorFunctionSpace}
    dimensions(V[i])
end

"""
    dimensions(V::T) where {T<:VectorFunctionSpace}

Return the dimension in each tensor-product direction
of a vector function space as a tuple.
"""
function dimensions(V::T) where {T<:VectorFunctionSpace}
    ntuple(i -> dimensions(V[i]), Val(length(V)))
end

"""
    extraction_operators(S::T; sparse::Bool=false) where {T<:VectorFunctionSpace}

Return the extraction operator for a vector function space.

If `sparse` is `true`, return a tuple of sparse matrices.
"""
function extraction_operators(S::T; sparse::Bool=false) where {T<:VectorFunctionSpace}
    return ntuple(k -> extraction_operator(S[k]; sparse=sparse), length(S))
end

function Base.show(io::IO, V::S) where {Dim,T,S<:VectorFunctionSpace{Dim,T}}
    ncomp = length(V)
    print(io, "$S with $ncomp components")
end



# mixed function spaces
"""
    abstract type MixedFunctionSpace{Dim,T}

Mixed spaces like [`MixedSplineSpace`](@ref) subtype this.

Mixed function spaces are expected to collect components as `struct` fields.
"""
abstract type MixedFunctionSpace{Dim,T} end

"""
    IgaBase.dimension(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}

Return the dimension of the `i`th component of a vector function space `field`
in a mixed space `V`.
"""
function IgaBase.dimension(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}
    @assert hasfield(T, field)
    V = getfield(V, field)
    @assert isa(V, VectorFunctionSpace)
    dimension(V, i)
end

"""
    IgaBase.dimension(V::T, field::Symbol) where {T<:MixedFunctionSpace}

Return the dimension of a scalar function space `field` in a
mixed space `V`.
"""
function IgaBase.dimension(V::T, field::Symbol) where {T<:MixedFunctionSpace}
    @assert hasfield(T, field)
    dimension(getfield(V, field))
end

"""
    IgaBase.dimension(V::T) where {T<:MixedFunctionSpace}

Return the dimension of a mixed function space.
"""
function IgaBase.dimension(V::T) where {T<:MixedFunctionSpace}
    fields = propertynames(V)
    dims = map(field -> dimension(getfield(V, field)), fields)
    sum(dims)
end

"""
    dimensions(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}

Return the dimension in each tensor-product direction of
the `i`th component of a vector function space `field` in
the mixed space `V` as a tuple.
"""
function dimensions(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}
    @assert hasfield(T, field)
    V = getfield(V, field)
    @assert isa(V, VectorFunctionSpace)
    dimensions(V, i)
end

"""
    dimensions(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}

Return the dimension in each tensor-product direction of the vector
function space `field` in the mixed space `V` as a tuple.
"""
function dimensions(V::T, field::Symbol) where {T<:MixedFunctionSpace}
    @assert hasfield(T, field)
    dimensions(getfield(V, field))
end

"""
    dimensions(V::T) where {T<:MixedFunctionSpace}

Return the dimension in each tensor-product direction of a
mixed function space as a tuple.
"""
function dimensions(V::T) where {T<:MixedFunctionSpace}
    fields = propertynames(V)
    map(field -> dimensions(getfield(V, field)), fields)
end

"""
    indices(space::S, field::Symbol, i::Int) where {S<:MixedFunctionSpace}

Return linear indices for `i`th component of a vector function space `field`
in mixed function space `space`.
"""
function indices(space::S, field::Symbol, i::Int) where {S<:MixedFunctionSpace}
    prev = (indices(space, field)).start - 1
    curr = getfield(space, field)
    inds = indices(curr, i) .+ prev
end

"""
    indices(space::S, field::Symbol) where {S<:MixedFunctionSpace}

Return linear indices for a function space `field` in
mixed function space `space`.
"""
function indices(space::S, field::Symbol) where {S<:MixedFunctionSpace}
    prev = 0
    for name in propertynames(space)
        curr = getfield(space, name)
        inds = indices(curr)
        (name == field) && return indices(curr) .+ prev
        prev += lastindex(inds)
    end
end

"""
    extraction_operators(S::T, field::Symbol; sparse::Bool=false) where {T<:MixedFunctionSpace}

Return the extraction operators for a vector function space `field` in 
mixed function space `S` in a tuple.

If `sparse` is `true`, return a tuple of sparse matrices.
"""
function extraction_operators(S::T, field::Symbol; sparse::Bool=false) where {T<:MixedFunctionSpace}
    return extraction_operators(getfield(S, field); sparse=sparse)
end

"""
    extraction_operator(S::T, field::Symbol; sparse::Bool=false) where {T<:MixedFunctionSpace}

Return the extraction operator for a scalar function space `field` in 
mixed function space `S`.

If `sparse` is `true`, return a sparse matrix.
"""
function extraction_operator(S::T, field::Symbol; sparse::Bool=false) where {T<:MixedFunctionSpace}
    return extraction_operator(getfield(S, field); sparse=sparse)
end

function Base.show(io::IO, V::S) where {Dim,T,S<:MixedFunctionSpace{Dim,T}}
    fields = propertynames(V)
    print(io, "$S with fields $fields")
end