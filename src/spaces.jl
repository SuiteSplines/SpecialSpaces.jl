"""
    abstract type FunctionSpace{Dim,Codim,T}

Concrete function spaces subtype this.

Parameters
==========

- `Dim`: dimension of the domain
- `Codim`: dimension of the codomain
- `T`: data type used for numbers (returned by `IgaBase.numbertype`)
"""
abstract type FunctionSpace{Dim,Codim,T} end

"""
    IgaBase.numbertype(::FunctionSpace{<:Any,<:Any,T}) where {T}

Return the data type used by a function space to represent numbers.
"""
IgaBase.numbertype(::FunctionSpace{<:Any,<:Any,T}) where {T} = T

"""
    dimfunspace(::FunctionSpace{Dim}) where {Dim}

Return domain dimension of function space.
"""
dimfunspace(::FunctionSpace{Dim}) where {Dim} = Dim

"""
    codimfunspace(::FunctionSpace{<:Any,Codim})
Return codomain dimension of function space
"""
codimfunspace(::FunctionSpace{<:Any,Codim}) where {Codim} = Codim


# vector function spaces
"""
    abstract type VectorFunctionSpace{Dim,T}

Concrete vector spaces like [`VectorSplineSpace`](@ref) subtype this.

Vector function spaces are expected to collect components
as a tuple in field `:V`.
"""
abstract type VectorFunctionSpace{Dim,Codim,T} <: FunctionSpace{Dim,Codim,T} end

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
abstract type MixedFunctionSpace{Dim,Codim,T} <: FunctionSpace{Dim,Codim,T} end

"""
    IgaBase.numbertype(::MixedFunctionSpace{<:Any,T}) where {T}

Return the data type for numbers used in space.
"""
IgaBase.numbertype(::MixedFunctionSpace{<:Any,<:Any,T}) where {T} = T

"""
    IgaBase.dimension(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}

Return the dimension of the `i`th component of a vector function space `field`
in a mixed space `V`.
"""
function IgaBase.dimension(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}
    @assert hasproperty(V, field)
    V = getproperty(V, field)
    @assert isa(V, VectorFunctionSpace)
    dimension(V, i)
end

"""
    IgaBase.dimension(V::T, field::Symbol) where {T<:MixedFunctionSpace}

Return the dimension of a scalar function space `field` in a
mixed space `V`.
"""
function IgaBase.dimension(V::T, field::Symbol) where {T<:MixedFunctionSpace}
    @assert hasproperty(V, field)
    dimension(getproperty(V, field))
end

"""
    IgaBase.dimension(V::T) where {T<:MixedFunctionSpace}

Return the dimension of a mixed function space.
"""
function IgaBase.dimension(V::T) where {T<:MixedFunctionSpace}
    fields = propertynames(V)
    dims = map(field -> dimension(getproperty(V, field)), fields)
    sum(dims)
end

"""
    dimensions(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}

Return the dimension in each tensor-product direction of
the `i`th component of a vector function space `field` in
the mixed space `V` as a tuple.
"""
function dimensions(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}
    @assert hasproperty(V, field)
    V = getproperty(V, field)
    @assert isa(V, VectorFunctionSpace)
    dimensions(V, i)
end

"""
    dimensions(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}

Return the dimension in each tensor-product direction of the vector
function space `field` in the mixed space `V` as a tuple.
"""
function dimensions(V::T, field::Symbol) where {T<:MixedFunctionSpace}
    @assert hasproperty(V, field)
    dimensions(getproperty(V, field))
end

"""
    dimensions(V::T) where {T<:MixedFunctionSpace}

Return the dimension in each tensor-product direction of a
mixed function space as a tuple.
"""
function dimensions(V::T) where {T<:MixedFunctionSpace}
    fields = propertynames(V)
    map(field -> dimensions(getproperty(V, field)), fields)
end

"""
    indices(space::S, field::Symbol, i::Int) where {S<:MixedFunctionSpace}

Return linear indices for `i`th component of a vector function space `field`
in mixed function space `space`.
"""
function indices(space::S, field::Symbol, i::Int) where {S<:MixedFunctionSpace}
    @assert hasproperty(space, field)
    prev = (indices(space, field)).start - 1
    curr = getproperty(space, field)
    inds = indices(curr, i) .+ prev
end

"""
    indices(space::S, field::Symbol) where {S<:MixedFunctionSpace}

Return linear indices for a function space `field` in
mixed function space `space`.
"""
function indices(space::S, field::Symbol) where {S<:MixedFunctionSpace}
    @assert hasproperty(space, field)
    prev = 0
    for name in propertynames(space)
        curr = getproperty(space, name)
        inds = indices(curr)
        (name == field) && return indices(curr) .+ prev
        prev += lastindex(inds)
    end
end

"""
    indices(space::S) where {S<:MixedFunctionSpace}

Return linear indices for a mixed function space.
"""
function indices(space::S) where {S<:MixedFunctionSpace}
    fields = propertynames(space)
    dims = map(field -> dimension(getproperty(space, field)), fields)
    Base.UnitRange(1, sum(dims))
end

"""
    extraction_operators(S::T, field::Symbol; sparse::Bool=false) where {T<:MixedFunctionSpace}

Return the extraction operators for a vector function space `field` in 
mixed function space `S` in a tuple.

If `sparse` is `true`, return a tuple of sparse matrices.
"""
function extraction_operators(S::T, field::Symbol; sparse::Bool=false) where {T<:MixedFunctionSpace}
    @assert hasproperty(S, field)
    return extraction_operators(getproperty(S, field); sparse=sparse)
end

"""
    extraction_operator(S::T, field::Symbol; sparse::Bool=false) where {T<:MixedFunctionSpace}

Return the extraction operator for a scalar function space `field` in 
mixed function space `S`.

If `sparse` is `true`, return a sparse matrix.
"""
function extraction_operator(S::T, field::Symbol; sparse::Bool=false) where {T<:MixedFunctionSpace}
    @assert hasproperty(S, field)
    return extraction_operator(getproperty(S, field); sparse=sparse)
end

function Base.show(io::IO, V::S) where {Dim,T,S<:MixedFunctionSpace{Dim,T}}
    fields = propertynames(V)
    print(io, "$S with fields $fields")
end