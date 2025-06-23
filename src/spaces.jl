# vector function spaces
abstract type VectorFunctionSpace{Dim,T} end

Base.parent(V::S) where {S<:VectorFunctionSpace} = getfield(V, :V)
Base.length(V::S) where {S<:VectorFunctionSpace} = length(parent(V))
Base.getindex(V::S, i::Int64) where {S<:VectorFunctionSpace} = getindex(parent(V), i)
Base.iterate(V::S) where {S<:VectorFunctionSpace} = iterate(parent(V))
Base.iterate(V::S, i::Int) where {S<:VectorFunctionSpace} = iterate(parent(V), i)

function indices(space::S) where {S<:VectorFunctionSpace}
    lastind = dimension(space)
    Base.UnitRange(1, lastind)
end

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

function IgaBase.dimension(V::T, k::Int) where {T<:VectorFunctionSpace}
    dimension(V[k])
end

function IgaBase.dimension(V::T) where {T<:VectorFunctionSpace}
    sum(map(dimension, V))
end

function dimensions(V::T, i::Int64) where {T<:VectorFunctionSpace}
    dimensions(V[i])
end

function dimensions(V::T) where {T<:VectorFunctionSpace}
    ntuple(i -> dimensions(V[i]), Val(length(V)))
end

function extraction_operators(S::T; sparse::Bool=false) where {T<:VectorFunctionSpace}
    return ntuple(k -> extraction_operator(S[k]; sparse=sparse), length(S))
end

function Base.show(io::IO, V::S) where {Dim,T,S<:VectorFunctionSpace{Dim,T}}
    ncomp = length(V)
    print(io, "$S with $ncomp components")
end



# mixed function spaces
abstract type MixedFunctionSpace{Dim,T} end

function IgaBase.dimension(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}
    @assert hasfield(T, field)
    V = getfield(V, field)
    @assert isa(V, VectorFunctionSpace)
    dimension(V, i)
end

function IgaBase.dimension(V::T, field::Symbol) where {T<:MixedFunctionSpace}
    @assert hasfield(T, field)
    dimension(getfield(V, field))
end

function IgaBase.dimension(V::T) where {T<:MixedFunctionSpace}
    fields = propertynames(V)
    dims = map(field -> dimension(getfield(V, field)), fields)
    sum(dims)
end

function dimensions(V::T, field::Symbol, i::Int64) where {T<:MixedFunctionSpace}
    @assert hasfield(T, field)
    V = getfield(V, field)
    @assert isa(V, VectorFunctionSpace)
    dimensions(V, i)
end

function dimensions(V::T, field::Symbol) where {T<:MixedFunctionSpace}
    @assert hasfield(T, field)
    dimensions(getfield(V, field))
end

function dimensions(V::T) where {T<:MixedFunctionSpace}
    fields = propertynames(V)
    map(field -> dimensions(getfield(V, field)), fields)
end

function indices(space::S, field::Symbol, i::Int) where {S<:MixedFunctionSpace}
    prev = (indices(space, field)).start - 1
    curr = getfield(space, field)
    inds = indices(curr, i) .+ prev
end

function indices(space::S, field::Symbol) where {S<:MixedFunctionSpace}
    prev = 0
    for name in propertynames(space)
        curr = getfield(space, name)
        inds = indices(curr)
        (name == field) && return indices(curr) .+ prev
        prev += lastindex(inds)
    end
end

function extraction_operators(S::T, field::Symbol; sparse::Bool=false) where {T<:MixedFunctionSpace}
    return extraction_operators(getfield(S, field); sparse=sparse)
end

function extraction_operator(S::T, field::Symbol; sparse::Bool=false) where {T<:MixedFunctionSpace}
    return extraction_operator(getfield(S, field); sparse=sparse)
end

function Base.show(io::IO, V::S) where {Dim,T,S<:MixedFunctionSpace{Dim,T}}
    fields = propertynames(V)
    print(io, "$S with fields $fields")
end