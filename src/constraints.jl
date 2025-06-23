"""
    struct UnivariateSplineSpaceConstraints{T}

Container for [`SplineSpace`](@ref) constraints, i.e. `cl`, `cr`, `cp`
vectors in `SplineSpace` constructor stored as `left`, `right`and `periodic` field.
"""
struct UnivariateSplineSpaceConstraints{T}
    left::Vector{T}
    right::Vector{T}
    periodic::Vector{T}
    function UnivariateSplineSpaceConstraints{T}() where {T<:Integer}
        new{T}(T[], T[], T[])
    end
end

"""
    UnivariateSplineSpaceConstraints()

Construct default [`UnivariateSplineSpaceConstraints`](@ref)
"""
UnivariateSplineSpaceConstraints() = UnivariateSplineSpaceConstraints{Int8}()

"""
    struct ScalarSplineSpaceConstraints{Dim,T}

Container for [`ScalarSplineSpace`](@ref) constraints.

Field `data` stores a tuple of [`UnivariateSplineSpaceConstraints`](@ref)
for each tensor-product dimension of [`ScalarSplineSpace`](@ref).
"""
struct ScalarSplineSpaceConstraints{Dim,T}
    data::NTuple{Dim,UnivariateSplineSpaceConstraints{T}}
    function ScalarSplineSpaceConstraints{Dim,T}() where {Dim,T<:Integer}
        new{Dim,T}(ntuple(dim -> UnivariateSplineSpaceConstraints{T}(), Dim))
    end
end

"""
    ScalarSplineSpaceConstraints{Dim}() where {Dim}
    ScalarSplineSpaceConstraints{Dim,T}() where {Dim,T<:Integer}

Construct default [`ScalarSplineSpaceConstraints`](@ref) for a
[`ScalarSplineSpace`](@ref) of dimension `Dim`.
"""
ScalarSplineSpaceConstraints{Dim}() where {Dim} = ScalarSplineSpaceConstraints{Dim,Int8}()

"""
    Base.length(C::ScalarSplineSpaceConstraints{Dim}) where {Dim}

Return number of dimensions `Dim`.
"""
Base.length(C::ScalarSplineSpaceConstraints{Dim}) where {Dim} = Dim

"""
    Base.getindex(C::ScalarSplineSpaceConstraints, i::T) where {T<:Integer}

Return `i`th component constraints in a [`ScalarSplineSpaceConstraints`](@ref)
"""
Base.getindex(C::ScalarSplineSpaceConstraints, i::T) where {T<:Integer} = getindex(C.data, i)

"""
    Base.iterate(C::ScalarSplineSpaceConstraints)
    Base.iterate(C::ScalarSplineSpaceConstraints, i::T) where {T<:Integer} = iterate(C.data, i)

Iterate of the collection of [`UnivariateSplineSpaceConstraints`](@ref) stored
in [`ScalarSplineSpaceConstraints`](@ref).
"""
Base.iterate(C::ScalarSplineSpaceConstraints) = iterate(C.data)
Base.iterate(C::ScalarSplineSpaceConstraints, i::T) where {T<:Integer} = iterate(C.data, i)

"""
    Base.eltype(::Type{S}) where {T<:Integer,S<:ScalarSplineSpaceConstraints{<:Any,T}}

Return element type of [`ScalarSplineSpaceConstraints`](@ref) collection.
"""
Base.eltype(::Type{S}) where {T<:Integer,S<:ScalarSplineSpaceConstraints{<:Any,T}} = UnivariateSplineSpaceConstraints{T}


"""
    struct ScalarSplineSpaceConstraints{Dim,T}

Container for [`ScalarSplineSpaceConstraints`](@ref).

Field `data` stores a tuple of [`ScalarSplineSpaceConstraints`](@ref)
for each component of [`VectorSplineSpace`](@ref).
"""
struct VectorSplineSpaceConstraints{Dim,Codim,T}
    data::NTuple{Codim,ScalarSplineSpaceConstraints{Dim,T}}
    function VectorSplineSpaceConstraints{Dim,Codim,T}() where {Dim,Codim,T<:Integer}
        new{Dim,Codim,T}(ntuple(codim -> ScalarSplineSpaceConstraints{Dim,T}(), Codim))
    end
    function VectorSplineSpaceConstraints{T}(args::Vararg{ScalarSplineSpaceConstraints{Dim},Codim}) where {Dim,Codim,T}
        new{Dim,Codim,T}(Tuple(args))
    end
end

"""
    VectorSplineSpaceConstraints{Dim}() where {Dim}
    VectorSplineSpaceConstraints{Dim,Codim}() where {Dim,Codim}
    VectorSplineSpaceConstraints(args::Vararg{ScalarSplineSpaceConstraints{Dim},Codim}) where {Dim,Codim}
    VectorSplineSpaceConstraints{T}(args::Vararg{ScalarSplineSpaceConstraints{Dim},Codim}) where {Dim,Codim,T}
    VectorSplineSpaceConstraints{Dim,Codim,T}() where {Dim,Codim,T<:Integer}

Construct [`VectorSplineSpaceConstraints`](@ref) for a
[`VectorSplineSpace`](@ref) of dimension `Dim` and codomain dimension `Codim`.
"""
VectorSplineSpaceConstraints{Dim}() where {Dim} = VectorSplineSpaceConstraints{Dim,Dim,Int8}()
VectorSplineSpaceConstraints{Dim,Codim}() where {Dim,Codim} = VectorSplineSpaceConstraints{Dim,Codim,Int8}()
VectorSplineSpaceConstraints(args::Vararg{ScalarSplineSpaceConstraints{Dim},Codim}) where {Dim,Codim} = VectorSplineSpaceConstraints{Int8}(args...)

"""
    Base.length(C::VectorSplineSpaceConstraints{<:Any,Codim}) where {Codim}

Return number of dimensions `Dim`.
"""
Base.length(C::VectorSplineSpaceConstraints{<:Any,Codim}) where {Codim} = Codim

"""
    Base.getindex(C::ScalarSplineSpaceConstraints, i::T) where {T<:Integer}

Return `i`th component constraints in a [`VectorSplineSpaceConstraints`](@ref)
"""
Base.getindex(C::VectorSplineSpaceConstraints, i::T) where {T<:Integer} = getindex(C.data, i)

"""
    Base.iterate(C::VectorSplineSpaceConstraints)
    Base.iterate(C::VectorSplineSpaceConstraints, i::T) where {T<:Integer} = iterate(C.data, i)

Iterate of the collection of [`ScalarSplineSpaceConstraints`](@ref) stored
in [`VectorSplineSpaceConstraints`](@ref).
"""
Base.iterate(C::VectorSplineSpaceConstraints) = iterate(C.data)
Base.iterate(C::VectorSplineSpaceConstraints, i::T) where {T<:Integer} = iterate(C.data, i)

"""
    Base.eltype(::Type{S}) where {Dim,T<:Integer,S<:VectorSplineSpaceConstraints{Dim,<:Any,T}}

Return element type of [`VectorSplineSpaceConstraints`](@ref) collection.
"""
Base.eltype(::Type{S}) where {Dim,T<:Integer,S<:VectorSplineSpaceConstraints{Dim,<:Any,T}} = ScalarSplineSpaceConstraints{Dim,T}

"""
    const MixedSplineSpaceConstraints

Type alias for `NamedTuple` which serves as a container for mixed space constraints.
"""
const MixedSplineSpaceConstraints = NamedTuple

"""
    left_constraint!(C::ScalarSplineSpaceConstraints; c::Vector{Int}=Int[1], dim::Int)

Push `c` to vector in `left` field of [`UnivariateSplineSpaceConstraints`](@ref)
stored at index `dim` in [`ScalarSplineSpaceConstraints`](@ref).
"""
function left_constraint!(C::ScalarSplineSpaceConstraints; c::Vector{Int}=Int[1], dim::Int)
    push!(C[dim].left, c...)
    return C
end

"""
    right_constraint!(C::ScalarSplineSpaceConstraints; c::Vector{Int}=Int[1], dim::Int)

Push `c` to vector in `right` field of [`UnivariateSplineSpaceConstraints`](@ref)
stored at index `dim` in [`ScalarSplineSpaceConstraints`](@ref).
"""
function right_constraint!(C::ScalarSplineSpaceConstraints; c::Vector{Int}=Int[1], dim::Int)
    push!(C[dim].right, c...)
    return C
end

"""
    periodic_constraint!(C::ScalarSplineSpaceConstraints; c::Vector{Int}, dim::Int)

Push `c` to vector in `periodic` field of [`UnivariateSplineSpaceConstraints`](@ref)
stored at index `dim` in [`ScalarSplineSpaceConstraints`](@ref).
"""
function periodic_constraint!(C::ScalarSplineSpaceConstraints; c::Vector{Int}, dim::Int)
    push!(C[dim].periodic, c...)
    return C
end

function Base.show(io::IO, ::C) where {T<:Integer,C<:UnivariateSplineSpaceConstraints{T}}
    print(io, C)
end

function Base.show(io::IO, ::C) where {Dim,T<:Integer,C<:ScalarSplineSpaceConstraints{Dim,T}}
    print(io, C)
end

function Base.show(io::IO, ::C) where {Dim,Codim,T<:Integer,C<:VectorSplineSpaceConstraints{Dim,Codim,T}}
    print(io, C)
end
