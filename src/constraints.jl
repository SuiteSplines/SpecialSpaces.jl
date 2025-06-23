struct UnivariateSplineSpaceConstraints{T}
    left::Vector{T}
    right::Vector{T}
    periodic::Vector{T}
    function UnivariateSplineSpaceConstraints{T}() where {T<:Integer}
        new{T}(T[], T[], T[])
    end
end
UnivariateSplineSpaceConstraints() = UnivariateSplineSpaceConstraints{Int8}()


struct ScalarSplineSpaceConstraints{Dim,T}
    data::NTuple{Dim,UnivariateSplineSpaceConstraints{T}}
    function ScalarSplineSpaceConstraints{Dim,T}() where {Dim,T<:Integer}
        new{Dim,T}(ntuple(dim -> UnivariateSplineSpaceConstraints{T}(), Dim))
    end
end
ScalarSplineSpaceConstraints{Dim}() where {Dim} = ScalarSplineSpaceConstraints{Dim,Int8}()

Base.length(C::ScalarSplineSpaceConstraints{Dim}) where {Dim} = Dim
Base.getindex(C::ScalarSplineSpaceConstraints, i::T) where {T<:Integer} = getindex(C.data, i)
Base.iterate(C::ScalarSplineSpaceConstraints) = iterate(C.data)
Base.iterate(C::ScalarSplineSpaceConstraints, i::T) where {T<:Integer} = iterate(C.data, i)
Base.eltype(::Type{S}) where {T<:Integer,S<:ScalarSplineSpaceConstraints{<:Any,T}} = UnivariateSplineSpaceConstraints{T}

struct VectorSplineSpaceConstraints{Dim,Codim,T}
    data::NTuple{Codim,ScalarSplineSpaceConstraints{Dim,T}}
    function VectorSplineSpaceConstraints{Dim,Codim,T}() where {Dim,Codim,T<:Integer}
        new{Dim,Codim,T}(ntuple(codim -> ScalarSplineSpaceConstraints{Dim,T}(), Codim))
    end
    function VectorSplineSpaceConstraints{T}(args::Vararg{ScalarSplineSpaceConstraints{Dim},Codim}) where {Dim,Codim,T}
        new{Dim,Codim,T}(Tuple(args))
    end
end
VectorSplineSpaceConstraints{Dim}() where {Dim} = VectorSplineSpaceConstraints{Dim,Dim,Int8}()
VectorSplineSpaceConstraints{Dim,Codim}() where {Dim,Codim} = VectorSplineSpaceConstraints{Dim,Codim,Int8}()
VectorSplineSpaceConstraints(args::Vararg{ScalarSplineSpaceConstraints{Dim},Codim}) where {Dim,Codim} = VectorSplineSpaceConstraints{Int8}(args...)

Base.length(C::VectorSplineSpaceConstraints{<:Any,Codim}) where {Codim} = Codim
Base.getindex(C::VectorSplineSpaceConstraints, i::T) where {T<:Integer} = getindex(C.data, i)
Base.iterate(C::VectorSplineSpaceConstraints) = iterate(C.data)
Base.iterate(C::VectorSplineSpaceConstraints, i::T) where {T<:Integer} = iterate(C.data, i)
Base.eltype(::Type{S}) where {Dim,T<:Integer,S<:VectorSplineSpaceConstraints{Dim,<:Any,T}} = ScalarSplineSpaceConstraints{Dim,T}

const MixedSplineSpaceConstraints = NamedTuple

function left_constraint!(C::ScalarSplineSpaceConstraints; c::Vector{Int}=Int[1], dim::Int)
    push!(C[dim].left, c...)
    return C
end

function right_constraint!(C::ScalarSplineSpaceConstraints; c::Vector{Int}=Int[1], dim::Int)
    push!(C[dim].right, c...)
    return C
end

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
