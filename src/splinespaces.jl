"""
    ScalarSplineSpace(S::ScalarSplineSpace{Dim,T}) where {Dim,T<:Real}
    ScalarSplineSpace(p::Degree, partition::Partition{Dim,T}) where {Dim,T<:Real}
    ScalarSplineSpace(degrees::NTuple{Dim,Degree}, partition::Partition{Dim,T}) where {Dim,T<:Real}
    ScalarSplineSpace(degrees::NTuple{Dim,Degree}, partition::Partition{Dim,T}, C::ScalarSplineSpaceConstraints{Dim}) where {Dim,T<:Real}
    ScalarSplineSpace(S::ScalarSplineSpace{Dim,T}, C::ScalarSplineSpaceConstraints{Dim}) where {Dim,T<:Real}

Construct a [`ScalarSplineSpace`](@ref).

If constructed from another [`ScalarSplineSpace`](@ref),
discard constraint conditions.
"""
function ScalarSplineSpace(degrees::NTuple{Dim,Degree}, partition::Partition{Dim,T}, C::ScalarSplineSpaceConstraints{Dim}) where {Dim,T<:Real}
    TensorProduct((p, Δ, c) -> SplineSpace(p, Δ; cperiodic=c.periodic, cleft=c.left, cright=c.right), degrees, partition.data, C)
end

function ScalarSplineSpace(S::ScalarSplineSpace{Dim,T}, C::ScalarSplineSpaceConstraints{Dim}) where {Dim,T<:Real}
    TensorProduct((s, c) -> SplineSpace(s.p, s.U; cperiodic=c.periodic, cleft=c.left, cright=c.right), S, C)
end

function ScalarSplineSpace(degrees::NTuple{Dim,Degree}, partition::Partition{Dim,T}) where {Dim,T<:Real}
    TensorProduct((p, Δ) -> SplineSpace(p, Δ), degrees, partition.data)
end

function ScalarSplineSpace(p::Degree, partition::Partition{Dim,T}) where {Dim,T<:Real}
    TensorProduct(Δ -> SplineSpace(p, Δ),partition.data)
end

function ScalarSplineSpace(S::ScalarSplineSpace{Dim,T}) where {Dim,T<:Real}
    TensorProduct(s -> SplineSpace(s.p, s.U), S)
end

"""
    dimensions(V::ScalarSplineSpace)

Return the dimension of a [`ScalarSplineSpace`](@ref) in each
tensor-product direction as a tuple.
"""
function dimensions(V::ScalarSplineSpace)
    ntuple(i -> dimsplinespace(V[i]), Val(length(V)))
end

"""
    dimensions(V::ScalarSplineSpace)

Return the dimension of a [`ScalarSplineSpace`](@ref).
"""
function IgaBase.dimension(V::ScalarSplineSpace)
    prod(dimensions(V))
end

"""
    indices(V::ScalarSplineSpace)

Return linear indices for a scalar function space.
"""
function indices(V::ScalarSplineSpace)
    lastind = dimension(V)
    Base.UnitRange(1, lastind)
end

"""
    ScalarSplineSpaceConstraints(::F) where {Dim,F<:ScalarSplineSpace{Dim}}

Construct a constraints container for a [`ScalarSplineSpace`](@ref).
"""
function ScalarSplineSpaceConstraints(::F) where {Dim,F<:ScalarSplineSpace{Dim}}
    ScalarSplineSpaceConstraints{Dim,Int8}()
end

"""
    extraction_operator(S::ScalarSplineSpace; sparse::Bool=false)

Return the extraction operator for a scalar function space.

If `sparse` is `true`, return a sparse matrix.
"""
function extraction_operator(S::ScalarSplineSpace; sparse::Bool=false)
    extraction_operator(S, Val(sparse))
end

function extraction_operator(S::ScalarSplineSpace, ::Val{false})
    return KroneckerProduct(d -> d.C, S.data; reverse=true)
end

function extraction_operator(S::ScalarSplineSpace, ::Val{true})
    return kron(reverse(ntuple(d -> S[d].C, length(S)))...)
end

struct VectorSplineSpace{Dim,Codim,T} <: VectorFunctionSpace{Dim,T}
    V::NTuple{Codim,ScalarSplineSpace{Dim,T}}
    function VectorSplineSpace(V::Vararg{ScalarSplineSpace{Dim,T},Codim}) where {Dim,Codim,T}
        @assert all(ntuple(k -> Partition(V[k]) == Partition(V[1]), Codim)) "all space components must be defined on the same partition"
        new{Dim,Codim,T}(V)
    end
    function VectorSplineSpace(V::NTuple{Codim,ScalarSplineSpace{Dim,T}}) where {Dim,Codim,T}
        @assert all(ntuple(k -> Partition(V[k]) == Partition(V[1]), Codim)) "all space components must be defined on the same partition"
        new{Dim,Codim,T}(V)
    end
    function VectorSplineSpace(S::ScalarSplineSpace{Dim,T}) where {Dim,T<:Real}
        new{Dim,Dim,T}(ntuple(dim -> S, Dim))
    end
end
Base.eltype(::Type{V}) where {Dim,T,V<:VectorSplineSpace{Dim,<:Any,T}} = ScalarSplineSpace{Dim,T}

"""
    VectorSplineSpace(degree::Degree, partition::Partition{Dim,T}) where {Dim,T<:Real}
    VectorSplineSpace(degrees::NTuple{Dim,Degree}, partition::Partition{Dim,T}) where {Dim,T<:Real}
    VectorSplineSpace(V::VectorSplineSpace{Dim,Codim,T}, C::VectorSplineSpaceConstraints{Dim,Codim}) where {Dim,Codim,T<:Real}
    VectorSplineSpace(S::ScalarSplineSpace{Dim,T}) where {Dim,T<:Real}
    VectorSplineSpace(V::NTuple{Codim,ScalarSplineSpace{Dim,T}}) where {Dim,Codim,T}
    VectorSplineSpace(V::Vararg{ScalarSplineSpace{Dim,T},Codim}) where {Dim,Codim,T}

Construct a [`VectorSplineSpace`](@ref).

Constraints on scalar spline space arguments are preserved.
"""
function VectorSplineSpace(V::VectorSplineSpace{Dim,Codim,T}, C::VectorSplineSpaceConstraints{Dim,Codim}) where {Dim,Codim,T<:Real}
    VectorSplineSpace(ntuple(d -> ScalarSplineSpace(V[d], C[d]), Codim))
end

function VectorSplineSpace(degrees::NTuple{Dim,Degree}, partition::Partition{Dim,T}) where {Dim,T<:Real}
    VectorSplineSpace(ntuple(d -> ScalarSplineSpace(degrees, partition), Dim))
end

function VectorSplineSpace(degree::Degree, partition::Partition{Dim,T}) where {Dim,T<:Real}
    degrees = ntuple(dim -> degree, Dim)
    VectorSplineSpace(ntuple(d -> ScalarSplineSpace(degrees, partition), Dim))
end

"""
    VectorSplineSpaceConstraints(::F) where {Dim,Codim,F<:VectorSplineSpace{Dim,Codim}}

Construct a constraints container for a [`VectorSplineSpace`](@ref).
"""
function VectorSplineSpaceConstraints(::F) where {Dim,Codim,F<:VectorSplineSpace{Dim,Codim}}
    VectorSplineSpaceConstraints{Dim,Codim}()
end

"""
    abstract type MixedSplineSpace{Dim,T} <: MixedFunctionSpace{Dim,T}

Concrete mixed spaces like [`RaviartThomas`](@ref) subtype this.
"""
abstract type MixedSplineSpace{Dim,T} <: MixedFunctionSpace{Dim,T} end


"""
    struct RaviartThomas{Dim,T} <: MixedSplineSpace{Dim,T}

Structure preserving pair of spline spaces for velocities
and pressure in two and three dimensions.
"""
struct RaviartThomas{Dim,T} <: MixedSplineSpace{Dim,T}
    V::VectorSplineSpace{Dim,Dim,T}
    Q::ScalarSplineSpace{Dim,T}
    function RaviartThomas(p::Degree, Δ::Partition{2,T}, C::MixedSplineSpaceConstraints{(:V,:Q)}) where {T<:Real}
        V₁ = ScalarSplineSpace((p, p - 1), Δ, C.V[1])
        V₂ = ScalarSplineSpace((p - 1, p), Δ, C.V[2])
        V = VectorSplineSpace(V₁, V₂)
        Q = ScalarSplineSpace((p - 1, p - 1), Δ, C.Q)
        new{2,T}(V, Q)
    end
    function RaviartThomas(p::Degree, Δ::Partition{3,T}, C::MixedSplineSpaceConstraints{(:V,:Q)}) where {T<:Real}
        V₁ = ScalarSplineSpace((p, p - 1, p - 1), Δ, C.V[1])
        V₂ = ScalarSplineSpace((p - 1, p, p - 1), Δ, C.V[2])
        V₃ = ScalarSplineSpace((p - 1, p - 1, p), Δ, C.V[3])
        V = VectorSplineSpace(V₁, V₂, V₃)
        Q = ScalarSplineSpace((p - 1, p - 1, p - 1), Δ, C.Q)
        new{3,T}(V, Q)
    end
end

"""
    RaviartThomas(p::Degree, Δ::Partition{Dim,T}) where {Dim,T<:Real}
    RaviartThomas(p::Degree, Δ::Partition{2,T}, C::MixedSplineSpaceConstraints{(:V,:Q)})
    RaviartThomas(p::Degree, Δ::Partition{3,T}, C::MixedSplineSpaceConstraints{(:V,:Q)}) where {T<:Real}

Construct a [`RaviartThomas`](@ref) mixed spline space.
"""
function RaviartThomas(p::Degree, Δ::Partition{Dim,T}) where {Dim,T<:Real}
    C = MixedSplineSpaceConstraints{(:V,:Q)}((VectorSplineSpaceConstraints{Dim}(), ScalarSplineSpaceConstraints{Dim}()))
    RaviartThomas(p, Δ, C)
end

"""
    MixedSplineSpaceConstraints(S::RaviartThomas{Dim}) where {Dim}

Construct a constraints container for [`RaviartThomas`](@ref).
"""
function MixedSplineSpaceConstraints(S::RaviartThomas{Dim}) where {Dim}
    MixedSplineSpaceConstraints{(:V, :Q)}((VectorSplineSpaceConstraints(S.V), ScalarSplineSpaceConstraints(S.Q)))
end

"""
    struct TaylorHood{Dim,T} <: MixedSplineSpace{Dim,T}

An inf-sup stable pair of spline spaces for velocities and pressure
in two and three dimensions.
"""
struct TaylorHood{Dim,T} <: MixedSplineSpace{Dim,T}
    V::VectorSplineSpace{Dim,Dim,T}
    Q::ScalarSplineSpace{Dim,T}
    
    @doc """
        TaylorHood(p::Degree, Δ::Partition{Dim,T}) where {Dim,T<:Real}
        TaylorHood(p::Degree, Δ::Partition{Dim,T}, C::MixedSplineSpaceConstraints{(:V,:Q)}) where {Dim,T<:Real}

    Construct a [`TaylorHood`](@ref) mixed spline space.
    """
    function TaylorHood(p::Degree, Δ::Partition{Dim,T}, C::MixedSplineSpaceConstraints{(:V,:Q)}) where {Dim,T<:Real}
        @assert p ≥ 2
        p = ntuple(i -> p, Dim)
        V = VectorSplineSpace(ntuple(i -> ScalarSplineSpace(p, Δ, C.V[i]), Dim)...)
        Q = ScalarSplineSpace(p .- 1, Δ, C.Q)
        new{Dim,T}(V, Q)
    end
    function TaylorHood(p::Degree, Δ::Partition{Dim,T}) where {Dim,T<:Real}
        @assert p ≥ 2
        p = ntuple(i -> p, Dim)
        V = VectorSplineSpace(ntuple(i -> ScalarSplineSpace(p, Δ), Dim)...)
        Q = ScalarSplineSpace(p .- 1, Δ)
        new{Dim,T}(V, Q)
    end
end

"""
    MixedSplineSpaceConstraints(S::RaviartThomas{Dim}) where {Dim}

Construct a constraints container for [`TaylorHood`](@ref).
"""
function MixedSplineSpaceConstraints(S::TaylorHood{Dim}) where {Dim}
    MixedSplineSpaceConstraints{(:V, :Q)}((VectorSplineSpaceConstraints(S.V), ScalarSplineSpaceConstraints(S.Q)))
end

function TensorProductBsplines.TensorProductBspline(space::VectorSplineSpace{Dim,Codim}) where {Dim,Codim}
    ntuple(d -> TensorProductBspline(space[d]), Codim)
end
