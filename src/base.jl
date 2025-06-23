"""
    const Partition{Dim,T} = CartesianProduct{Dim,Tuple{Vararg{T,Dim}},Tuple{Vararg{IncreasingRange{T},Dim}}}

Alias for Cartesian product defining a partition.
"""
const Partition{Dim,T} = CartesianProduct{Dim,Tuple{Vararg{T,Dim}},Tuple{Vararg{S,Dim}}} where {S<:IncreasingSequence{T}}

"""
    const Domain{Dim,T} = CartesianProduct{Dim,Tuple{Vararg{T,Dim}},Tuple{Vararg{Interval{T},Dim}}}

Alias for Cartesian product of intervals defining a domain.
"""
const Domain{Dim,T} = CartesianProduct{Dim,Tuple{Vararg{T,Dim}},Tuple{Vararg{Interval{T},Dim}}}

"""
Type alias for a tensor-product splines space
"""
const ScalarSplineSpace{Dim,T} = TensorProduct{Dim,SplineSpace{T}}
Base.eltype(::Type{TensorProduct{Dim,SplineSpace{T}}}) where {Dim,T} = SplineSpace{T}

"""
    Partition(S::ScalarSplineSpace{Dimt,T})

Returns the partition corresponding to a scalar spline space.
"""
function Partition(S::ScalarSplineSpace{Dim,T}) where {Dim,T<:Real}
    CartesianProduct(s -> breakpoints(s), S)
end

"""
    Partition(domain::S, n::NTuple{Dim, Int64}) where {Dim,T<:Real,S<:Domain{Dim,T}}

Returns an uniform partition of `domain` with `nₖ` breakpoints in `k`th dimension.
"""
function Partition(domain::S, n::NTuple{Dim,Int64}) where {Dim,T<:Real,S<:Domain{Dim,T}}
    @assert all(n .> 1) "number of breakpoints in each dimension must be larger than one"
    CartesianProduct((d, n) -> IncreasingRange(d, n), domain, n)
end