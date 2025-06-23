using CartesianProducts, SortedSequences, UnivariateSplines, KroneckerProducts
using LinearAlgebra

Ω¹ = Interval(0.0, 1.0)
Ω² = Ω¹ ⨱ Ω¹
Ω³ = Ω¹ ⨱ Ω¹ ⨱ Ω¹

@testset "Vector spaces dimensions and indices" begin
    T = eltype(Ω¹)
    Δ = Partition(Ω², (4, 7))
    p₁, p₂ = 3, 4

    # init scalar spaces from degrees and partition
    V₁ = ScalarSplineSpace((p₁, p₂), Δ)
    V₂ = ScalarSplineSpace((p₂, p₁), Δ)

    # get scalar spaces dimension
    dim₁ = dimension(V₁)
    dim₂ = dimension(V₂)

    # get scalar spaces dimensions (per tensor-product dim)
    dims₁ = dimensions(V₁)
    dims₂ = dimensions(V₂)

    # init vector space
    V = VectorSplineSpace(V₁, V₂)

    # test Cartesian dimensions
    @test dimensions(V) == (dims₁, dims₂)
    @test dimensions(V, 1) == dims₁
    @test dimensions(V, 2) == dims₂

    # test tensor-product linear indices
    inds = indices(V)
    @test typeof(inds) <: UnitRange 
    @test inds == 1:(dim₁ + dim₂)

    # test component tensor-product linear indices
    @test indices(V, 1) == 1:dim₁
    @test indices(V, 2) == (dim₁+1):(dim₁+dim₂)
end

@testset "Mixed spaces dimensions and indices" begin
    p, Δ = 2, Partition(Ω², (10, 20))
    th = TaylorHood(p, Δ)

    @test dimension(th, :V) == dimension(th, :V, 1) + dimension(th, :V, 2)
    @test dimensions(th, :V) == (dimensions(th, :V, 1), dimensions(th, :V, 2))
    @test indices(th, :V) == 1:dimension(th, :V)
    @test indices(th, :V, 1) == 1:dimension(th, :V, 1)
    @test indices(th, :V, 2) == (dimension(th, :V, 1)+1):dimension(th, :V)
    @test indices(th, :Q) == (dimension(th, :V)+1):dimension(th)
end