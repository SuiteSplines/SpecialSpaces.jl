using CartesianProducts, SortedSequences, UnivariateSplines, KroneckerProducts
using LinearAlgebra, IgaBase

Ω¹ = Interval(0.0, 1.0)
Ω² = Ω¹ ⨱ Ω¹
Ω³ = Ω¹ ⨱ Ω¹ ⨱ Ω¹

@testset "Scalar spline spaces" begin
    T = eltype(Ω¹)
    Δ = Partition(Ω², (4, 5))
    p₁, p₂ = 3, 4

    # init scalar space from degree and partition
    V = ScalarSplineSpace(p₁, Δ)
    @test typeof(V) <: ScalarSplineSpace{2,T}
    @test numbertype(V) == T
    @test eltype(V) <: SplineSpace{T}
    @test typeof(V[1]) <: SplineSpace{T}
    @test typeof(V[2]) <: SplineSpace{T}
    @test Degree(V[1]) == p₁
    @test Degree(V[2]) == p₁
    @test num_elements(V[1]) == 3
    @test num_elements(V[2]) == 4

    # init scalar space from degrees tuple and partition
    V = ScalarSplineSpace((p₁, p₂), Δ)
    @test typeof(V) <: ScalarSplineSpace{2,T}
    @test eltype(V) <: SplineSpace{T}
    @test typeof(V[1]) <: SplineSpace{T}
    @test typeof(V[2]) <: SplineSpace{T}
    @test Degree(V[1]) == p₁
    @test Degree(V[2]) == p₂
    @test num_elements(V[1]) == 3
    @test num_elements(V[2]) == 4

    # init scalar space with the same degrees and partition as V
    W = ScalarSplineSpace(V)
    @test typeof(W) <: ScalarSplineSpace{2,T}
    @test eltype(W) <: SplineSpace{T}
    @test typeof(W[1]) <: SplineSpace{T}
    @test typeof(W[2]) <: SplineSpace{T}
    @test Degree(W[1]) == p₁
    @test Degree(W[2]) == p₂
    @test num_elements(W[1]) == 3
    @test num_elements(W[2]) == 4

    # get tensor-product linear indices
    nind = dimension(W)
    inds = indices(W)
    @test typeof(inds) <: UnitRange
    @test inds.start == 1
    @test inds.stop == nind

    # init scalar space constraints
    C = ScalarSplineSpaceConstraints(W)
    @test typeof(C) <: ScalarSplineSpaceConstraints{2,<:Any}

    # init scalar spline space with constraints
    left_constraint!(C; dim=1)
    S₁ = ScalarSplineSpace(2, Δ)
    S₂ = ScalarSplineSpace(2, Δ, C)
    dims₁ = dimensions(S₁)
    dims₂ = dimensions(S₂)
    @test dims₂ == dims₁ .- (1, 0)
end

@testset "Vector spline spaces" begin
    T = eltype(Ω¹)
    Δ = Partition(Ω², (4, 5))
    p₁, p₂ = 3, 4

    # init scalar spaces from degrees and partition
    V₁ = ScalarSplineSpace((p₁, p₂), Δ)
    V₂ = ScalarSplineSpace((p₂, p₁), Δ)

    # init vector space from degrees and parition (codim = dim)
    U = VectorSplineSpace((p₁, p₂), Δ)

    # init vector space from two scalar spaces (codim = 2)
    V = VectorSplineSpace(V₁, V₂)

    # init vector space from one scalar space (codim = dim)
    W = VectorSplineSpace(V₁)
    
    # init vector space from degree and parition (codim = dim)
    X = VectorSplineSpace(p₁, Δ)

    # init vector space constraints
    C = VectorSplineSpaceConstraints(X)
    @test typeof(C) <: VectorSplineSpaceConstraints{2,2,<:Any}

    # bulk of tests
    @test typeof(V) <: VectorSplineSpace{2,2,T}
    @test eltype(V) <: ScalarSplineSpace{2,T}
    @test numbertype(V) == T
    @test typeof(V[1]) <: ScalarSplineSpace{2,T}
    @test typeof(V[2]) <: ScalarSplineSpace{2,T}
    @test map(Degree, V[1]) == map(Degree, V₁)
    @test map(Degree, V[2]) == map(Degree, V₂)
    @test map(num_elements, V[1]) == map(num_elements, V₁)
    @test map(num_elements, V[2]) == map(num_elements, V₂)
    @test map(Degree, U[1]) == map(Degree, V₁)
    @test map(Degree, U[2]) == map(Degree, V₁)
    @test map(num_elements, V[1]) == map(num_elements, V₁)
    @test map(num_elements, V[2]) == map(num_elements, V₁)
    @test map(Degree, X[1]) == (p₁, p₁)
    @test map(Degree, X[2]) == (p₁, p₁)
    @test map(num_elements, X[1]) == map(num_elements, V₁)
    @test map(num_elements, X[2]) == map(num_elements, V₁)
    @test Partition(V) == Δ
end


@testset "Scalar and vector space dimensions" begin
    Δ = Partition(Ω², (4, 5))
    p₁, p₂ = 3, 4

    V₁ = ScalarSplineSpace((p₁, p₂), Δ)
    V₂ = ScalarSplineSpace((p₂, p₁), Δ)
    V = VectorSplineSpace(V₁, V₂)

    dims_V₁ = ntuple(i -> dimsplinespace(V₁[i]), Val(length(V₁)))
    dims_V₂ = ntuple(i -> dimsplinespace(V₂[i]), Val(length(V₁)))
    dims_V = (dims_V₁, dims_V₂)

    @test dimensions(V₁) == dims_V₁
    @test dimensions(V₂) == dims_V₂
    @test dimensions(V) == dims_V

    dim_V₁ = prod(ntuple(i -> dimsplinespace(V₁[i]), Val(length(V₁))))
    dim_V₂ = prod(ntuple(i -> dimsplinespace(V₂[i]), Val(length(V₁))))
    dim_V = dim_V₁ + dim_V₂

    @test dimension(V₁) == dim_V₁
    @test dimension(V₂) == dim_V₂
    @test dimension(V) == dim_V
    @test dimension(V, 1) == dim_V₁
    @test dimension(V, 2) == dim_V₂
end

@testset "Raviart-Thomas (mixed) function space dimensions" begin
    Δ = Partition(Ω², (4, 5))
    p = 4

    S = RaviartThomas(p, Δ)

    dim_V = dimension(S.V)
    dim_Q = dimension(S.Q)
    dim_S = dim_V + dim_Q
    @test indices(S) == 1:dim_S

    @test dimension(S, :V) == dim_V
    @test dimension(S, :Q) == dim_Q
    @test dimension(S) == dim_S

    dims_V = dimensions(S.V)
    dims_Q = dimensions(S.Q)
    dims_S = (dims_V, dims_Q)

    @test dimensions(S, :V) == dims_V
    @test dimensions(S, :Q) == dims_Q
    @test dimensions(S) == dims_S
end


@testset "Raviart-Thomas spline spaces" begin
    n1, n2, n3 = 10, 20, 30
    Δ₁ = Partition(Ω², (n1 + 1, n2 + 1))
    Δ₂ = Partition(Ω³, (n1 + 1, n2 + 1, n3 + 1))
    p = 4

    T = eltype(Ω¹)

    V₁ = RaviartThomas(p, Δ₁)
    V₂ = RaviartThomas(p, Δ₂)

    # verify space construction
    @test typeof(V₁) <: RaviartThomas{2,3,T}
    @test typeof(V₂) <: RaviartThomas{3,4,T}
    @test numbertype(V₁) == T
    @test numbertype(V₂) == T
    @test Partition(V₁) == Δ₁
    @test Partition(V₂) == Δ₂

    @test all(map(Degree, V₁.V[1]) .== (p, p - 1))
    @test all(map(Degree, V₁.V[2]) .== (p-1, p))
    @test all(map(Degree, V₁.Q) .== (p-1, p-1))

    @test all(map(Degree, V₂.V[1]) .== (p, p-1, p-1))
    @test all(map(Degree, V₂.V[2]) .== (p-1, p, p-1))
    @test all(map(Degree, V₂.V[3]) .== (p-1, p-1, p))
    @test all(map(Degree, V₂.Q) .== (p-1, p-1, p-1))

    @test all(map(num_elements, V₁.V[1]) .== (n1, n2))
    @test all(map(num_elements, V₁.V[2]) .== (n1, n2))
    @test all(map(num_elements, V₁.Q) .== (n1, n2))

    @test all(map(num_elements, V₂.V[1]) .== (n1, n2, n3))
    @test all(map(num_elements, V₂.V[2]) .== (n1, n2, n3))
    @test all(map(num_elements, V₂.V[3]) .== (n1, n2, n3))
    @test all(map(num_elements, V₂.Q) .== (n1, n2, n3))

    # test mixed space constraints for Raviart-Thomas
    C₁ = MixedSplineSpaceConstraints(V₁)
    C₂ = MixedSplineSpaceConstraints(V₂)
    @test typeof(C₁) <: MixedSplineSpaceConstraints{(:V,:Q)}
    @test typeof(C₂) <: MixedSplineSpaceConstraints{(:V,:Q)}
    @test typeof(C₁.V) <: VectorSplineSpaceConstraints{2,2}
    @test typeof(C₂.V) <: VectorSplineSpaceConstraints{3,3}
    @test typeof(C₁.Q) <: ScalarSplineSpaceConstraints{2}
    @test typeof(C₂.Q) <: ScalarSplineSpaceConstraints{3}
    @test length(C₁.V) == 2
    @test length(C₂.V) == 3
end

@testset "Taylor-Hood spline spaces" begin
    n1, n2, n3 = 10, 20, 30
    Δ₁ = Partition(Ω², (n1 + 1, n2 + 1))
    Δ₂ = Partition(Ω³, (n1 + 1, n2 + 1, n3 + 1))
    p = 4

    T = eltype(Ω¹)

    V₁ = TaylorHood(p, Δ₁)
    V₂ = TaylorHood(p, Δ₂)

    # verify space construction
    @test typeof(V₁) <: TaylorHood{2,3,T}
    @test typeof(V₂) <: TaylorHood{3,4,T}

    @test all(map(Degree, V₁.V[1]) .== (p, p))
    @test all(map(Degree, V₁.V[2]) .== (p, p))
    @test all(map(Degree, V₁.Q) .== (p-1, p-1))

    @test all(map(Degree, V₂.V[1]) .== (p, p, p))
    @test all(map(Degree, V₂.V[2]) .== (p, p, p))
    @test all(map(Degree, V₂.V[3]) .== (p, p, p))
    @test all(map(Degree, V₂.Q) .== (p-1, p-1, p-1))

    @test all(map(num_elements, V₁.V[1]) .== (n1, n2))
    @test all(map(num_elements, V₁.V[2]) .== (n1, n2))
    @test all(map(num_elements, V₁.Q) .== (n1, n2))

    @test all(map(num_elements, V₂.V[1]) .== (n1, n2, n3))
    @test all(map(num_elements, V₂.V[2]) .== (n1, n2, n3))
    @test all(map(num_elements, V₂.V[3]) .== (n1, n2, n3))
    @test all(map(num_elements, V₂.Q) .== (n1, n2, n3))

    # test mixed space constraints for Taylor-Hood
    C₁ = MixedSplineSpaceConstraints(V₁)
    C₂ = MixedSplineSpaceConstraints(V₂)
    @test typeof(C₁) <: MixedSplineSpaceConstraints{(:V,:Q)}
    @test typeof(C₂) <: MixedSplineSpaceConstraints{(:V,:Q)}
    @test typeof(C₁.V) <: VectorSplineSpaceConstraints{2,2}
    @test typeof(C₂.V) <: VectorSplineSpaceConstraints{3,3}
    @test typeof(C₁.Q) <: ScalarSplineSpaceConstraints{2}
    @test typeof(C₂.Q) <: ScalarSplineSpaceConstraints{3}
    @test length(C₁.V) == 2
    @test length(C₂.V) == 3
end

@testset "ScalarSplineSpace extraction operators" begin
    C₁ = ScalarSplineSpaceConstraints{2}()
    left_constraint!(C₁; dim=1)
    right_constraint!(C₁; dim=1)
    left_constraint!(C₁; dim=2)
    right_constraint!(C₁; dim=2)

    C₂ = ScalarSplineSpaceConstraints{2}()
    left_constraint!(C₁; dim=1)
    right_constraint!(C₁; dim=1)

    Δ = Partition(Ω², (4, 5))
    p₁, p₂ = 3, 4

    S₁ = ScalarSplineSpace((p₁, p₂), Δ, C₁)

    op = extraction_operator(S₁)
    op₁_test = KroneckerProduct(s -> s.C, S₁; reverse=true)
    @test op == op₁_test

    S₂ = ScalarSplineSpace((p₁, p₂), Δ)
    S₃ = ScalarSplineSpace(S₂, C₂)
    op₃_test = KroneckerProduct(s -> s.C, S₃; reverse=true)
    op = extraction_operator(S₃)
    @test op == op₃_test

    op = extraction_operator(S₃, sparse=true)
    @test Matrix(op) == Matrix(op₃_test)
end

@testset "VectorSplineSpace extraction operators" begin
    C₁ = ScalarSplineSpaceConstraints{2}()
    left_constraint!(C₁; dim=1)
    right_constraint!(C₁; dim=1)
    left_constraint!(C₁; dim=2)
    right_constraint!(C₁; dim=2)

    C₂ = ScalarSplineSpaceConstraints{2}()
    left_constraint!(C₁; dim=1)
    right_constraint!(C₁; dim=1)

    C = VectorSplineSpaceConstraints(C₁, C₂)

    Δ = Partition(Ω², (4, 5))
    p₁, p₂ = 3, 4

    S₁ = ScalarSplineSpace((p₁, p₂), Δ, C₁)
    S₂ = ScalarSplineSpace((p₁, p₂), Δ, C₂)
    V = VectorSplineSpace(S₁, S₂)

    op₁_test = KroneckerProduct(s -> s.C, S₁; reverse=true)
    op₂_test = KroneckerProduct(s -> s.C, S₂; reverse=true)
    op = extraction_operators(V)
    @test op == (op₁_test, op₂_test)

    V = VectorSplineSpace(V, C)
    op = extraction_operators(V)
    @test op[1] == op₁_test
    @test op[2] == op₂_test
end

@testset "MixedFunctionSpace extraction operators" begin
    C₁ = ScalarSplineSpaceConstraints{2}()
    left_constraint!(C₁; dim=1)
    right_constraint!(C₁; dim=1)
    left_constraint!(C₁; dim=2)
    right_constraint!(C₁; dim=2)

    C₂ = ScalarSplineSpaceConstraints{2}()
    left_constraint!(C₂; dim=1)
    right_constraint!(C₂; dim=1)


    C = MixedSplineSpaceConstraints{(:V,:Q)}((
        VectorSplineSpaceConstraints{2}(),
        ScalarSplineSpaceConstraints{2}()
    ))

    left_constraint!(C.V[1]; dim=1)
    right_constraint!(C.V[1]; dim=1)
    left_constraint!(C.V[1]; dim=2)
    right_constraint!(C.V[1]; dim=2)
    left_constraint!(C.V[2]; dim=1)
    right_constraint!(C.V[2]; dim=1)

    p, Δ = 4, Partition(Ω², (4, 5))

    S₁ = ScalarSplineSpace((p,p), Δ, C₁)
    S₂ = ScalarSplineSpace((p, p), Δ, C₂)
    th = TaylorHood(p, Δ, C)

    Qdim = dimension(th, :Q) # 42

    op₁_test = KroneckerProduct(s -> s.C, S₁; reverse=true)
    op₂_test = KroneckerProduct(s -> s.C, S₂; reverse=true)
    op₃_test = I(Qdim)

    @test extraction_operators(th, :V) == (op₁_test, op₂_test)
    @test extraction_operator(th, :Q) == op₃_test
end

@testset "IterableMixedSplineSpace" begin
    Δ = Partition(Ω², (4, 5))
    p = 4

    # construct regular Raviart-Thomas space
    rt = RaviartThomas(p, Δ)

    # construct iterable version of Raviart-Thomas space
    irt = IterableMixedSplineSpace((V=rt.V, Q=rt.Q))    

    # bulk of tests
    @test irt.V == rt.V
    @test irt.Q == rt.Q
    @test irt[1] == rt.V
    @test irt[2] == rt.Q
    @test dimension.(irt) == [dimension(rt, :V), dimension(rt, :Q)]
    @test indices(irt) == indices(rt)
    @test indices(irt, :V) == indices(rt, :V)
    @test indices(irt, :Q) == indices(rt, :Q)
    @test dimfunspace(irt) == dimfunspace(rt)
    @test codimfunspace(irt) == codimfunspace(rt)
end