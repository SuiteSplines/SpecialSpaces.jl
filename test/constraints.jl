using IgaBase, AbstractMappings

@testset "Univariate spline space constraints" begin
    # init univariate spline constraints container
    C = UnivariateSplineSpaceConstraints{Int64}()
    @test typeof(C) <: UnivariateSplineSpaceConstraints{Int64}

    C = UnivariateSplineSpaceConstraints()
    @test typeof(C) <: UnivariateSplineSpaceConstraints{Int8}

    # check interface
    @test hasfield(UnivariateSplineSpaceConstraints, :left)
    @test hasfield(UnivariateSplineSpaceConstraints, :right)
    @test hasfield(UnivariateSplineSpaceConstraints, :periodic)

    push!(C.left, 1)
    push!(C.right, 2)
    push!(C.periodic, 3)

    @test C.left == [1]
    @test C.right == [2]
    @test C.periodic == [3]
end

@testset "Scalar spline space constraints" begin
    # check iterable element type
    @test eltype(ScalarSplineSpaceConstraints{2,Int}) == UnivariateSplineSpaceConstraints{Int}

    # init scalar spline space constraints
    C = ScalarSplineSpaceConstraints{2}()

    # set some constraints
    left_constraint!(C; dim=1)
    left_constraint!(C; c=[1, 2], dim=2)
    right_constraint!(C; dim=1)
    right_constraint!(C; c=[1, 2], dim=2)
    periodic_constraint!(C; c=[1], dim=1)
    periodic_constraint!(C; c=[1, 2], dim=2)

    # test univariate containers
    @test C[1].left == [1]
    @test C[2].left == [1,2]
    @test C[1].right == [1]
    @test C[2].right == [1,2]
    @test C[1].periodic == [1]
    @test C[2].periodic == [1,2]

    # test iteration over container and eltype on instance
    @test all(map(typeof, C) .<: eltype(C))
end

@testset "Vector spline space constraints" begin
    # check iterable element type
    @test eltype(VectorSplineSpaceConstraints{2,3,Int8}) == ScalarSplineSpaceConstraints{2,Int8}

    # init vector spline space constraints
    C = VectorSplineSpaceConstraints{2}()

    # set some constraints
    left_constraint!(C[1]; dim=1)
    left_constraint!(C[1]; c=[1, 2], dim=2)
    right_constraint!(C[2]; dim=1)
    right_constraint!(C[2]; c=[1, 2], dim=2)

    # check univariate containers
    @test C[1][1].left == [1]
    @test C[1][2].left == [1,2]
    @test C[2][1].right == [1]
    @test C[2][2].right == [1,2]

    # test iteration over container and eltype on instance
    @test all(map(typeof, C) .<: eltype(C))
end

@testset "Mixed spline space constraints" begin
    # init named tuple with vector and scalar constraints
    C = MixedSplineSpaceConstraints{(:V, :Q)}((
        VectorSplineSpaceConstraints{2}(),
        ScalarSplineSpaceConstraints{2}()
    ))

    # set some constraints
    left_constraint!(C.Q; dim=1)
    left_constraint!(C.Q; c=[1, 2], dim=2)
    right_constraint!(C.Q; dim=1)
    right_constraint!(C.Q; c=[1, 2], dim=2)
    periodic_constraint!(C.Q; c=[1], dim=1)
    periodic_constraint!(C.Q; c=[1, 2], dim=2)

    left_constraint!(C.V[1]; dim=1)
    left_constraint!(C.V[1]; c=[1, 2], dim=2)

    right_constraint!(C.V[2]; dim=1)
    right_constraint!(C.V[2]; c=[1, 2], dim=2)

    # test univariate containers
    @test C.Q[1].left == [1]
    @test C.Q[2].left == [1,2]
    @test C.Q[1].right == [1]
    @test C.Q[2].right == [1,2]
    @test C.Q[1].periodic == [1]
    @test C.Q[2].periodic == [1,2]

    @test C.V[1][1].left == [1]
    @test C.V[1][2].left == [1,2]

    @test C.V[2][1].right == [1]
    @test C.V[2][2].right == [1,2]
end

@testset "Clamped scalar spline spaces" begin
    # mapping domain
    Ω = Interval(1.0, 2.0) ⨱ Interval(3.0, 4.0) ⨱ Interval(5.0, 6.0)

    # mapping partition
    Δ = Partition(Ω, (5, 5, 5))

    # proto scalar spline space
    S = ScalarSplineSpace(2, Δ)

    
    # test constant mapping without any constraints
    fʰ = GeometricMapping(Ω, S)
    fʰ[1].coeffs .= 1.0
    @evaluate y = fʰ(Δ)
    @test all(y .== 1.0)

    # test constant mapping with clamped constraint on :left and :bottom
    C = ScalarSplineSpaceConstraints(S)
    clamped_constraint!(C, :left, :bottom)

    fʰ = GeometricMapping(Ω, ScalarSplineSpace(S,C))
    fʰ[1].coeffs .= 1.0
    @evaluate y = fʰ(Δ)
    @test all(boundary(y, boundary_number(:left)) .== 0.0)
    @test all(boundary(y, boundary_number(:bottom)) .== 0.0)

    # test constant mapping with clamped constraint on each boundary
    for side = [:left, :right, :bottom, :top, :back, :front]
        C = ScalarSplineSpaceConstraints(S)
        clamped_constraint!(C, side)

        fʰ = GeometricMapping(Ω, ScalarSplineSpace(S,C))
        fʰ[1].coeffs .= 1.0
        @evaluate y = fʰ(Δ)
        @test all(boundary(y, boundary_number(side)) .== 0)
    end

    # test check for valid boundary label
    @test_throws AssertionError("invalid boundary label") clamped_constraint!(C, :left, :foo)
end

@testset "Clamped vector spline spaces" begin
    # mapping domain
    Ω = Interval(1.0, 2.0) ⨱ Interval(3.0, 4.0) ⨱ Interval(5.0, 6.0)

    # mapping partition
    Δ = Partition(Ω, (5, 5, 5))

    # proto vector spline space
    V = VectorSplineSpace(2, Δ)


    # test constant mapping without any constraints in each dimension
    fʰ = GeometricMapping(Ω, V)
    fʰ[1].coeffs .= 1.0
    fʰ[2].coeffs .= 2.0
    fʰ[3].coeffs .= 3.0
    @evaluate y = fʰ(Δ)
    @test all(isapprox.(y.data[1], 1.0, atol=1e-12))
    @test all(isapprox.(y.data[2], 2.0, atol=1e-12))
    @test all(isapprox.(y.data[3], 3.0, atol=1e-12))

    # test constant mapping with clamped constraint on :left and :right in dimensions 1 and 3
    C = VectorSplineSpaceConstraints(V)
    clamped_constraint!(C, :left, :right; dim=[1,3])

    fʰ = GeometricMapping(Ω, VectorSplineSpace(V,C))
    fʰ[1].coeffs .= 1.0
    fʰ[2].coeffs .= 2.0
    fʰ[3].coeffs .= 3.0
    @evaluate! y = fʰ(Δ)
    @test all(boundary(y.data[1], boundary_number(:left)) .== 0)
    @test all(boundary(y.data[2], boundary_number(:left)) .== 2.0)
    @test all(boundary(y.data[3], boundary_number(:left)) .== 0)
    @test all(boundary(y.data[1], boundary_number(:right)) .== 0)
    @test all(boundary(y.data[2], boundary_number(:right)) .== 2.0)
    @test all(boundary(y.data[3], boundary_number(:right)) .== 0)

    # test constant mapping with clamped constraint on each boundary in all dimensions
    for side = [:left, :right, :bottom, :top, :back, :front]
        C = VectorSplineSpaceConstraints(V)
        clamped_constraint!(C, side)

        fʰ = GeometricMapping(Ω, VectorSplineSpace(V,C))
        fʰ[1].coeffs .= 1.0
        fʰ[2].coeffs .= 2.0
        fʰ[3].coeffs .= 3.0
        @evaluate y = fʰ(Δ)
        @test all(boundary(y.data[1], boundary_number(side)) .== 0)
        @test all(boundary(y.data[2], boundary_number(side)) .== 0)
        @test all(boundary(y.data[3], boundary_number(side)) .== 0)
    end

    # test check for valid boundary label
    @test_throws AssertionError("invalid boundary label") clamped_constraint!(C, :left, :foo)
end