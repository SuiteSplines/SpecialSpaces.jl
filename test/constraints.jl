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