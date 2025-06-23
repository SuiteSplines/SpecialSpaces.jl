using TensorProductBsplines, AbstractMappings, CartesianProducts, IgaBase

Ω¹ = Interval(0.0, 1.0)
Ω² = Ω¹ ⨱ Ω¹
Ω³ = Ω¹ ⨱ Ω¹ ⨱ Ω¹

@testset "(Field) ScalarSplineSpace Dim = 2" begin
    Δ = Partition(Ω², (7, 3))
    p = (4, 7)

    S = ScalarSplineSpace(p, Δ)

    ndofs = dimension(S)
    x = collect(1:ndofs)

    θʰ = Field(S)
    setcoeffs!(θʰ, S, x)

    @test all(θʰ[1].coeffs .== reshape(x, dimensions(S)))
    @test all(x .== getcoeffs(θʰ))
end

@testset "(Field) ScalarScalarSpace Dim = 3" begin
    Δ = Partition(Ω³, (7, 3, 5))
    p = (4, 7, 5)

    S = ScalarSplineSpace(p, Δ)
    ndofs = dimension(S)
    x = collect(1:ndofs)

    θʰ = Field(S)
    setcoeffs!(θʰ, S, x)

    @test all(θʰ[1].coeffs .== reshape(x, dimensions(S)))
    @test all(x .== getcoeffs(θʰ))
end

@testset "(Field) VectorSplineSpace Dim = 2" begin
    Δ = Partition(Ω², (7, 3))
    p = (5, 7)

    V = VectorSplineSpace(p, Δ)
    ndofs = dimension(V)
    nu = map(dimension, V)

    x = collect(1:ndofs)
    xu = (1:nu[1], nu[1]+1:nu[1]+nu[2])

    uʰ = Field(V)
    setcoeffs!(uʰ, V, x)

    @test all(uʰ[1].coeffs .== reshape(xu[1], dimensions(V[1])))
    @test all(uʰ[2].coeffs .== reshape(xu[2], dimensions(V[2])))
    @test all(getcoeffs(uʰ) .== x)
end

@testset "(Field) VectorSplineSpace Dim = 3" begin
    Δ = Partition(Ω³, (7, 3, 5))
    p = (5, 7, 6)

    V = VectorSplineSpace(p, Δ)
    ndofs = dimension(V)
    nu = map(dimension, V)

    x = collect(1:ndofs)
    xu = (1:nu[1], nu[1]+1:nu[1]+nu[2], nu[1]+nu[2]+1:nu[1]+nu[2]+nu[3])

    uʰ = Field(V)
    setcoeffs!(uʰ, V, x)

    @test all(uʰ[1].coeffs .== reshape(xu[1], dimensions(V[1])))
    @test all(uʰ[2].coeffs .== reshape(xu[2], dimensions(V[2])))
    @test all(uʰ[3].coeffs .== reshape(xu[3], dimensions(V[3])))
    @test all(getcoeffs(uʰ) .== x)
end

@testset "(Field) RaviartThomas Dim = 2" begin
    Δ = Partition(Ω², (4, 5))
    p = 4

    S = RaviartThomas(p, Δ)

    Vdim = dimensions(S, :V)
    Qdim = dimensions(S, :Q)
    nu₁ = dimension(S, :V, 1)
    nu₂ = dimension(S, :V, 2)
    np = dimension(S, :Q)

    dofs_u₁ = (2:Vdim[1][1]-1, 1:Vdim[1][2])
    dofs_u₂ = (1:Vdim[2][1], 2:Vdim[2][2]-1)
    dofs_p = (1:Qdim[1], 1:Qdim[2])

    slice_u₁ = 1:nu₁
    slice_u₂ = nu₁+1:nu₁+nu₂
    slice_p = nu₁+nu₂+1:nu₁+nu₂+np

    x = zeros(nu₁ + nu₂ + np)
    x[slice_u₁] .= 1
    x[slice_u₂] .= 2
    x[slice_p] .= 3

    uʰ = Field(S, :V)
    pʰ = Field(S, :Q)

    setcoeffs!(uʰ, S, :V, x)
    setcoeffs!(pʰ, S, :Q, x)

    @test all(uʰ[1].coeffs .== 1)
    @test all(uʰ[2].coeffs .== 2)
    @test all(pʰ[1].coeffs .== 3)
end

@testset "(Field) RaviartThomas Dim = 3" begin
    Δ = Partition(Ω³, (4, 5, 6))
    p = 4

    S = RaviartThomas(p, Δ)

    Vdim = dimensions(S, :V)
    Qdim = dimensions(S, :Q)
    nu₁ = dimension(S, :V, 1)
    nu₂ = dimension(S, :V, 2)
    nu₃ = dimension(S, :V, 3)
    np = dimension(S, :Q)

    dofs_u₁ = (2:Vdim[1][1], 1:Vdim[1][2], 1:Vdim[1][3])
    dofs_u₂ = (1:Vdim[2][1], 2:Vdim[2][2], 1:Vdim[2][3])
    dofs_u₂ = (1:Vdim[2][1], 1:Vdim[2][2], 2:Vdim[2][3])
    dofs_p = (1:Qdim[1], 1:Qdim[2], 1:Qdim[3])

    slice_u₁ = 1:nu₁
    slice_u₂ = nu₁+1:nu₁+nu₂
    slice_u₃ = nu₁+nu₂+1:nu₁+nu₂+nu₃
    slice_p = nu₁+nu₂+nu₃+1:nu₁+nu₂+nu₃+np

    x = zeros(nu₁ + nu₂ + nu₃ + np)
    x[slice_u₁] .= 1
    x[slice_u₂] .= 2
    x[slice_u₃] .= 3
    x[slice_p] .= 4

    uʰ = Field(S, :V)
    pʰ = Field(S, :Q)
    setcoeffs!(uʰ, S, :V, x)
    setcoeffs!(pʰ, S, :Q, x)

    @test all(uʰ[1].coeffs .== 1)
    @test all(uʰ[2].coeffs .== 2)
    @test all(uʰ[3].coeffs .== 3)
    @test all(pʰ[1].coeffs .== 4)
end


@testset "(GeometricMapping) ScalarSplineSpace Dim = 2" begin
    Δ = Partition(Ω², (7, 3))
    p = (4, 7)

    S = ScalarSplineSpace(p, Δ)

    ndofs = dimension(S)
    x = collect(1:ndofs)

    θʰ = GeometricMapping(Ω², S)
    setcoeffs!(θʰ, S, x)

    @test all(θʰ[1].coeffs .== reshape(x, dimensions(S)))
    @test all(x .== getcoeffs(θʰ))
end

@testset "(GeometricMapping) ScalarScalarSpace Dim = 3" begin
    Δ = Partition(Ω³, (7, 3, 5))
    p = (4, 7, 5)

    S = ScalarSplineSpace(p, Δ)
    ndofs = dimension(S)
    x = collect(1:ndofs)

    θʰ = GeometricMapping(Ω³, S)
    setcoeffs!(θʰ, S, x)

    @test all(θʰ[1].coeffs .== reshape(x, dimensions(S)))
    @test all(x .== getcoeffs(θʰ))
end

@testset "(GeometricMapping) VectorSplineSpace Dim = 2" begin
    Δ = Partition(Ω², (7, 3))
    p = (5, 7)

    V = VectorSplineSpace(p, Δ)
    ndofs = dimension(V)
    nu = map(dimension, V)

    x = collect(1:ndofs)
    xu = (1:nu[1], nu[1]+1:nu[1]+nu[2])

    uʰ = GeometricMapping(Ω², V)
    setcoeffs!(uʰ, V, x)

    @test all(uʰ[1].coeffs .== reshape(xu[1], dimensions(V[1])))
    @test all(uʰ[2].coeffs .== reshape(xu[2], dimensions(V[2])))
    @test all(getcoeffs(uʰ) .== x)
end

@testset "(GeometricMapping) VectorSplineSpace Dim = 3" begin
    Δ = Partition(Ω³, (7, 3, 5))
    p = (5, 7, 6)

    V = VectorSplineSpace(p, Δ)
    ndofs = dimension(V)
    nu = map(dimension, V)

    x = collect(1:ndofs)
    xu = (1:nu[1], nu[1]+1:nu[1]+nu[2], nu[1]+nu[2]+1:nu[1]+nu[2]+nu[3])

    uʰ = GeometricMapping(Ω³, V)
    setcoeffs!(uʰ, V, x)

    @test all(uʰ[1].coeffs .== reshape(xu[1], dimensions(V[1])))
    @test all(uʰ[2].coeffs .== reshape(xu[2], dimensions(V[2])))
    @test all(uʰ[3].coeffs .== reshape(xu[3], dimensions(V[3])))
    @test all(getcoeffs(uʰ) .== x)
end

@testset "(GeometricMapping) RaviartThomas Dim = 2" begin
    Δ = Partition(Ω², (4, 5))
    p = 4

    S = RaviartThomas(p, Δ)

    Vdim = dimensions(S, :V)
    Qdim = dimensions(S, :Q)
    nu₁ = dimension(S, :V, 1)
    nu₂ = dimension(S, :V, 2)
    np = dimension(S, :Q)

    dofs_u₁ = (2:Vdim[1][1]-1, 1:Vdim[1][2])
    dofs_u₂ = (1:Vdim[2][1], 2:Vdim[2][2]-1)
    dofs_p = (1:Qdim[1], 1:Qdim[2])

    slice_u₁ = 1:nu₁
    slice_u₂ = nu₁+1:nu₁+nu₂
    slice_p = nu₁+nu₂+1:nu₁+nu₂+np

    x = zeros(nu₁ + nu₂ + np)
    x[slice_u₁] .= 1
    x[slice_u₂] .= 2
    x[slice_p] .= 3

    uʰ = GeometricMapping(Ω², S, :V)
    pʰ = GeometricMapping(Ω², S, :Q)

    setcoeffs!(uʰ, S, :V, x)
    setcoeffs!(pʰ, S, :Q, x)

    @test all(uʰ[1].coeffs .== 1)
    @test all(uʰ[2].coeffs .== 2)
    @test all(pʰ[1].coeffs .== 3)
end

@testset "(GeometricMapping) RaviartThomas Dim = 3" begin
    Δ = Partition(Ω³, (4, 5, 6))
    p = 4

    S = RaviartThomas(p, Δ)

    Vdim = dimensions(S, :V)
    Qdim = dimensions(S, :Q)
    nu₁ = dimension(S, :V, 1)
    nu₂ = dimension(S, :V, 2)
    nu₃ = dimension(S, :V, 3)
    np = dimension(S, :Q)

    dofs_u₁ = (2:Vdim[1][1], 1:Vdim[1][2], 1:Vdim[1][3])
    dofs_u₂ = (1:Vdim[2][1], 2:Vdim[2][2], 1:Vdim[2][3])
    dofs_u₂ = (1:Vdim[2][1], 1:Vdim[2][2], 2:Vdim[2][3])
    dofs_p = (1:Qdim[1], 1:Qdim[2], 1:Qdim[3])

    slice_u₁ = 1:nu₁
    slice_u₂ = nu₁+1:nu₁+nu₂
    slice_u₃ = nu₁+nu₂+1:nu₁+nu₂+nu₃
    slice_p = nu₁+nu₂+nu₃+1:nu₁+nu₂+nu₃+np

    x = zeros(nu₁ + nu₂ + nu₃ + np)
    x[slice_u₁] .= 1
    x[slice_u₂] .= 2
    x[slice_u₃] .= 3
    x[slice_p] .= 4

    uʰ = GeometricMapping(Ω³, S, :V)
    pʰ = GeometricMapping(Ω³, S, :Q)
    setcoeffs!(uʰ, S, :V, x)
    setcoeffs!(pʰ, S, :Q, x)

    @test all(uʰ[1].coeffs .== 1)
    @test all(uʰ[2].coeffs .== 2)
    @test all(uʰ[3].coeffs .== 3)
    @test all(pʰ[1].coeffs .== 4)
end