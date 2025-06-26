using Test
using SpecialSpaces

tests = [
    "base",
    "constraints",
    "spaces",
    "splinespaces",
    "mappings",
    "boundaries",
]

@testset "SpecialSpaces" begin
    for t in tests
        fp = joinpath(dirname(@__FILE__), "$t.jl")
        println("$fp ...")
        include(fp)
    end
end # @testset
