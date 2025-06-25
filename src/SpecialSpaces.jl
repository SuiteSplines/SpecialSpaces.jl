module SpecialSpaces

    using IgaBase
    using SortedSequences
    using CartesianProducts
    using KroneckerProducts
    using UnivariateSplines
    using AbstractMappings
    using TensorProductBsplines

    include("base.jl")
    include("constraints.jl")
    include("spaces.jl")
    include("splinespaces.jl")
    include("mappings.jl")

    export Partition, Domain
    export indices, dimensions
    export FunctionSpace, VectorFunctionSpace, MixedFunctionSpace
    export dimfunspace, codimfunspace
    export ScalarSplineSpace, VectorSplineSpace, MixedSplineSpace
    export IterableMixedSplineSpace, RaviartThomas, TaylorHood
    export UnivariateSplineSpaceConstraints
    export ScalarSplineSpaceConstraints
    export VectorSplineSpaceConstraints
    export MixedSplineSpaceConstraints
    export extraction_operator, extraction_operators
    export left_constraint!, right_constraint!, periodic_constraint!
    export setcoeffs!, getcoeffs

end # module
