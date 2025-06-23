using SpecialSpaces
using Documenter

DocMeta.setdocmeta!(SpecialSpaces, :DocTestSetup, :(using SpecialSpaces); recursive=true)

makedocs(;
    modules=[SpecialSpaces],
    authors="Michał Mika and contributors",
    sitename="SpecialSpaces.jl",
    format=Documenter.HTML(;
        canonical="https://SuiteSplines.github.io/SpecialSpaces.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/SuiteSplines/SpecialSpaces.jl",
    devbranch="main",
)
