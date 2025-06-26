@testset "Boundary labels" begin
    @test boundary_symbol(1) == :left
    @test boundary_symbol(2) == :right
    @test boundary_symbol(3) == :bottom
    @test boundary_symbol(4) == :top
    @test boundary_symbol(5) == :back
    @test boundary_symbol(6) == :front

    @test boundary_number(:left) == 1
    @test boundary_number(:right) == 2
    @test boundary_number(:bottom) == 3
    @test boundary_number(:top) == 4
    @test boundary_number(:back) == 5
    @test boundary_number(:front) == 6
end

@testset "Check validity of boundary label" begin
    @test SpecialSpaces.check_boundary_label(Val(2), :left) == true
    @test SpecialSpaces.check_boundary_label(Val(2), :front) == false
    @test SpecialSpaces.check_boundary_label(Val(2), :foo) == false
    @test SpecialSpaces.check_boundary_label(Val(3), :left) == true
    @test SpecialSpaces.check_boundary_label(Val(3), :front) == true
    @test SpecialSpaces.check_boundary_label(Val(3), :foo) == false
    @test all(SpecialSpaces.check_boundary_label.(Val(2), (:left, :right))) == true
    @test all(SpecialSpaces.check_boundary_label.(Val(2), (:left, :front))) == false
    @test all(SpecialSpaces.check_boundary_label.(Val(3), (:left, :front))) == true
    @test all(SpecialSpaces.check_boundary_label.(Val(3), (:left, :foo))) == false
end