using ChordChain, Test, ToeplitzMatrices

@testset "CirculantArrayStyle" begin

    @testset "Unary Broadcasting" begin
        a = Circulant([1., 2., 3.])
        b = exp.(a)
        @test b isa Circulant
        @test b.v == exp.([1., 2., 3.])
    end


    @testset "Scalar Broadcasting" begin
        a = Circulant([1, 2, 3])
        b = a .+ 2
        @test b isa Circulant
        @test b.v == [3, 4, 5]
    end

    @testset "Circulant Broadcasting" begin
        a = Circulant([1, 2, 3])
        b = Circulant([7, 8,9])
        c = a .+ b
        @test c isa Circulant
        @test c.v == [8, 10, 12]
    end

    @testset "Matrix Default" begin
        a = Circulant([1, 2, 3])
        b = ones(3,3)
        c = a .+ b
        @test c isa Matrix
    end

    @testset "Combined with Plus" begin
        a = Circulant([1., 2., 3.])
        b = Circulant([7., 8., 9.])
        c = (a .+ 2) .+ b
        @test c isa Circulant
        @test c.v == (a.v.+ 2) .+ b.v
    end

    @testset "Combined with Power" begin
        a = Circulant([1., 2., 3.])
        b = Circulant([7., 8., 9.])
        c = (a .^ 2) .+ b
        @test c isa Circulant
        @test c.v == (a.v.^ 2) .+ b.v
    end
end

@testset "FillArray Multiplication" begin
    @testset "Ones" begin
        A = Circulant([1,2,3])
        b = Ones(3)
        @test Matrix(A) * b == A * b
    end

    @testset "Fill" begin
        A = Circulant([1,2,3])
        b = Fill(2, 3)
        @test Matrix(A) * b == A * b
    end
end
