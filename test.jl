using ChordChain, Test, LinearAlgebra, StatsBase

@testset "Log matmul" begin
    for _ in 1:10
        A = randn(3, 3)
        b = randn(3)
        @test all(ChordChain.log_matmul(A, b) .≈ log.(exp.(A) * exp.(b)))
    end
end

@testset "Uniform PDD" begin
    D = 4
    P_DD = stack([ones(D) ./ D for _ in 1:D])
    obs = cumprod(0.8 * ones(D))
    y_CT = stack([circshift(obs, i) for i in 0:5])
    results = [forward_backward_logscale(log.(P_DD), y_CT, 1.0I(4)), forward_backward(P_DD, y_CT, 1.0I(4))]
    @test allequal(x -> trunc.(x; digits=4), results)
    results_DT = results[1]
    @test all(argmax.(eachcol(results_DT)) .== (mod.(0:5, 4) .+ 1))
    @test sum(results_DT, dims=1) ≈ ones(1, 6)
end

@testset "Simulation" begin
    D = 4
    C = 4
    T = 20
    for _ in 1:10
        P_DD = rand(D, D)
        P_DD ./= sum(P_DD, dims=1)
        templates_DC = 1.0I(4) .+ 0.1
        z_T = zeros(Int, T)
        z_T[1] = 1
        y_CT = zeros(C, T)
        for t in 1:T
            y_CT[:, t] = templates_DC[:, z_T[t]] + 0.1 * randn(D)
            if t < T
                z_T[t+1] = sample(ProbabilityWeights(P_DD[:, z_T[t]]))
            end
        end

        results = [forward_backward_logscale(log.(P_DD), y_CT, templates_DC), forward_backward(P_DD, y_CT, templates_DC)]
        @test allequal(x -> trunc.(x; digits=4), results)
        pred_DT = results[1]
        expected_correct = sum(pred_DT[CartesianIndex.(z_T, 1:T)])
        @test expected_correct >= T * 0.7
        @test sum(pred_DT, dims=1) ≈ ones(1, T)
    end
end
