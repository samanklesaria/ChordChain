using ChordChain, Test, LinearAlgebra, StatsBase

@testset "Uniform PDD" begin
    D = 4
    P_DD = stack([ones(D) ./ D for _ in 1:D])
    obs = cumprod(0.8 * ones(D))
    y_CT = stack([circshift(obs, i) for i in 0:5])
    pred_DT = forward_backward(P_DD, y_CT, 1.0I(4), ones(4), 1.0)
    @test all(argmax.(eachcol(pred_DT)) .== (mod.(0:5, 4) .+ 1))
    @test sum(pred_DT, dims=1) ≈ ones(1, 6)
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

        norms_D = sum(templates_DC.^2; dims=2)
        pred_DT = forward_backward(P_DD, y_CT, templates_DC, norms_D, 0.1^2)
        expected_correct = sum(pred_DT[CartesianIndex.(z_T, 1:T)])
        @test expected_correct >= T * 0.95
        @test all(argmax.(eachcol(pred_DT)) .== z_T)
        @test sum(pred_DT, dims=1) ≈ ones(1, T)
    end
end
