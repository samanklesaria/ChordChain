using ChordChain, Test, LinearAlgebra

let
    D = 4
    P_DD = stack([ones(D) ./ D for _ in 1:D])
    obs = cumprod(0.8 * ones(D))
    y_CT = stack([circshift(obs, i) for i in 0:5])
    results_DT = forward_backward(P_DD, y_CT, 1.0I(4))
    @test all(argmax.(eachcol(results_DT)) .== (mod.(0:5, 4) .+ 1))
    @test sum(results_DT, dims=1) ≈ ones(1,6)
end
