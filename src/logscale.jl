
"Numerically stable version of `log.(exp.(A) * exp.(v))`"
log_matmul(A, v) = logsumexp.(eachrow(A) .+ Ref(v))

@testset "Log matmul" begin
    for _ in 1:10
        A = randn(3, 3)
        b = randn(3)
        @test all(log_matmul(A, b) .≈ log.(exp.(A) * exp.(b)))
    end
end

@sizecheck function forward_backward_logscale(P_DD, y_CT, templates_DC)
    alpha_DT = zeros(D, T)

    # Forward
    for t in 1:T
        lik_D = 2 * (templates_DC * y_CT[:, t])
        alpha_DT[:, t] = alpha_DT[:, t] .+ lik_D
        if t < T
            alpha_DT[:, t+1] .= log_matmul(P_DD, alpha_DT[:, t])
        end
    end

    # Backward
    beta_DT = zeros(D, T)
    for t in (T-1):-1:1
        lik_D = 2 * (templates_DC * y_CT[:, t+1])
        beta_DT[:, t] .= log_matmul(P_DD', beta_DT[:, t+1] .+ lik_D)
    end

    softmax(alpha_DT .+ beta_DT; dims=1)
end
