module ChordChain
using Base: AbstractArrayOrBroadcasted
using SizeCheck
using DataFrames, StatsBase, ToeplitzMatrices, PythonCall, DelimitedFiles, MusicTheory, BlockArrays, LinearAlgebra, LogExpFunctions
using Infiltrator
using GLMakie

export forward_backward_logscale, forward_backward, load_df, accuracy, doit

const ACCIDENTALS = Dict('b' => ♭, '#' => ♯)
const SCALES = Dict{String,Int8}("maj" => 0, "min" => 1)

to_pitch(a) = semitone(length(a) == 1 ? PitchClass(Symbol(a[1])) : PitchClass(Symbol(a[1]), ACCIDENTALS[a[2]]))

# States `z` capture the current chord's scale (major or minor) and root.
# States are represented by integers: 1:12 are major (starting from C) and 13:24 are minor.
# The special no_chord state 25 represents silence or non-melodic content.

# To construct a transition matrix that is equivariant to transposition, we build it out of "hops".
# Hops are represented by integers. 0:11 represent the intervals transitioned to a major chord,
# 12:23 represent the intervals transitioned to a minor chord, and
# 24 represents a transition to the no_chord state.
# We keep track of hop-probabilities separately for major starting chords, minor starting chords, and no_chord.

@sizecheck function load_df(k)
    kagglehub = pyimport("kagglehub")
    mcgill_path = pyconvert(String, kagglehub.dataset_download("jacobvs/mcgill-billboard"))
    train_dfs = DataFrame[]
    test_dfs = DataFrame[]
    for f in readdir(joinpath(mcgill_path, "annotations/annotations/"))[1:k]
        if !startswith(f, '.')

            # Extract and normalize chroma
            cqt_data = readdlm(joinpath(mcgill_path, "metadata/metadata/$f/bothchroma.csv"), ',')
            chroma24 = Matrix{Float64}(cqt_data[:, 3:end])
            y_TC = chroma24[:, 1:12] .+ chroma24[:, 13:24] .+ 1e-8
            y_TC = circshift(y_TC, (0, -3)) # First chroma bin corresponds to A, not C
            sums_T = norm.(eachrow(y_TC))
            y_TC ./= sums_T
            frame_secs = cqt_data[2, 2] - cqt_data[1, 2]

            # Extract annotations
            adata = readdlm(joinpath(mcgill_path, "annotations/annotations/$f/majmin.lab"))
            chords = split.(adata[:, 3], ":")
            a_mask = length.(chords) .== 2
            if sum(a_mask) == 0
                continue # This file contains no annotations
            end
            chords_PA = reduce(hcat, chords[a_mask])
            pitch_A = to_pitch.(chords_PA[1, :])
            scale_A = [SCALES[a] for a in chords_PA[2, :]]
            timing_A2 = min.(T, round.(Int, Matrix{Float64}(adata[a_mask, 1:2]) ./ frame_secs) .+ 1)
            z_A = 1 .+ pitch_A .+ 12 .* scale_A
            adf = DataFrame(z=z_A, scale=scale_A, pitch=pitch_A)

            # Align the annotations by frame
            tdf = DataFrame(z=fill(Int8(25), T), scale=zeros(Int8, T), pitch=zeros(Int8, T))
            for a in 1:A
                tdf[timing_A2[a, 1]:timing_A2[a, 2]-1, :] .= adf[a:a, :]
            end
            tdf[!, :y] = eachrow(y_TC)
            tdf = tdf[timing_A2[1, 1]:timing_A2[A, 2]-1, :]
            tdf.scale[tdf.z.==25] .= 2 # Set scale to 2 for no_chord

            # Canonicalize templates and transitions
            hops = mod.(tdf.pitch[2:end] .- tdf.pitch[1:end-1], 12) .+ 12 .* tdf.scale[2:end]
            hops[tdf.z[2:end].==0] .= 24
            push!(hops, 24)
            shifted_y = circshift.(tdf.y, .-tdf.pitch)

            push!(train_dfs, DataFrame(scale=tdf.scale, shifted_y=shifted_y, hops=hops))
            push!(test_dfs, tdf[!, [:y, :z]])
        end
    end
    reduce(vcat, train_dfs), test_dfs
end

"Numerically stable version of `log.(exp.(A) * exp.(v))`"
log_matmul(A, v) = logsumexp.(eachrow(A) .+ Ref(v))

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

@sizecheck function forward_backward(P_DD, y_CT, templates_DC)
    z_DT = zeros(D, T)
    z_DT[:, 1] .= ones(D) ./ D
    obs_prob_T = zeros(T)

    # Forward
    for t in 1:T
        lik_D = exp.(2 * (templates_DC * y_CT[:, t]))
        joint_D = z_DT[:, t] .* lik_D
        obs_prob_T[t] = sum(joint_D)
        z_DT[:, t] .= joint_D ./ obs_prob_T[t]
        if t < T
            z_DT[:, t+1] = P_DD * z_DT[:, t]
        end
    end

    # Backward
    betas_DT = zeros(D, T)
    betas_DT[:, T] = ones(D)
    for t in T:-1:2
        lik_D = exp.(2 * (templates_DC * y_CT[:, t]))
        betas_DT[:, t-1] .= (P_DD' * (lik_D .* betas_DT[:, t])) ./ obs_prob_T[t]
    end

    betas_DT .* z_DT
end

function to_block(P_DM, i, j)
    if i == 25:25 && j != 3
        return repeat(P_DM[i, j], 1, 12)
    elseif j == 3
        return P_DM[i, j:j]
    else
        return Circulant(P_DM[i, j])
    end
end

function heatmap_with_colorbar(m)
    f = Figure()
    ax = Axis(f[1, 1])
    hm = heatmap!(ax, m)
    Colorbar(f[1, 2], hm)
    f
end

# From the plot, we see that the self-transition probabilities are thousands of times higher than the transition probabilities between chords.
# To combat this, what can we do?
# We can use a non-exponential distribution for the chord transition timing.
# But first things first: how well does this work?

# Hm. Nan. Does that come from the fact that that our FB alg isn't in log space?
# Let's do a log space one. We might be able to do something simpler as a result.
# Also: let's test the FB algorithm by simulating from known distributions and trying to recover the truth.

# Hm: note that we aren't removing empty chroma chords anymore. So these will have zero dot products.
# So they'lll produce probability zero in the HMM. That's probably the source of the NaNs.

# To fix this, we can do what we were going to do anyway: use Gaussians pdfs rather than dot products.
# We'll need to get the variance of the chord templates.
# The other problem is normalization. We've been normalizing the templates. But is that really wise?
# Perhaps what makes the 'no-chord' so characteristic is that it doesn't have any chroma content.

# We don't actually have normalized templates: the mean of unit-norm vectors is not necesarily unit norm!

# Can we we make the distance calculation fast still?

# TODO:
# 1. Calculate the variance of the chord templates.
# 2. Implement a fast distance calculation method using these variances.
# 3. Use the normal pdf instead of the dot product for the likelihood calculation.
# 4. Rewrite the HMM in log space.

@sizecheck function mean_and_cov(x_T)
    m_C = mean(x_T)
    x_CT = stack(x_T)
    X_CT = x_CT .- reshape(m_C, C, 1)
    (y=[m_C], y_var=[X_CT * X_CT' ./ (T - 1)])
end

@sizecheck function accuracy(df, test_dfs)
    all_hops = crossjoin(DataFrame(scale=0:2), DataFrame(hops=0:24))
    hop_counts = combine(groupby(df, [:scale, :hops]), :hops => (x -> size(x, 1)) => :count)
    hop_counts = coalesce.(leftjoin(all_hops, hop_counts, on=[:scale, :hops], order=:left), 0)
    hop_count_mat = reshape(hop_counts.count, (:, 3))
    P_DM = hop_count_mat ./ sum(hop_count_mat; dims=1)
    templates_CM = reduce(hcat, combine(groupby(df, :scale), :shifted_y => (x -> [mean(x)]) => :y).y)
    # df = combine(groupby(df, :scale), :shifted_y => (x -> mean_and_cov(x)) => [:y, :y_var])
    # templates_CM = stack(df.y)
    # templates_CCM = stack(df.y_var) # TODO: how to turn this into templates_DCC?
    templates_DC = mortar(reshape(
        [Circulant(templates_CM[:, 1]), Circulant(templates_CM[:, 2]), templates_CM[:, 3:3]], 1, 3))'
    blocks = [to_block(P_DM, i, j) for j in [1, 2, 3] for i in [1:12, 13:24, 25:25]]
    P_DD = mortar(reshape(blocks, 3, 3))
    sum(test_dfs) do tdf
        y_CT = reduce(hcat, tdf.y)
        z_T = tdf.z
        pred_DT = forward_backward(P_DD, y_CT, templates_DC)
        mean(pred_DT[CartesianIndex.(z_T, 1:T)]) / length(test_dfs)
    end
end

# 64%? That's terrible!
function doit()
    accuracy(load_df(100)...)
end

end # module ChordChain
