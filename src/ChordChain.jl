module ChordChain
using SizeCheck
using DataFrames, StatsBase, ToeplitzMatrices, PythonCall, DelimitedFiles, MusicTheory, BlockArrays, LinearAlgebra
using Infiltrator
using GLMakie

export forward_backward, load_df, accuracy

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
            y_TC = chroma24[:, 1:12] + chroma24[:, 13:24]
            y_TC = circshift(y_TC, (0, -3)) # First chroma bin corresponds to A, not C
            sums_T = norm.(eachrow(y_TC))
            chroma_mask_T = sums_T .> 0
            y_TC[chroma_mask_T, :] ./= sums_T[chroma_mask_T, :]
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
            tdf[!, :shifted_y] = shifted_y

            push!(train_dfs, DataFrame(scale=tdf.scale, shifted_y=shifted_y, hops=hops))
            push!(test_dfs, tdf[!, [:y, :z, :shifted_y]])
        end
    end
    reduce(vcat, train_dfs), test_dfs
end

@sizecheck function forward_backward(P_DD, y_CT, templates_DC)
    z_DT = zeros(D, T)
    z_DT[:, 1] .= ones(D) ./ D
    obs_prob_T = zeros(T)

    # Forward
    for t in 1:T
        lik_D = templates_DC * y_CT[:, t]
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
        lik_D = templates_DC * y_CT[:, t]
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

@sizecheck function accuracy(df, test_dfs)
    all_hops = crossjoin(DataFrame(scale=0:2), DataFrame(hops=0:24))
    hop_counts = combine(groupby(df, [:scale, :hops]), :hops => (x -> size(x, 1)) => :count)
    hop_counts = coalesce.(leftjoin(all_hops, hop_counts, on=[:scale, :hops], order=:left), 0)
    hop_count_mat = reshape(hop_counts.count, (:, 3))
    P_DM = hop_count_mat ./ sum(hop_count_mat; dims=1)
    templates_CM = reduce(hcat, combine(groupby(df, :scale), :shifted_y => (x -> [mean(x)]) => :y).y)
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

end # module ChordChain
