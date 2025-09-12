module ChordChain
using SizeCheck
using DataFrames, StatsBase, ToeplitzMatrices, PythonCall, DelimitedFiles, MusicTheory, BlockArrays, LinearAlgebra, LogExpFunctions, FillArrays
using Smoothers
import DataFrames: StackedVector
using Infiltrator
using GLMakie

include("broadcast.jl")

export forward_backward, load_df, accuracy, doit

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

const CHUNK = 8

@sizecheck function load_df(k)
    kagglehub = pyimport("kagglehub")
    mcgill_path = pyconvert(String, kagglehub.dataset_download("jacobvs/mcgill-billboard"))
    train_dfs = DataFrame[]
    test_dfs = DataFrame[]
    for f in readdir(joinpath(mcgill_path, "annotations/annotations/"))[1:k]
        if !startswith(f, '.')

            # Check if annotations exist
            adata = readdlm(joinpath(mcgill_path, "annotations/annotations/$f/majmin.lab"))
            chords = split.(adata[:, 3], ":")
            a_mask = length.(chords) .== 2
            if sum(a_mask) == 0
                continue # This file contains no annotations
            end

            # Extract and normalize chroma
            cqt_data = readdlm(joinpath(mcgill_path, "metadata/metadata/$f/bothchroma.csv"), ',')
            chroma24 = Matrix{Float64}(cqt_data[:, 3:end])
            y_SC = chroma24[:, 1:12] .+ chroma24[:, 13:24] .+ 1e-8
            y_SC = circshift(y_SC, (0, -3)) # First chroma bin corresponds to A, not C
            y_SC ./= mapslices(norm, y_SC; dims=2)
            y_SC = mapslices(x -> hma(x, 13), y_SC; dims=1)
            y_TC = y_SC[1:CHUNK:end, :]
            frame_secs = CHUNK * (cqt_data[2, 2] - cqt_data[1, 2])

            # Extract annotations
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

@sizecheck function forward_backward(P_DD, y_CT, templates_DC, template_norm_D, Lam_DC)
    z_DT = zeros(D, T)
    z_DT[:, 1] .= ones(D) ./ D
    obs_prob_T = zeros(T)

    products_DT = (templates_DC .* Lam_DC) * y_CT
    norms_DT = template_norm_D .+ Lam_DC * (y_CT .^ 2)
    lik_DT = exp.(products_DT .- 0.5 .* norms_DT)

    # Forward
    for t in 1:T
        joint_D = z_DT[:, t] .* lik_DT[:, t]
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
        betas_DT[:, t-1] .= (P_DD' * (lik_DT[:, t] .* betas_DT[:, t])) ./ obs_prob_T[t]
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

@sizecheck function mean_and_var(x_T)
    m_C = mean(x_T)
    x_CT = stack(x_T) .- reshape(m_C, C, 1)
    (y=[m_C], y_var=[vec(sum(x_CT .^ 2; dims=2) ./ (T - 1))])
end

# We're always picking no_chord. Something's screwy. Is there just no variance there?
# No, the variances there actually look quite good.
# But something is broken.

@sizecheck function accuracy(df, test_dfs)
    all_hops = crossjoin(DataFrame(scale=0:2), DataFrame(hops=0:24))
    hop_counts = combine(groupby(df, [:scale, :hops]), :hops => (x -> size(x, 1)) => :count)
    hop_counts = coalesce.(leftjoin(all_hops, hop_counts, on=[:scale, :hops], order=:left), 0)
    hop_count_mat = reshape(hop_counts.count, (:, 3))
    P_DM = hop_count_mat ./ sum(hop_count_mat; dims=1)
    df = combine(groupby(df, :scale), :shifted_y => (x -> mean_and_var(x)) => [:y, :y_var])
    templates_CM = stack(df.y)
    templates_var_CM = stack(df.y_var)
    templates_DC = mortar(reshape(
        [Circulant(templates_CM[:, 1]), Circulant(templates_CM[:, 2]), templates_CM[:, 3:3]], 1, 3))'
    blocks = [to_block(P_DM, i, j) for j in [1, 2, 3] for i in [1:12, 13:24, 25:25]]
    P_DD = mortar(reshape(blocks, 3, 3))
    return templates_var_CM
    Lam_CM = 1 ./ templates_var_CM
    Lam_DC = mortar(reshape(
        [Circulant(Lam_CM[:, 1]), Circulant(Lam_CM[:, 2]), Lam_CM[:, 3:3]], 1, 3))'
    template_norm_D = (templates_DC .^ 2 .* Lam_DC) * Ones(C)
    tdf = test_dfs[1]
    # sum(test_dfs) do tdf
    y_CT = reduce(hcat, tdf.y)
    z_T = tdf.z
    pred_DT = forward_backward(P_DD, y_CT, templates_DC, template_norm_D, Lam_DC)
    println("Accuracy ", mean(pred_DT[CartesianIndex.(z_T, 1:T)])) #  / length(test_dfs))
    visualize_results(pred_DT, z_T)
    # end
end

@sizecheck function visualize_results(pred_DL, z_L)
    fig = Figure()
    ax = Axis(fig[1, 1])
    hm = heatmap!(ax, pred_DL')
    Colorbar(fig[1, 2], hm)
    scatter!(ax, 1:L, z_L, strokewidth=0.5, color=:red)
    fig
end

# 77%
function doit()
    accuracy(load_df(100)...)
end

end # module ChordChain
