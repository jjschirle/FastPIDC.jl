using FastPIDC
using Test
using CUDA
using Statistics

const BB_NUMERIC_DATA_DIR = joinpath(dirname(@__FILE__), "data")

"""
Return the unique undirected edge weights and their `(i, j)` node-index pairs.
Only the strict upper triangle is used so the diagonal and symmetric duplicate
entries do not distort the numerical diagnostics.
"""
function _bb_undirected_scores(scores::AbstractMatrix{<:Real})
    n, m = size(scores)
    n == m || throw(ArgumentError("score matrix must be square"))

    number_of_edges = n * (n - 1) ÷ 2
    weights = Vector{Float64}(undef, number_of_edges)
    pairs = Vector{Tuple{Int,Int}}(undef, number_of_edges)

    edge = 1
    @inbounds for j = 2:n
        for i = 1:(j-1)
            weights[edge] = Float64(scores[i, j])
            pairs[edge] = (i, j)
            edge += 1
        end
    end

    return weights, pairs
end

function _bb_top_edge_set(
    weights::AbstractVector{<:Real},
    pairs::AbstractVector{Tuple{Int,Int}},
    k::Int,
)
    top_k = min(k, length(weights))
    order = sortperm(eachindex(weights); by = i -> (-weights[i], i))
    return Set(pairs[order[1:top_k]]), top_k
end

if CUDA.functional()
    @testset "Bayesian-block CPU/CPU vs GPU/GPU numeric diagnostics" begin
        dataset = joinpath(BB_NUMERIC_DATA_DIR, "toy_small_200.txt")

        config_cpu = PIDCConfig(
            backend = :cpu,
            bb_backend = :cpu,
            verbose = false,
        )
        config_gpu = PIDCConfig(
            backend = :cuda,
            bb_backend = :cuda,
            verbose = false,
        )

        # CPU/CPU means CPU Bayesian blocks followed by CPU PUC scoring.
        nodes_cpu = get_nodes(
            dataset;
            discretizer = "bayesian_blocks",
            estimator = "maximum_likelihood",
            number_of_bins = 10,
            bb_backend = config_cpu.bb_backend,
            verbose = false,
        )
        mi_cpu, puc_cpu = FastPIDC.compute_puc_full(
            nodes_cpu;
            config = config_cpu,
            base = 2,
        )

        # GPU/GPU means CUDA Bayesian blocks followed by CUDA PUC scoring.
        nodes_gpu = get_nodes(
            dataset;
            discretizer = "bayesian_blocks",
            estimator = "maximum_likelihood",
            number_of_bins = 10,
            bb_backend = config_gpu.bb_backend,
            verbose = false,
        )
        mi_gpu, puc_gpu = FastPIDC.compute_puc_full(
            nodes_gpu;
            config = config_gpu,
            base = 2,
        )

        @test getfield.(nodes_gpu, :label) == getfield.(nodes_cpu, :label)
        @test size(mi_gpu) == size(mi_cpu)
        @test size(puc_gpu) == size(puc_cpu)
        @test mi_cpu == transpose(mi_cpu)
        @test puc_cpu == transpose(puc_cpu)
        @test isapprox(mi_gpu, transpose(mi_gpu); atol = 1e-8, rtol = 1e-10)
        @test isapprox(puc_gpu, transpose(puc_gpu); atol = 1e-8, rtol = 1e-10)

        cpu_weights, edge_pairs = _bb_undirected_scores(puc_cpu)
        gpu_weights, gpu_edge_pairs = _bb_undirected_scores(puc_gpu)
        @test gpu_edge_pairs == edge_pairs

        absolute_errors = abs.(gpu_weights .- cpu_weights)
        max_absolute_error = maximum(absolute_errors)

        # Relative error and ratios are undefined or uninformative when the CPU
        # reference is effectively zero, so exclude only those values.
        reference_floor = 1e-12
        nonzero_reference = abs.(cpu_weights) .> reference_floor
        relative_errors = absolute_errors[nonzero_reference] ./
                          abs.(cpu_weights[nonzero_reference])
        ratios = gpu_weights[nonzero_reference] ./ cpu_weights[nonzero_reference]

        max_relative_error =
            isempty(relative_errors) ? 0.0 : maximum(relative_errors)
        mean_ratio = isempty(ratios) ? 1.0 : mean(ratios)

        cpu_top, top_k = _bb_top_edge_set(cpu_weights, edge_pairs, 250)
        gpu_top, _ = _bb_top_edge_set(gpu_weights, edge_pairs, 250)
        top_overlap = length(intersect(cpu_top, gpu_top))

        println("\n--- Bayesian Blocks CPU/CPU vs GPU/GPU Diagnostic ---")
        println("[Diagnostics] Top $top_k Edge Overlap: $top_overlap / $top_k")
        println("[Diagnostics] Max Absolute Error: $max_absolute_error")
        println("[Diagnostics] Max Relative Error: $max_relative_error")
        println("[Diagnostics] Mean Ratio (GPU/CPU): $mean_ratio")

        # Retain broad integrity assertions while leaving the printed metrics
        # available for judging smaller CUDA/libm rank and score differences.
        @test top_overlap == top_k
        @test all(
            isapprox.(
                cpu_weights,
                gpu_weights;
                atol = 1e-8,
                rtol = 1e-10,
            ),
        )
    end
else
    @warn "CUDA unavailable. Skipping Bayesian-block CPU/CPU vs GPU/GPU numeric diagnostics."
end
