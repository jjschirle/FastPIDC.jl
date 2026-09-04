using FastPIDC
using CUDA
using Distributed
using Statistics

# Manual benchmark: run this file directly. Do not include it from test/runtests.jl.
const TARGET_CPU_WORKERS = 10
const CURRENT_CPU_WORKERS = max(nprocs() - 1, 0)

if CURRENT_CPU_WORKERS < TARGET_CPU_WORKERS
    needed = TARGET_CPU_WORKERS - CURRENT_CPU_WORKERS
    active_project = Base.active_project()

    if active_project === nothing
        addprocs(needed)
    else
        addprocs(needed; exeflags = "--project=$(dirname(active_project))")
    end
end

@everywhere using FastPIDC

CUDA.functional() || error("CUDA is not functional on this system.")

const DATA_DIR = joinpath(dirname(@__FILE__), "data")
const DATASET = joinpath(DATA_DIR, "toy_large_1k.txt")
const WARMUP_DATASET = joinpath(DATA_DIR, "toy_small_200.txt")

function load_nodes(path::String, config::PIDCConfig)
    return get_nodes(
        path;
        discretizer = config.discretizer,
        estimator = config.estimator,
        bb_backend = config.bb_backend,
        verbose = config.verbose,
    )
end

function infer_puc(nodes, config::PIDCConfig)
    return InferredNetwork(
        PUCNetworkInference(),
        nodes;
        estimator = config.estimator,
        config = config,
    )
end

function timed_cuda(f::F) where {F}
    CUDA.synchronize()
    result = nothing
    elapsed = @elapsed begin
        result = f()
        CUDA.synchronize()
    end
    return result, elapsed
end

function edge_key(edge)
    label_1 = edge.nodes[1].label
    label_2 = edge.nodes[2].label
    return label_1 <= label_2 ? (label_1, label_2) : (label_2, label_1)
end

function numeric_diagnostics(cpu_network, gpu_network; top_k::Int = 250)
    cpu_weights = Dict(edge_key(edge) => edge.weight for edge in cpu_network.edges)
    gpu_weights = Dict(edge_key(edge) => edge.weight for edge in gpu_network.edges)

    Set(keys(cpu_weights)) == Set(keys(gpu_weights)) ||
        error("CPU and GPU networks do not contain the same edge set.")

    edge_keys = collect(keys(cpu_weights))
    cpu_values = [cpu_weights[key] for key in edge_keys]
    gpu_values = [gpu_weights[key] for key in edge_keys]

    absolute_errors = abs.(gpu_values .- cpu_values)
    nonzero_reference = abs.(cpu_values) .> 1.0e-12
    relative_errors = absolute_errors[nonzero_reference] ./ abs.(cpu_values[nonzero_reference])
    ratios = gpu_values[nonzero_reference] ./ cpu_values[nonzero_reference]

    effective_top_k = min(top_k, length(cpu_network.edges), length(gpu_network.edges))
    cpu_top = Set(edge_key(edge) for edge in cpu_network.edges[1:effective_top_k])
    gpu_top = Set(edge_key(edge) for edge in gpu_network.edges[1:effective_top_k])

    println("\n--- Correctness Check ---")
    println(
        "Top $effective_top_k Edge Overlap: ",
        length(intersect(cpu_top, gpu_top)),
        " / ",
        effective_top_k,
    )
    println("Max Absolute Error: ", maximum(absolute_errors))
    println(
        "Max Relative Error: ",
        isempty(relative_errors) ? NaN : maximum(relative_errors),
    )
    println("Mean Ratio (GPU/CPU): ", isempty(ratios) ? NaN : mean(ratios))
end

function main()
    isfile(DATASET) || error("Benchmark dataset not found: $DATASET")

    config_cpu = PIDCConfig(
        backend = :cpu,
        bb_backend = :cpu,
        discretizer = "bayesian_blocks",
        estimator = "maximum_likelihood",
        verbose = false,
    )

    config_cuda = PIDCConfig(
        backend = :cuda,
        bb_backend = :cuda,
        discretizer = "bayesian_blocks",
        estimator = "maximum_likelihood",
        verbose = false,
    )

    # Compile both execution paths before timing the full dataset.
    if isfile(WARMUP_DATASET)
        println("Warming up CPU and CUDA paths with $WARMUP_DATASET ...")

        warm_cpu_nodes = load_nodes(WARMUP_DATASET, config_cpu)
        infer_puc(warm_cpu_nodes[1:min(100, length(warm_cpu_nodes))], config_cpu)

        warm_cuda_nodes = load_nodes(WARMUP_DATASET, config_cuda)
        infer_puc(warm_cuda_nodes[1:min(100, length(warm_cuda_nodes))], config_cuda)

        warm_cpu_nodes = nothing
        warm_cuda_nodes = nothing
        GC.gc()
        CUDA.reclaim()
    else
        println(
            "Warmup dataset not found; reported timings include first-call compilation.",
        )
    end

    println("\nDataset: $DATASET")
    println("CPU workers for PUC: $(nprocs() - 1)")
    println("Julia threads for BB preparation/CPU BB: $(Threads.nthreads())")

    println("\n--- CPU BB + CPU PUC ---")
    GC.gc()
    cpu_nodes = nothing
    t_cpu_bb = @elapsed cpu_nodes = load_nodes(DATASET, config_cpu)
    println("CPU Bayesian-block time: $t_cpu_bb seconds")
    println("Loaded $(length(cpu_nodes)) CPU-binned nodes.")

    cpu_network = nothing
    t_cpu_puc = @elapsed cpu_network = infer_puc(cpu_nodes, config_cpu)
    println("CPU PUC time: $t_cpu_puc seconds")

    t_cpu_total = t_cpu_bb + t_cpu_puc
    println("CPU total time: $t_cpu_total seconds")

    println("\n--- CUDA BB + CUDA PUC ---")
    GC.gc()
    CUDA.reclaim()

    gpu_nodes, t_cuda_bb = timed_cuda(() -> load_nodes(DATASET, config_cuda))
    println("CUDA Bayesian-block time: $t_cuda_bb seconds")
    println("Loaded $(length(gpu_nodes)) CUDA-binned nodes.")

    gpu_network, t_cuda_puc = timed_cuda(() -> infer_puc(gpu_nodes, config_cuda))
    println("CUDA PUC time: $t_cuda_puc seconds")

    t_cuda_total = t_cuda_bb + t_cuda_puc
    println("CUDA total time: $t_cuda_total seconds")

    numeric_diagnostics(cpu_network, gpu_network)

    println("\n--- Speedup ---")
    println("Bayesian blocks: $(round(t_cpu_bb / t_cuda_bb; digits = 2))x")
    println("PUC scoring:     $(round(t_cpu_puc / t_cuda_puc; digits = 2))x")
    println("End to end:      $(round(t_cpu_total / t_cuda_total; digits = 2))x")

    return nothing
end

main()
