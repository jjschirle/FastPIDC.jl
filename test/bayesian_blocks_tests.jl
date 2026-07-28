using FastPIDC
using Test

# Frozen copy of the Bayesian-block dynamic program immediately before the
# prefix-count refactor. This remains test-only so optimized implementations
# can be required to reproduce the previous edges bit for bit.
function bayesian_blocks_reference(data::AbstractArray{<:AbstractFloat})
    sorted_data = sort(vec(data))
    m = length(sorted_data)

    unique_data = Vector{Float64}(undef, m)
    nn_vec = Vector{Float64}(undef, m)
    n = 0
    i = 1
    @inbounds while i <= m
        v = sorted_data[i]
        j = i + 1
        while j <= m && sorted_data[j] == v
            j += 1
        end
        n += 1
        unique_data[n] = v
        nn_vec[n] = j - i
        i = j
    end
    resize!(unique_data, n)
    resize!(nn_vec, n)

    edges = zeros(n + 1)
    edges[1] = unique_data[1]
    for i = 1:(n-1)
        edges[i+1] = 0.5 * (unique_data[i] + unique_data[i+1])
    end
    edges[end] = unique_data[end]
    block_length = unique_data[end] .- edges

    count_vec = zeros(n)
    best = zeros(n)
    last = zeros(Int64, n)
    widths = Vector{Float64}(undef, n)
    fit_vec = Vector{Float64}(undef, n)

    @inbounds for K = 1:n
        block_length_K1 = block_length[K+1]
        for i = 1:K
            widths[i] = block_length[i] - block_length_K1
        end
        for i = 1:K
            count_vec[i] += nn_vec[K]
        end

        prior = 4 - log(73.53 * 0.05 * ((K)^-0.478))
        for i = 1:K
            fit_vec[i] = count_vec[i] * log(count_vec[i] / widths[i]) - prior
        end
        for i = 2:K
            fit_vec[i] += best[i-1]
        end

        i_max = 1
        best_val = fit_vec[1]
        for i = 2:K
            if fit_vec[i] > best_val
                best_val = fit_vec[i]
                i_max = i
            end
        end
        last[K] = i_max
        best[K] = best_val
    end

    change_points = zeros(Int64, n)
    i_cp = n + 1
    ind = n + 1
    while true
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 1
            break
        end
        ind = last[ind-1]
    end
    change_points = change_points[i_cp:end]
    return edges[change_points]
end

function float_bits(values::AbstractVector{Float64})
    return collect(reinterpret(UInt64, values))
end

@testset "Bayesian blocks prefix-count equivalence" begin
    cases = Vector{Float64}[
        [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 8.0, 13.0],
        [-4.0, -2.0, -2.0, -1.0, 0.0, 0.0, 0.25, 0.5, 2.0, 9.0],
        [0.0, 0.0, 1.0e-12, 2.0e-12, 0.1, 0.1, 1.0, 10.0, 100.0],
        collect(range(-3.0, 5.0; length = 24)),
    ]

    # Deterministic zero-inflated pseudo-random cases without adding a test
    # dependency on Random.jl.
    for seed = 1:20
        state = UInt64(seed)
        values = Vector{Float64}(undef, 64)
        for i = eachindex(values)
            state = state * 6364136223846793005 + 1442695040888963407
            value = Float64(Int(state % UInt64(2001)) - 1000) / 1000
            values[i] = state % UInt64(5) == 0 ? 0.0 : value
        end
        push!(cases, values)
    end

    algorithm = FastPIDC.DiscretizeBayesianBlocks()
    for values in cases
        reference_edges = bayesian_blocks_reference(values)
        optimized_edges = FastPIDC.binedges(algorithm, values)

        @test optimized_edges == reference_edges
        @test float_bits(optimized_edges) == float_bits(reference_edges)

        reference_ids = FastPIDC.encode(
            FastPIDC.LinearDiscretizer(reference_edges),
            values,
        )
        optimized_ids = FastPIDC.encode(
            FastPIDC.LinearDiscretizer(optimized_edges),
            values,
        )
        @test optimized_ids == reference_ids
    end
end

@testset "Typed Node constructor preserves legacy results" begin
    values = [0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 3.5, 8.0, 8.0, 13.0]

    # Reproduce the mixed-type row accepted by the legacy constructor.
    legacy_line = Matrix{Any}(undef, 1, length(values) + 1)
    legacy_line[1, 1] = "G1"
    legacy_line[1, 2:end] .= values

    typed_node = Node("G1", values, "bayesian_blocks", "maximum_likelihood", 10)
    legacy_node = Node(legacy_line, "bayesian_blocks", "maximum_likelihood", 10)

    @test typed_node.label == legacy_node.label
    @test typed_node.number_of_bins == legacy_node.number_of_bins
    @test typed_node.binned_values == legacy_node.binned_values
    @test typed_node.probabilities == legacy_node.probabilities
    @test float_bits(typed_node.probabilities) == float_bits(legacy_node.probabilities)
end


@testset "Bayesian blocks backend configuration" begin
    @test PIDCConfig().bb_backend == :cuda
    @test PIDCConfig(bb_backend = :cpu).bb_backend == :cpu
    @test_throws ArgumentError PIDCConfig(bb_backend = :invalid)
end
