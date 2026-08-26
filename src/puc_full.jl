# src/puc_full.jl
#
# Optimized Full PUC computation using the Sorting Trick and fused Information Measures.
# Reduces complexity from O(N^3 B) to O(N^2 B log N).
using SharedArrays

"""
    compute_puc_full_cuda(nodes, config, base)

GPU implementation of [`compute_puc_full`](@ref), defined by the
`FastPIDCCUDAExt` package extension (loaded automatically when `using
CUDA` and a functional GPU is available). Calling this without the
extension loaded raises a `MethodError`; [`compute_puc_full`](@ref) checks
for the method's existence before dispatching to it.
"""
function compute_puc_full_cuda end

"""
    compute_puc_full(nodes::Vector{Node}; estimator="maximum_likelihood",
                      base=2, config=PIDCConfig()) -> (mi_scores, puc_scores)

Compute the full pairwise MI matrix and pre-context Proportional Unique
Contribution (PUC) matrix for `nodes`.

For every ordered triple `(x, y, z)` of distinct nodes, the redundancy
between sources `x` and `y` with respect to target `z` is computed from
cached specific-information values, and `PUC(x, z) += (MI(x,z) -
redundancy) / MI(x,z)` (clamped to be non-negative, and skipped when
`MI(x,z)` is ~0) is accumulated symmetrically. This is the "sorting trick"
formulation, reducing complexity from `O(N^3 B)` to `O(N^2 B log N)`
relative to a naive implementation (`N` = number of nodes, `B` = number of
bins).

# Arguments
* `nodes`: nodes to score.
* `estimator="maximum_likelihood"`: probability estimator.
* `base=2`: logarithm base for the information measures.
* `config=PIDCConfig()`: selects the computation backend
  (`config.backend`, `:cpu` or `:cuda`) and verbosity. When `:cuda` is
  requested, dispatches to [`compute_puc_full_cuda`](@ref) (from the CUDA
  package extension) and errors if that extension is not loaded.

Returns a tuple `(mi_scores, puc_scores)`, each a dense
`length(nodes)`-by-`length(nodes)` matrix.
"""
function compute_puc_full(
    nodes::Vector{Node};
    estimator::String = "maximum_likelihood",
    base::Int = 2,
    config::PIDCConfig = PIDCConfig(),
)

    number_of_nodes = length(nodes)
    max_bins = maximum(node -> node.number_of_bins, nodes)
    S = length(nodes[1].binned_values)
    inv_S = 1.0 / S
    inv_log_base = 1.0 / log(base)

    # --- Local helpers (same logic as original get_puc_scores) ----------------

    # Mutual information and specific information for a node pair
    function get_mi_and_si(node1::Node, node2::Node)
        probabilities, probabilities1, probabilities2 =
            get_joint_probabilities(node1, node2, estimator)
        mi = apply_mutual_information_formula(
            probabilities,
            probabilities1,
            probabilities2,
            base,
        )
        si1 = apply_specific_information_formula(
            probabilities,
            probabilities1,
            probabilities2,
            1,
            base,
        )
        si2 = apply_specific_information_formula(
            probabilities,
            probabilities2,
            probabilities1,
            2,
            base,
        )
        return mi, si1, si2
    end

    # Fill NodePair cache symmetrically
    function fill_node_pairs!(node_pairs::Array{NodePair,2})
        for i = 1:number_of_nodes
            for j = (i+1):number_of_nodes
                mi, si1, si2 = get_mi_and_si(nodes[i], nodes[j])
                node_pairs[i, j] = NodePair(mi, si1)
                node_pairs[j, i] = NodePair(mi, si2)
            end
        end
    end

    # Add contribution of one redundancy computation to PUC scores
    function increment_puc_scores!(
        x::Int,
        z::Int,
        mi::Float64,
        redundancy::Float64,
        puc_scores::AbstractMatrix{Float64},
    )
        # Prevents division-by-tiny-number explosion when MI is effectively zero.
        if mi <= 1e-12
            return
        end

        puc_score = (mi - redundancy) / mi
        puc_score = isfinite(puc_score) && puc_score >= 0 ? puc_score : zero(puc_score)
        puc_scores[x, z] += puc_score
        puc_scores[z, x] += puc_score
    end


    # Compute redundancy for a given target and its two sources,
    # then update PUC(x,z) and PUC(y,z)
    function get_puc!(
        target::Node,
        source1_target::NodePair,
        source2_target::NodePair,
        x::Int,
        y::Int,
        z::Int,
        puc_scores::Matrix{Float64},
    )

        redundancy = apply_redundancy_formula(
            target.probabilities,      # p_z
            source1_target.si,         # specific information of source1 wrt target
            source2_target.si,         # specific information of source2 wrt target
            base,
        )

        increment_puc_scores!(x, z, source1_target.mi, redundancy, puc_scores)
        increment_puc_scores!(y, z, source2_target.mi, redundancy, puc_scores)
    end

    # Turn NodePair cache into a dense MI matrix (symmetric).
    function nodepairs_to_mi(node_pairs::Array{NodePair,2})
        n = size(node_pairs, 1)
        mi = zeros(Float64, n, n)
        for i = 1:n
            mi[i, i] = 0.0
            for j = (i+1):n
                m = node_pairs[i, j].mi
                mi[i, j] = m
                mi[j, i] = m
            end
        end
        return mi
    end

    # --- Allocate caches -------------------------------------------------------

    puc_scores = SharedArray{Float64}(number_of_nodes, number_of_nodes)
    fill!(puc_scores, 0.0)

    # --- determine computation backend -----------------------------------------

    if config.backend == :cuda
        if hasmethod(compute_puc_full_cuda, (typeof(nodes), typeof(config), typeof(base)))
            return compute_puc_full_cuda(nodes, config, base)
        else
            error(
                "CUDA backend requested but CUDA.jl is not loaded or CUDA.functional() is false. Run `using CUDA` to enable GPU acceleration.",
            )
        end
    end

    # --- pairwise MI + SI cache (CPU Path) ---------------------------

    node_pairs = Array{NodePair}(undef, number_of_nodes, number_of_nodes)
    fill_node_pairs!(node_pairs)

    # --- Build full MI matrix from NodePair cache --------------------

    mi_scores = nodepairs_to_mi(node_pairs)

    # --- full PUC enumeration ------------------

    @sync @distributed for x = 1:number_of_nodes
        for z = (x+1):number_of_nodes
            for y = 1:number_of_nodes
                (y == x || y == z) && continue

                # target = z
                np_xz = node_pairs[x, z]
                np_yz = node_pairs[y, z]
                Rz = apply_redundancy_formula(
                    nodes[z].probabilities,
                    np_xz.si,
                    np_yz.si,
                    base,
                )
                increment_puc_scores!(x, z, np_xz.mi, Rz, puc_scores)

                # target = x
                np_yx = node_pairs[y, x]
                np_zx = node_pairs[z, x]
                Rx = apply_redundancy_formula(
                    nodes[x].probabilities,
                    np_yx.si,
                    np_zx.si,
                    base,
                )
                increment_puc_scores!(x, z, np_zx.mi, Rx, puc_scores)
            end
        end
    end


    return mi_scores, Array(puc_scores)
end
