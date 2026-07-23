# Binary diagnostic score output shared by MI and PUC dumps.

function _dump_score_matrix(
    scores::AbstractMatrix{Float64},
    nodes::AbstractVector{Node},
    file_path::AbstractString,
    score_name::Symbol,
)
    n = length(nodes)
    size(scores) == (n, n) ||
        throw(DimensionMismatch("score matrix size $(size(scores)) does not match $n nodes"))

    output_path = _score_output_path(file_path, score_name)
    npzwrite(output_path, scores)
    _write_genes_file(_score_genes_path(output_path, score_name), nodes)

    return output_path
end

"""
    dump_mi_scores(mi_scores, nodes, config)

Write the full symmetric MI score matrix to `<stem>_mi.npy` as Float64. The
matrix uses the same `<stem>_genes.txt` row/column sidecar as the final network
output.
"""
function dump_mi_scores(
    mi_scores::AbstractMatrix{Float64},
    nodes::AbstractVector{Node},
    config::PIDCConfig,
)
    path = config.dump_mi_path
    path === nothing && return nothing

    return _dump_score_matrix(mi_scores, nodes, path, :mi)
end
