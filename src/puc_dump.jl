# Binary pre-context PUC score output.

"""
    dump_puc_scores(scores, nodes, config)

Write the full symmetric pre-context PUC score matrix to `<stem>_puc.npy` as
Float64. The matrix uses the same `<stem>_genes.txt` row/column sidecar as the
final network output.
"""
function dump_puc_scores(
    scores::AbstractMatrix{Float64},
    nodes::AbstractVector{Node},
    config::PIDCConfig,
)
    path = config.dump_puc_path
    path === nothing && return nothing

    return _dump_score_matrix(scores, nodes, path, :puc)
end
