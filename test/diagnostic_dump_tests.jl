using Test
using FastPIDC
using NPZ

const DIAGNOSTIC_OUT_DIR = joinpath(dirname(@__FILE__), "baseline_outputs")
isdir(DIAGNOSTIC_OUT_DIR) || mkpath(DIAGNOSTIC_OUT_DIR)

@testset "Binary diagnostic dumps" begin
    output_file = joinpath(DIAGNOSTIC_OUT_DIR, "toy_diagnostics.npy")
    mi_file = joinpath(DIAGNOSTIC_OUT_DIR, "toy_diagnostics_mi.npy")
    puc_file = joinpath(DIAGNOSTIC_OUT_DIR, "toy_diagnostics_puc.npy")
    genes_file = joinpath(DIAGNOSTIC_OUT_DIR, "toy_diagnostics_genes.txt")
    obsolete_sidecars = [
        joinpath(DIAGNOSTIC_OUT_DIR, "toy_diagnostics_mi_genes.txt"),
        joinpath(DIAGNOSTIC_OUT_DIR, "toy_diagnostics_puc_genes.txt"),
    ]

    for path in [output_file, mi_file, puc_file, genes_file, obsolete_sidecars...]
        isfile(path) && rm(path)
    end

    nodes = [
        Node("G1", Int[], 0, Float64[]),
        Node("G2", Int[], 0, Float64[]),
        Node("G3", Int[], 0, Float64[]),
        Node("G4", Int[], 0, Float64[]),
    ]
    mi_scores = [
        0.0 0.9 0.8 0.1
        0.9 0.0 0.7 0.2
        0.8 0.7 0.0 0.3
        0.1 0.2 0.3 0.0
    ]
    puc_scores = [
        0.0 0.15 0.25 0.75
        0.15 0.0 0.35 0.85
        0.25 0.35 0.0 0.95
        0.75 0.85 0.95 0.0
    ]

    cfg = PIDCConfig(
        backend = :cpu,
        dump_mi_path = output_file,
        dump_puc_path = output_file,
    )

    @test !(:dump_mi_fraction in fieldnames(PIDCConfig))
    @test !(:dump_puc_fraction in fieldnames(PIDCConfig))

    FastPIDC.dump_mi_scores(mi_scores, nodes, cfg)
    FastPIDC.dump_puc_scores(puc_scores, nodes, cfg)

    @test isfile(mi_file)
    @test isfile(puc_file)
    @test isfile(genes_file)
    @test all(path -> !isfile(path), obsolete_sidecars)

    loaded_mi = npzread(mi_file)
    loaded_puc = npzread(puc_file)
    @test eltype(loaded_mi) == Float64
    @test eltype(loaded_puc) == Float64
    @test loaded_mi == mi_scores
    @test loaded_puc == puc_scores
    @test readlines(genes_file) == [node.label for node in nodes]

    # The final network uses the same sidecar rather than creating a second copy.
    edges = Edge[]
    for i = 1:(length(nodes)-1)
        for j = (i+1):length(nodes)
            push!(edges, Edge((nodes[i], nodes[j]), mi_scores[i, j]))
        end
    end
    write_network_npy(output_file, InferredNetwork(nodes, edges))

    @test isfile(output_file)
    @test readlines(genes_file) == [node.label for node in nodes]
    @test all(path -> !isfile(path), obsolete_sidecars)
end
