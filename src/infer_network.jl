# Helper functions for inferring a network from a data file

function _validate_bb_backend(bb_backend::Symbol)
    bb_backend in (:cpu, :cuda) ||
        throw(ArgumentError("bb_backend must be :cpu or :cuda, got :$bb_backend"))
    return bb_backend
end


function _bayesian_blocks_cuda_available()
    return hasmethod(bayesian_blocks_cuda_available, Tuple{}) &&
           bayesian_blocks_cuda_available()
end

function _solve_bayesian_blocks_batch(
    problems::Vector{BayesianBlocksProblem};
    bb_backend::Symbol,
    verbose::Bool,
)
    _validate_bb_backend(bb_backend)
    isempty(problems) && return BayesianBlocksSolution[]

    if bb_backend == :cuda
        if _bayesian_blocks_cuda_available() &&
           hasmethod(solve_bayesian_blocks_cuda, (typeof(problems), Bool))
            solutions = solve_bayesian_blocks_cuda(problems, verbose)
            solutions !== nothing && return solutions
        end

        @warn "CUDA Bayesian blocks requested, but no functional CUDA GPU was found. Falling back to the CPU reference implementation."
    end

    solutions = Vector{BayesianBlocksSolution}(undef, length(problems))
    Threads.@threads for i = eachindex(problems)
        solutions[i] = solve_bayesian_blocks_cpu(problems[i])
    end
    return solutions
end

function _node_from_bayesian_solution(
    label::AbstractString,
    values::Vector{Float64},
    problem::BayesianBlocksProblem,
    solution::BayesianBlocksSolution,
    estimator,
)
    edges = problem.edges[solution.change_points]
    binned_values = encode(LinearDiscretizer(edges), values)
    number_of_bins = length(edges) - 1
    probabilities = get_probabilities(
        estimator,
        get_frequencies_from_bin_ids(binned_values, number_of_bins),
    )
    return Node(String(label), binned_values, number_of_bins, probabilities)
end

"""
    _build_nodes(labels, value_at; ...)

Construct nodes from a callable returning one gene's observations. Bayesian
blocks are prepared once on the CPU, then solved as a batch by the selected
backend. Quantization or other input transformations remain upstream of
FastPIDC; this function operates on the values exactly as supplied.
"""
function _build_nodes(
    labels::AbstractVector,
    value_at;
    discretizer,
    estimator,
    number_of_bins,
    bb_backend::Symbol,
    verbose::Bool,
)
    number_of_nodes = length(labels)
    _validate_bb_backend(bb_backend)

    if discretizer == "bayesian_blocks" && bb_backend == :cuda &&
       !_bayesian_blocks_cuda_available()
        @warn "CUDA Bayesian blocks requested, but no functional CUDA GPU was found. Falling back to the CPU reference implementation."
        bb_backend = :cpu
    end

    if discretizer != "bayesian_blocks" || bb_backend == :cpu
        # Keep the CPU reference on the original per-gene construction path.
        # This avoids retaining every gene's prepared unique-value arrays in
        # memory when batching is unnecessary.
        nodes = Array{Node}(undef, number_of_nodes)
        Threads.@threads for i = 1:number_of_nodes
            nodes[i] = Node(
                string(labels[i]),
                collect(Float64, vec(value_at(i))),
                discretizer,
                estimator,
                number_of_bins,
            )
        end
        return nodes
    end

    preparation_start_ns = time_ns()

    # Sorting and unique-value compression are shared by the CPU and CUDA
    # solvers. The resulting U_g values are also all the CUDA backend needs to
    # form lightweight workload buckets; no extra scan of the expression data
    # is required for bucket selection.
    problems = Vector{Union{Nothing,BayesianBlocksProblem}}(undef, number_of_nodes)
    fallback = zeros(UInt8, number_of_nodes)
    Threads.@threads for i = 1:number_of_nodes
        values = collect(Float64, vec(value_at(i)))
        try
            problems[i] = prepare_bayesian_blocks(values)
        catch
            problems[i] = nothing
            fallback[i] = 1
        end
    end

    preparation_seconds = (time_ns() - preparation_start_ns) / 1.0e9
    if verbose
        unique_counts = sort!([
            problem === nothing ? 0 : length(problem.prefix_counts) for problem in problems
        ])
        nonzero_counts = [u for u in unique_counts if u != 0]
        if !isempty(nonzero_counts)
            percentile_index(p) = clamp(ceil(Int, p * length(nonzero_counts)), 1, length(nonzero_counts))
            candidate_work = UInt128(0)
            for u in nonzero_counts
                candidate_work += UInt128(u) * UInt128(u + 1) ÷ UInt128(2)
            end
            println(
                "[FastPIDC] Bayesian-block preparation: " *
                "$(round(preparation_seconds; digits = 2)) s",
            )
            println(
                "[FastPIDC] Bayesian-block U_g: min=$(first(nonzero_counts)), " *
                "median=$(nonzero_counts[percentile_index(0.50)]), " *
                "p90=$(nonzero_counts[percentile_index(0.90)]), " *
                "p99=$(nonzero_counts[percentile_index(0.99)]), " *
                "max=$(last(nonzero_counts)); candidate evaluations=$candidate_work",
            )
        end
    end

    active_indices = Int[]
    active_problems = BayesianBlocksProblem[]
    for i = 1:number_of_nodes
        problem = problems[i]
        if fallback[i] == 0 && problem !== nothing && length(problem.prefix_counts) > 1
            push!(active_indices, i)
            push!(active_problems, problem)
        end
    end

    solve_start_ns = time_ns()
    active_solutions = _solve_bayesian_blocks_batch(
        active_problems;
        bb_backend = bb_backend,
        verbose = verbose,
    )
    solve_seconds = (time_ns() - solve_start_ns) / 1.0e9
    verbose && println(
        "[FastPIDC] Bayesian-block dynamic program ($bb_backend): " *
        "$(round(solve_seconds; digits = 2)) s",
    )

    solutions = Vector{Union{Nothing,BayesianBlocksSolution}}(undef, number_of_nodes)
    fill!(solutions, nothing)
    for (i, solution) in zip(active_indices, active_solutions)
        solutions[i] = solution
    end
    empty!(active_problems)

    encoding_start_ns = time_ns()
    nodes = Array{Node}(undef, number_of_nodes)
    Threads.@threads for i = 1:number_of_nodes
        values = collect(Float64, vec(value_at(i)))
        problem = problems[i]
        solution = solutions[i]

        if fallback[i] != 0 || problem === nothing
            nodes[i] = Node(
                string(labels[i]),
                values,
                "uniform_width",
                estimator,
                number_of_bins,
            )
            println("Bayesian blocks failed for $(labels[i]), fell back to uniform width")
        elseif length(problem.prefix_counts) == 1
            binned_values = ones(Int, length(values))
            probabilities = get_probabilities(
                estimator,
                get_frequencies_from_bin_ids(binned_values, 1),
            )
            nodes[i] = Node(String(labels[i]), binned_values, 1, probabilities)
        else
            nodes[i] = _node_from_bayesian_solution(
                string(labels[i]),
                values,
                problem,
                solution::BayesianBlocksSolution,
                estimator,
            )
        end
        problems[i] = nothing
    end

    encoding_seconds = (time_ns() - encoding_start_ns) / 1.0e9
    verbose && println(
        "[FastPIDC] Bayesian-block bin encoding: " *
        "$(round(encoding_seconds; digits = 2)) s",
    )

    return nodes
end

# Entry point: checks extension and routes to the right loader

"""
    get_nodes(data_file_path::String; <keyword arguments>) -> Vector{Node}

Gets an array of all [`Node`](@ref)s from a data file. Dispatches to
[`get_nodes_h5`](@ref) for `.h5` files, or to the whitespace/delimited text
loader otherwise (see the text-file method below for its format and
keyword arguments).
"""
function get_nodes(
    data_file_path::String;
    delim::Union{Char,Bool} = false,
    discretizer = "bayesian_blocks",
    estimator = "maximum_likelihood",
    number_of_bins = 10,
    bb_backend::Symbol = :cuda,
    verbose::Bool = false,
)
    if endswith(data_file_path, ".h5")
        return get_nodes_h5(
            data_file_path;
            discretizer,
            estimator,
            number_of_bins,
            bb_backend,
            verbose,
        )
    else
        return get_nodes_text(
            data_file_path;
            delim,
            discretizer,
            estimator,
            number_of_bins,
            bb_backend,
            verbose,
        )
    end
end

"""
    get_nodes_h5(data_file_path::String; <keyword arguments>) -> Vector{Node}

Gets an array of all [`Node`](@ref)s from an HDF5 (`.h5`) expression file.

The file must contain a `"gene_names"` dataset, and an expression matrix
under one of `"matrix_sparse_csc"`, `"matrix_dense"`, `"X"`, `"matrix"` or
`"data"` (checked in that order). The matrix may be a dense HDF5 dataset
(assumed to be `(genes, cells)` in C order, as written by
[AnnData](https://anndata.readthedocs.io/)/scanpy, and transposed on load)
or an HDF5 group holding a CSC sparse matrix (`"data"`, `"indices"`,
`"indptr"` datasets plus a `"shape"` attribute, using Python's 0-based
indexing, which is converted to Julia's 1-based indexing on load). In both
cases the on-disk layout is `(cells, genes)` after loading.

# Arguments
* `data_file_path`: path to the `.h5` file.
* `discretizer="bayesian_blocks"`: algorithm for discretizing the data.
* `estimator="maximum_likelihood"`: algorithm for estimating probabilities.
* `number_of_bins=10`: will be overwritten if using `"bayesian_blocks"`.

Throws an `ArgumentError` if the expected datasets, matrix key or matrix
group members/attributes are missing.
"""
function get_nodes_h5(
    data_file_path::String;
    discretizer = "bayesian_blocks",
    estimator = "maximum_likelihood",
    number_of_bins = 10,
    bb_backend::Symbol = :cuda,
    verbose::Bool = false,
)
    nodes = Node[]
    
    h5open(data_file_path, "r") do f
        if !haskey(f, "gene_names")
            throw(ArgumentError("Invalid HDF5 schema in $(data_file_path). Missing required dataset: 'gene_names'"))
        end
        gene_names = String.(read(f["gene_names"]))
        number_of_nodes = length(gene_names)
        
        matrix_key = ""
        for key in ["matrix_sparse_csc", "matrix_dense", "X", "matrix", "data"]
            if haskey(f, key)
                matrix_key = key
                break
            end
        end
        
        if matrix_key == ""
            throw(ArgumentError("Could not find expression data. Expected key 'X' or similar."))
        end
        
        data_obj = f[matrix_key]
        
        # Determine Dense vs Sparse dynamically
        if isa(data_obj, HDF5.Group)
            # Validate Datasets
            if !all(k -> haskey(data_obj, k), ["data", "indices", "indptr"])
                throw(ArgumentError("Sparse matrix group '$matrix_key' is missing required datasets: 'data', 'indices', or 'indptr'."))
            end

            # Validate Attributes
            if !haskey(attributes(data_obj), "shape")
                throw(ArgumentError("Sparse matrix group '$matrix_key' is missing the required attribute: 'shape'."))
            end
            
            shape = tuple(read_attribute(data_obj, "shape")...)
            
            # Since Python saved a (Cells, Genes) CSC matrix, we can feed the 
            # 1-indexed components directly into Julia. The resulting matrix 
            # will be structurally identical: (Cells, Genes)
            X_raw = SparseMatrixCSC(
                shape[1], shape[2], 
                read(data_obj["indptr"]) .+ 1, 
                read(data_obj["indices"]) .+ 1, 
                read(data_obj["data"])
            )
            
        elseif isa(data_obj, HDF5.Dataset)
            # Python saved (Cells, Genes) C-Order. Julia reads (Genes, Cells).
            # We permute dimensions to make it (Cells, Genes) so our slicing logic is uniform.
            X_raw = permutedims(read(data_obj))
        else
            throw(ArgumentError("Object at '$matrix_key' is neither an HDF5 Group nor a Dataset."))
        end
        
        # Since X_raw is (Cells, Genes), column `i` is gene `i`.
        value_at = i -> (@view X_raw[:, i])
        nodes = _build_nodes(
            gene_names,
            value_at;
            discretizer,
            estimator,
            number_of_bins,
            bb_backend,
            verbose,
        )
    end
    
    return nodes
end

# --- Legacy text Loader (Preserved for backwards compatibility) ---

"""
    get_nodes(data_file_path::String; <keyword arguments>)

Gets an array of all Nodes from a data file. It is assumed that the first
line of the file is headers (which are discarded) and the subsequent lines
each represent one node, and are of the form:

Label    data_value1  data_value2 ...

though a different delimiter may be specified.

Arguments:
* `data_file_path`: path to the data file
* `delim=false`: the file's delimiter. Leave as false if it is whitespace
* `discretizer="bayesian_blocks"`: algorithm for discretizing the data
* `estimator="maximum_likelihood"`: algorithm for estimating probabilities
* `number_of_bins=10`: will be overwritten if using "bayesian_blocks"
* `bb_backend=:cuda`: Bayesian-block dynamic-program backend (`:cuda` or `:cpu`)
* `verbose=false`: print Bayesian-block phase, workload, and bucket diagnostics

The "maximum_likelihood" estimator is recommended for PUC and PIDC.
"""
function get_nodes_text(
    data_file_path::String;
    delim::Union{Char,Bool} = false,
    discretizer = "bayesian_blocks",
    estimator = "maximum_likelihood",
    number_of_bins = 10,
    bb_backend::Symbol = :cuda,
    verbose::Bool = false,
)
    lines = open(data_file_path) do io
        if delim == false
            readdlm(io; skipstart = 1)
        else
            readdlm(io, delim; skipstart = 1)
        end
    end
    
    labels = string.(lines[:, 1])
    value_at = i -> (@view lines[i, 2:end])
    return _build_nodes(
        labels,
        value_at;
        discretizer,
        estimator,
        number_of_bins,
        bb_backend,
        verbose,
    )
end


"""
    write_network_file(file_path::String, inferred_network::InferredNetwork)

Writes a network file from an InferredNetwork type. Each line of the file
will contain an edge, and since networks are assumed undirected, each edge
will be written in both directions with the same weight:

...

LabelX   LabelY  WeightXY

LabelY   LabelX  WeightXY

...

Arguments:
* `file_path`: path to the output file
* `inferred_network`: network that was inferred
"""
function write_network_file(file_path::String, inferred_network::InferredNetwork)

    open(file_path, "w") do out_file
        for edge in inferred_network.edges
            nodes = edge.nodes
            println(out_file, "$(nodes[1].label)\t$(nodes[2].label)\t$(edge.weight)")
            println(out_file, "$(nodes[2].label)\t$(nodes[1].label)\t$(edge.weight)")
        end
    end

end

"""
    write_network_npy(file_path::String, inferred_network::InferredNetwork)

Writes an inferred undirected weighted network as a dense NumPy binary file (.npy)
plus a sidecar gene list file preserving row/column order.

Outputs:
* `file_path`            : N x N dense weighted adjacency matrix in .npy format (Float32)
* `<stem>_genes.txt`     : one gene label per line, matching matrix row/column order

To load in Python:
    import numpy as np

    A = np.load("network.npy")

    with open("network_genes.txt") as f:
        genes = [line.strip() for line in f]
"""
function write_network_npy(file_path::String, inferred_network::InferredNetwork)

    file_path = _npy_output_path(file_path)

    # Extract node labels and map them to matrix indices
    labels = [String(node.label) for node in inferred_network.nodes]
    n = length(labels)

    labels_to_ids = Dict{String,Int}()
    for (i, label) in enumerate(labels)
        labels_to_ids[label] = i
    end

    # Build symmetric dense adjacency matrix initialized with zeros
    A = zeros(Float32, n, n)

    for edge in inferred_network.edges
        i = labels_to_ids[String(edge.nodes[1].label)]
        j = labels_to_ids[String(edge.nodes[2].label)]
        
        # Cast to Float32 to save disk space and memory footprint
        w = Float32(edge.weight)

        # store both directions for symmetric adjacency
        A[i, j] = w
        A[j, i] = w
    end

    # Write dense NumPy binary file (NPZ.jl handles raw arrays)
    npzwrite(file_path, A)

    # Write matching gene list sidecar
    _write_genes_file(_network_genes_path(file_path), inferred_network.nodes)

    return nothing
end

"""
    read_network_file(file_path::AbstractString)

Reads a network file and creates an InferredNetwork type. Assumes that the input
is such that each line contains an edge and each edge is written in both
directions with the same weight:

...

LabelX   LabelY  WeightXY

LabelY   LabelX  WeightXY

...
"""
function read_network_file(file_path::AbstractString)
    mat = readdlm(file_path)[1:2:end, :]
    edges = []
    nodes = Set()

    for i = 1:size(mat, 1)
        n1_label, n2_label, weight = mat[i, :]
        n1_label = string(n1_label)
        n2_label = string(n2_label)
        n1 = Node(n1_label, [], 0, [])
        n2 = Node(n2_label, [], 0, [])
        new_edge = Edge((n1, n2), weight)
        push!(edges, new_edge)
        push!(nodes, n1_label, n2_label)
    end

    nodes = [Node(n, [], 0, []) for n in nodes]
    return InferredNetwork(nodes, edges)
end

"""
    get_adjacency_matrix(inferred_network::InferredNetwork, threshold = 1.0; <keyword arguments>)

Gets an adjacency matrix given an InferredNetwork and a threshold.

Arguments:
* `inferred_network`: network that was inferred
* `threshold=0.1`: threshold above which to keep edges in the network
* `absolute=false`: interpret threshold as an absolute confidence score

If `absolute` is false, threshold will be interpreted as the percentage of edges to keep.
"""
function get_adjacency_matrix(
    inferred_network::InferredNetwork,
    threshold = 0.1;
    absolute = false,
)

    number_of_nodes = length(inferred_network.nodes)
    adjacency_matrix = zeros(Bool, (number_of_nodes, number_of_nodes))

    labels_to_ids = Dict(node.label => i for (i, node) in enumerate(inferred_network.nodes))
    ids_to_labels = Dict(i => node.label for (i, node) in enumerate(inferred_network.nodes))

    number_of_edges =
        absolute ? findfirst(x -> x.weight < threshold, inferred_network.edges) - 1 :
        Int(round(length(inferred_network.edges) * threshold))

    for edge in inferred_network.edges[1:number_of_edges]
        node1 = labels_to_ids[edge.nodes[1].label]
        node2 = labels_to_ids[edge.nodes[2].label]
        adjacency_matrix[node1, node2] = true
        adjacency_matrix[node2, node1] = true
    end

    return adjacency_matrix, labels_to_ids, ids_to_labels

end

"""
    infer_network(data_file_path::String, inference::AbstractNetworkInference; <keyword arguments>)

Infers a network, given a data file and a network inference algorithm. It
is assumed that the first line of the file is headers (which are
discarded) and the subsequent lines each represent one node, and are of
the form:

Label    data_value1  data_value2 ...

though a different delimiter may be specified.

Arguments:
* `data_file_path`: path to the data file
* `inference`: network inference algorithm (e.g. `PIDCNetworkInference()`)
* `delim=false`: the file's delimiter. Leave as false if it is whitespace
* `discretizer="bayesian_blocks"`: algorithm for discretizing the data
* `estimator="maximum_likelihood"`: algorithm for estimating probabilities
* `number_of_bins=10`: will be overwritten if using "bayesian_blocks"
* `config.bb_backend`: Bayesian-block backend used while constructing nodes
* `base=2`: base for the information measures
* `out_file_path=""`: path to output file. If empty, will not write a file

The "maximum_likelihood" estimator is recommended for PUC and PIDC.
"""
function infer_network(
    data_file_path::String,
    inference::AbstractNetworkInference;
    delim::Union{Char,Bool} = false,
    discretizer = "bayesian_blocks",
    estimator = "maximum_likelihood",
    number_of_bins = 10,
    base = 2,
    out_file_path = "",
    output_format::Symbol = :tsv,
    config::PIDCConfig = PIDCConfig(),
)

    println("Getting nodes...")
    nodes = get_nodes(
        data_file_path,
        delim = delim,
        discretizer = discretizer,
        estimator = estimator,
        number_of_bins = number_of_bins,
        bb_backend = config.bb_backend,
        verbose = config.verbose,
    )

    println("Inferring network...")
    inferred_network = InferredNetwork(
        inference,
        nodes,
        estimator = estimator,
        base = base,
        config = config,
    )

    if length(out_file_path) > 1
        println("Writing network to file...")

        if output_format == :tsv
            write_network_file(out_file_path, inferred_network)
        elseif output_format == :npy
            write_network_npy(out_file_path, inferred_network)
        else
            error("Unsupported output_format=$(output_format). Use :tsv or :npy.")
        end
    end

    return inferred_network

end
