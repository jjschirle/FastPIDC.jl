import h5py
import numpy as np
import pytest
from scipy import sparse

from fastpidc.io import (
    get_adjacency_matrix,
    get_nodes,
    read_network_file,
    write_network_file,
    write_network_npy,
)
from fastpidc.network import PIDCNetworkInference, infer_network_from_nodes
from fastpidc.types import PIDCConfig


@pytest.fixture
def text_data_file(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("header ignored\nA 0.1 0.2 0.9 0.8 0.1 0.9\nB 0.9 0.8 0.1 0.2 0.9 0.1\nC 0.5 0.4 0.6 0.5 0.4 0.6\n")
    return path


def test_get_nodes_text_basic(text_data_file):
    nodes = get_nodes(str(text_data_file), discretizer="uniform_width", number_of_bins=2)
    assert [n.label for n in nodes] == ["A", "B", "C"]
    assert all(n.number_of_bins == 2 for n in nodes)
    assert nodes[0].binned_values.size == 6


def test_get_nodes_text_explicit_delimiter(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("header\nA,1,2,3,4,5,6,7,8\nB,8,7,6,5,4,3,2,1\n")
    nodes = get_nodes(str(path), delim=",", discretizer="uniform_width", number_of_bins=2)
    assert [n.label for n in nodes] == ["A", "B"]


def test_get_nodes_h5_dense(tmp_path):
    path = tmp_path / "data.h5"
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4))  # (cells, genes)
    with h5py.File(path, "w") as f:
        f.create_dataset("gene_names", data=[f"G{i}".encode() for i in range(4)])
        f.create_dataset("matrix_dense", data=x)

    nodes = get_nodes(str(path), discretizer="uniform_width", number_of_bins=3)
    assert [n.label for n in nodes] == ["G0", "G1", "G2", "G3"]
    np.testing.assert_allclose(nodes[0].probabilities.sum(), 1.0)


def test_get_nodes_h5_sparse_matches_dense(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.poisson(2, size=(60, 5)).astype(np.float64)
    xs = sparse.csc_matrix(x)

    sparse_path = tmp_path / "sparse.h5"
    with h5py.File(sparse_path, "w") as f:
        f.create_dataset("gene_names", data=[f"G{i}".encode() for i in range(5)])
        grp = f.create_group("matrix_sparse_csc")
        grp.create_dataset("data", data=xs.data)
        grp.create_dataset("indices", data=xs.indices.astype(np.int64))
        grp.create_dataset("indptr", data=xs.indptr.astype(np.int64))
        grp.attrs["shape"] = np.array(xs.shape, dtype=np.int64)

    dense_path = tmp_path / "dense.h5"
    with h5py.File(dense_path, "w") as f:
        f.create_dataset("gene_names", data=[f"G{i}".encode() for i in range(5)])
        f.create_dataset("matrix_dense", data=x)

    sparse_nodes = get_nodes(str(sparse_path))
    dense_nodes = get_nodes(str(dense_path))
    for s, d in zip(sparse_nodes, dense_nodes):
        assert s.number_of_bins == d.number_of_bins
        np.testing.assert_array_equal(s.binned_values, d.binned_values)


def test_get_nodes_h5_missing_gene_names_raises(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix_dense", data=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="gene_names"):
        get_nodes(str(path))


def test_write_and_read_network_file_roundtrip(tmp_path, text_data_file):
    nodes = get_nodes(str(text_data_file), discretizer="uniform_width", number_of_bins=2)
    network = infer_network_from_nodes(PIDCNetworkInference(), nodes, config=PIDCConfig(backend="cpu"))

    out_path = tmp_path / "edges.tsv"
    write_network_file(str(out_path), network)
    reloaded = read_network_file(str(out_path))

    original_weights = {frozenset((e.nodes[0].label, e.nodes[1].label)): e.weight for e in network.edges}
    reloaded_weights = {frozenset((e.nodes[0].label, e.nodes[1].label)): e.weight for e in reloaded.edges}
    assert original_weights.keys() == reloaded_weights.keys()
    for key, weight in original_weights.items():
        assert reloaded_weights[key] == pytest.approx(weight)


def test_write_network_npy(tmp_path, text_data_file):
    nodes = get_nodes(str(text_data_file), discretizer="uniform_width", number_of_bins=2)
    network = infer_network_from_nodes(PIDCNetworkInference(), nodes, config=PIDCConfig(backend="cpu"))

    out_path = tmp_path / "edges.npy"
    write_network_npy(str(out_path), network)

    matrix = np.load(out_path)
    genes = (tmp_path / "edges_genes.txt").read_text().splitlines()
    assert genes == [n.label for n in network.nodes]
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(len(nodes)))


def test_get_adjacency_matrix_relative_threshold(text_data_file):
    nodes = get_nodes(str(text_data_file), discretizer="uniform_width", number_of_bins=2)
    network = infer_network_from_nodes(PIDCNetworkInference(), nodes, config=PIDCConfig(backend="cpu"))
    adjacency, labels_to_ids, ids_to_labels = get_adjacency_matrix(network, 1.0 / 3)
    assert adjacency.sum() == 2  # one edge kept, symmetric -> two True entries
    assert set(labels_to_ids) == {n.label for n in nodes}
    assert ids_to_labels[labels_to_ids["A"]] == "A"


def test_get_adjacency_matrix_absolute_threshold(text_data_file):
    nodes = get_nodes(str(text_data_file), discretizer="uniform_width", number_of_bins=2)
    network = infer_network_from_nodes(PIDCNetworkInference(), nodes, config=PIDCConfig(backend="cpu"))
    top_weight = network.edges[0].weight
    adjacency, _, _ = get_adjacency_matrix(network, top_weight, absolute=True)
    assert adjacency.sum() == 2
