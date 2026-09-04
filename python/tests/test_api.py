import numpy as np
import pytest

from fastpidc import PIDCConfig, infer_network
from fastpidc.network import MINetworkInference, PIDCNetworkInference


@pytest.fixture
def text_data_file(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("header ignored\nA 0.1 0.2 0.9 0.8 0.1 0.9\nB 0.9 0.8 0.1 0.2 0.9 0.1\nC 0.5 0.4 0.6 0.5 0.4 0.6\n")
    return path


def test_infer_network_writes_tsv(tmp_path, text_data_file):
    out_path = tmp_path / "edges.tsv"
    network = infer_network(
        str(text_data_file),
        PIDCNetworkInference(),
        discretizer="uniform_width",
        number_of_bins=2,
        out_file_path=str(out_path),
        config=PIDCConfig(backend="cpu"),
    )
    assert len(network.edges) == 3
    lines = out_path.read_text().splitlines()
    assert len(lines) == 6  # 3 edges x 2 directions


def test_infer_network_writes_npy(tmp_path, text_data_file):
    out_path = tmp_path / "edges.npy"
    infer_network(
        str(text_data_file),
        MINetworkInference(),
        discretizer="uniform_width",
        number_of_bins=2,
        out_file_path=str(out_path),
        output_format="npy",
        config=PIDCConfig(backend="cpu"),
    )
    matrix = np.load(out_path)
    assert matrix.shape == (3, 3)


def test_infer_network_no_file_written_by_default(tmp_path, text_data_file):
    network = infer_network(str(text_data_file), MINetworkInference(), discretizer="uniform_width", number_of_bins=2)
    assert len(network.nodes) == 3
    assert list(tmp_path.iterdir()) == [text_data_file]


def test_infer_network_rejects_unknown_output_format(text_data_file):
    with pytest.raises(ValueError, match="output_format"):
        infer_network(str(text_data_file), MINetworkInference(), output_format="xml")
