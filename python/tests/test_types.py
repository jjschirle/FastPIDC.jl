import numpy as np
import pytest

from fastpidc.types import Node, PIDCConfig, network_genes_path, score_genes_path, score_output_path


def test_pidc_config_rejects_bad_backend():
    with pytest.raises(ValueError, match="backend"):
        PIDCConfig(backend="tpu")


def test_pidc_config_rejects_bad_bb_backend():
    with pytest.raises(ValueError, match="bb_backend"):
        PIDCConfig(bb_backend="tpu")


def test_pidc_config_defaults():
    config = PIDCConfig()
    assert config.backend == "cuda"
    assert config.bb_backend == "cuda"
    assert config.discretizer == "bayesian_blocks"
    assert config.estimator == "maximum_likelihood"


def test_node_from_raw_values():
    node = Node.from_raw_values("A", np.arange(10.0), "uniform_width", "maximum_likelihood", 5)
    assert node.label == "A"
    assert node.number_of_bins == 5
    assert node.probabilities.sum() == pytest.approx(1.0)


def test_score_output_path_avoids_double_suffix():
    assert score_output_path("out.tsv", "mi") == "out_mi.npy"
    assert score_output_path("out_mi.tsv", "mi") == "out_mi.npy"


def test_score_output_path_rejects_bad_name():
    with pytest.raises(ValueError):
        score_output_path("out.tsv", "nonsense")


def test_network_genes_path():
    assert network_genes_path("out.tsv") == "out_genes.txt"


def test_score_genes_path_shares_main_network_sidecar():
    # Score dumps intentionally reuse the same `<stem>_genes.txt` sidecar as
    # the main network output (see dump.py), not a `_puc`/`_mi`-suffixed one.
    assert score_genes_path("out.tsv", "puc") == "out_genes.txt"
