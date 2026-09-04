import numpy as np
import pytest

from fastpidc.dump import dump_mi_scores, dump_puc_scores
from fastpidc.types import Node, PIDCConfig


@pytest.fixture
def nodes():
    return [Node(label) for label in ("A", "B", "C")]


def test_dump_mi_scores_writes_npy_and_genes(tmp_path, nodes):
    scores = np.arange(9, dtype=np.float64).reshape(3, 3)
    config = PIDCConfig(dump_mi_path=str(tmp_path / "out.tsv"))

    path = dump_mi_scores(scores, nodes, config)

    assert path == str(tmp_path / "out_mi.npy")
    np.testing.assert_array_equal(np.load(path), scores)
    # Shares the same `_genes.txt` sidecar as the main network output.
    genes = (tmp_path / "out_genes.txt").read_text().splitlines()
    assert genes == ["A", "B", "C"]


def test_dump_puc_scores_noop_when_path_not_set(nodes):
    scores = np.zeros((3, 3))
    assert dump_puc_scores(scores, nodes, PIDCConfig()) is None


def test_dump_score_matrix_shape_mismatch_raises(tmp_path, nodes):
    scores = np.zeros((2, 2))
    config = PIDCConfig(dump_mi_path=str(tmp_path / "out.tsv"))
    with pytest.raises(ValueError, match="shape"):
        dump_mi_scores(scores, nodes, config)


def test_dump_puc_path_suffix_not_duplicated(tmp_path, nodes):
    scores = np.zeros((3, 3))
    config = PIDCConfig(dump_puc_path=str(tmp_path / "out_puc.npy"))
    path = dump_puc_scores(scores, nodes, config)
    assert path == str(tmp_path / "out_puc.npy")
