"""GPU backend: optionality, chunk sizing, and CPU/GPU agreement.

The GPU tests skip themselves when ``cupy`` or a functional device is
missing; the rest run everywhere, since keeping CUDA strictly optional is
itself part of the package's contract.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from fastpidc.cuda import (
    _MAX_CHUNK_SIZE,
    _bb_kernel_name,
    _bb_memory_batches,
    _bb_problem_bytes,
    _bb_quantile_buckets,
    _bb_threads_for_max_u,
    _chunk_size_for_free_memory,
    _smallest_unsigned_dtype,
    cuda_available,
)
from fastpidc.discretizers import prepare_bayesian_blocks, solve_bayesian_blocks_cpu
from fastpidc.io import get_nodes
from fastpidc.puc import compute_puc_full
from fastpidc.types import Node, PIDCConfig

GIB = 2**30


def _make_nodes(n_nodes: int = 8, n_samples: int = 200, n_bins: int = 4, seed: int = 0) -> list[Node]:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, n_bins, size=n_samples)
    nodes = []
    for i in range(n_nodes):
        noise = rng.integers(0, n_bins, size=n_samples)
        mix = np.where(rng.random(n_samples) < 0.8 - 0.08 * i, base, noise) % n_bins
        nodes.append(
            Node.from_raw_values(f"N{i}", mix.astype(np.float64), "uniform_width", "maximum_likelihood", n_bins)
        )
    return nodes


def test_import_does_not_require_cupy():
    # A CPU-only install must be able to import the package (and everything it
    # re-exports) without cupy being present or even imported.
    script = "import sys; import fastpidc; assert 'cupy' not in sys.modules, sorted(sys.modules)"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_cuda_available_returns_a_bool():
    assert isinstance(cuda_available(), bool)


def test_chunk_size_is_capped_by_the_maximum():
    # Plenty of memory for a small problem: the fixed cap applies.
    assert _chunk_size_for_free_memory(n=1000, k_bins=4, free_bytes=64 * GIB) == _MAX_CHUNK_SIZE


def test_chunk_size_never_exceeds_the_number_of_genes():
    assert _chunk_size_for_free_memory(n=10, k_bins=4, free_bytes=64 * GIB) == 10


def test_chunk_size_shrinks_when_memory_is_tight():
    tight = _chunk_size_for_free_memory(n=5000, k_bins=64, free_bytes=8 * GIB)
    roomy = _chunk_size_for_free_memory(n=5000, k_bins=64, free_bytes=64 * GIB)
    assert 1 <= tight < roomy <= _MAX_CHUNK_SIZE


def test_chunk_size_raises_when_a_single_gene_chunk_does_not_fit():
    # An adaptive discretizer choosing thousands of bins makes the per-target
    # intermediates exceed device memory; that must be an explicit error rather
    # than an allocator failure deep inside the kernel launch loop.
    with pytest.raises(RuntimeError, match="single-gene chunk"):
        _chunk_size_for_free_memory(n=20000, k_bins=4000, free_bytes=8 * GIB)


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_matches_cpu_backend():
    nodes = _make_nodes()
    cpu_mi, cpu_puc = compute_puc_full(nodes, config=PIDCConfig(backend="cpu"))
    gpu_mi, gpu_puc = compute_puc_full(nodes, config=PIDCConfig(backend="cuda"))

    np.testing.assert_allclose(gpu_mi, cpu_mi, atol=1e-12)
    np.testing.assert_allclose(gpu_puc, cpu_puc, atol=1e-9)


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_result_is_independent_of_chunk_size():
    from fastpidc.cuda import compute_puc_full_cuda

    nodes = _make_nodes(n_nodes=12)
    auto_mi, auto_puc = compute_puc_full_cuda(nodes)
    chunked_mi, chunked_puc = compute_puc_full_cuda(nodes, chunk_size=3)

    np.testing.assert_array_equal(chunked_mi, auto_mi)
    np.testing.assert_array_equal(chunked_puc, auto_puc)


# --- Bayesian blocks: kernel selection ------------------------------------


@pytest.mark.parametrize(
    "max_value,expected",
    [(0, np.uint8), (255, np.uint8), (256, np.uint16), (65535, np.uint16), (65536, np.uint32)],
)
def test_smallest_unsigned_dtype(max_value, expected):
    assert _smallest_unsigned_dtype(max_value) == np.dtype(expected)


def test_smallest_unsigned_dtype_rejects_negative():
    with pytest.raises(ValueError):
        _smallest_unsigned_dtype(-1)


@pytest.mark.parametrize(
    "count,index,expected",
    [
        (np.uint8, np.uint8, "bayesian_blocks_dp_u8_u8"),
        (np.uint32, np.uint16, "bayesian_blocks_dp_u32_u16"),
        (np.uint64, np.uint64, "bayesian_blocks_dp_u64_u64"),
    ],
)
def test_bb_kernel_name_matches_the_shared_source(count, index, expected):
    # FastPIDC.jl's `_bb_kernel_name` must resolve these same names.
    assert _bb_kernel_name(np.dtype(count), np.dtype(index)) == expected


def test_bb_kernel_name_rejects_a_wider_back_pointer_type():
    # U_g never exceeds the observation count, so this pair is not instantiated.
    with pytest.raises(ValueError, match="wider"):
        _bb_kernel_name(np.dtype(np.uint8), np.dtype(np.uint16))


def test_bb_kernel_name_rejects_signed_types():
    with pytest.raises(ValueError, match="unsigned"):
        _bb_kernel_name(np.dtype(np.int32), np.dtype(np.uint8))


@pytest.mark.parametrize("max_u,expected", [(32, 32), (33, 64), (512, 64), (513, 128), (4096, 128), (4097, 256)])
def test_bb_threads_for_max_u(max_u, expected):
    assert _bb_threads_for_max_u(max_u) == expected


# --- Bayesian blocks: workload bucketing and the memory budget --------------


def _bb_problems(sizes=(3, 40, 7, 900, 120, 5)):
    rng = np.random.default_rng(0)
    return [prepare_bayesian_blocks(rng.normal(size=size)) for size in sizes]


def test_bb_quantile_buckets_cover_every_problem_exactly_once():
    problems = _bb_problems()
    buckets = _bb_quantile_buckets(problems)
    assert sorted(i for bucket in buckets for i in bucket) == list(range(len(problems)))
    assert 1 <= len(buckets) <= 4


def test_bb_quantile_buckets_of_empty_input():
    assert _bb_quantile_buckets([]) == []


def test_bb_problem_bytes_counts_every_device_buffer():
    problem = _bb_problems(sizes=(40,))[0]
    u = problem.prefix_counts.size
    expected = 8 * (u + 1) + 1 * u + 8 * u + 1 * u  # lengths + prefix + best + back-pointers
    assert _bb_problem_bytes(problem, np.dtype(np.uint8), np.dtype(np.uint8)) == expected


def test_bb_memory_batches_keeps_one_batch_when_everything_fits():
    problems = _bb_problems()
    bucket = list(range(len(problems)))
    assert _bb_memory_batches(bucket, problems, 2**40, np.dtype(np.uint8), np.dtype(np.uint8)) == [bucket]


def test_bb_memory_batches_splits_to_stay_under_the_budget():
    # Equal-sized problems make the packing exact: a budget of two problems
    # must yield three batches of two, in order.
    problems = _bb_problems(sizes=(50,) * 6)
    bucket = list(range(len(problems)))
    count_dtype = index_dtype = np.dtype(np.uint16)
    per_problem = _bb_problem_bytes(problems[0], count_dtype, index_dtype)
    assert all(_bb_problem_bytes(p, count_dtype, index_dtype) == per_problem for p in problems)

    batches = _bb_memory_batches(bucket, problems, 2 * per_problem, count_dtype, index_dtype)

    assert batches == [[0, 1], [2, 3], [4, 5]]
    for batch in batches:
        assert sum(_bb_problem_bytes(problems[i], count_dtype, index_dtype) for i in batch) <= 2 * per_problem


def test_bb_memory_batches_raises_when_one_problem_cannot_fit():
    # Bounding device memory is the point of the budget, so a problem that
    # cannot fit at all must be an explicit error, not an allocator failure.
    problems = _bb_problems(sizes=(900,))
    with pytest.raises(RuntimeError, match="exceeds the CUDA batch budget"):
        _bb_memory_batches([0], problems, 1024, np.dtype(np.uint16), np.dtype(np.uint16))


# --- Bayesian blocks: GPU behavior -----------------------------------------


def _bb_gpu_cases() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    cases = [
        np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 8.0, 13.0]),
        np.concatenate((np.zeros(20), [5.0])),  # maximal partition
        np.concatenate((np.zeros(12), np.arange(0.5, 8.5, 0.5), np.full(12, 15.0))),
        np.array([i**2 / 101 for i in range(300)]),  # needs the uint16 kernel
        rng.normal(size=1000),
        np.round(rng.normal(size=800), 2),  # heavy repeats
    ]
    cases += [np.where(rng.random(400) < 0.3, 0.0, rng.normal(size=400)) for _ in range(8)]
    return cases


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_bayesian_blocks_matches_the_cpu_reference():
    from fastpidc.cuda import solve_bayesian_blocks_cuda

    problems = [prepare_bayesian_blocks(values) for values in _bb_gpu_cases()]
    cpu = [solve_bayesian_blocks_cpu(problem) for problem in problems]
    gpu = solve_bayesian_blocks_cuda(problems)

    for problem, cpu_solution, gpu_solution in zip(problems, cpu, gpu):
        # The selected partition - and so every bin edge - must match exactly.
        np.testing.assert_array_equal(gpu_solution.change_points, cpu_solution.change_points)
        np.testing.assert_array_equal(
            problem.edges[gpu_solution.change_points], problem.edges[cpu_solution.change_points]
        )
        # CUDA and host `log` may differ by a few ULP, so only the objective
        # value is compared with a tolerance (as FastPIDC.jl's tests do).
        assert gpu_solution.score == pytest.approx(cpu_solution.score, abs=1e-9, rel=1e-12)


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_bayesian_blocks_handles_a_constant_problem():
    from fastpidc.cuda import solve_bayesian_blocks_cuda

    problem = prepare_bayesian_blocks(np.full(32, 2.5))
    solution = solve_bayesian_blocks_cuda([problem])[0]
    np.testing.assert_array_equal(solution.change_points, [0, 1])
    assert solution.score == 0.0


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_bayesian_blocks_is_deterministic():
    from fastpidc.cuda import solve_bayesian_blocks_cuda

    problems = [prepare_bayesian_blocks(values) for values in _bb_gpu_cases()]
    first = solve_bayesian_blocks_cuda(problems)
    second = solve_bayesian_blocks_cuda(problems)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.change_points, b.change_points)
        assert a.score == b.score


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_bayesian_blocks_of_no_problems():
    from fastpidc.cuda import solve_bayesian_blocks_cuda

    assert solve_bayesian_blocks_cuda([]) == []


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
@pytest.mark.parametrize("data_file_name", ["toy_small_200.txt", "toy_small_200.h5"])
def test_gpu_and_cpu_node_building_agree(julia_test_data, data_file_name):
    # End-to-end: the batched GPU path must produce nodes indistinguishable
    # from the per-node CPU path, including the probability vectors.
    path = str(julia_test_data / data_file_name)
    gpu_nodes = get_nodes(path, bb_backend="cuda")
    cpu_nodes = get_nodes(path, bb_backend="cpu")

    assert [n.label for n in gpu_nodes] == [n.label for n in cpu_nodes]
    for from_gpu, from_cpu in zip(gpu_nodes, cpu_nodes):
        assert from_gpu.number_of_bins == from_cpu.number_of_bins, from_gpu.label
        np.testing.assert_array_equal(from_gpu.binned_values, from_cpu.binned_values)
        np.testing.assert_array_equal(from_gpu.probabilities, from_cpu.probabilities)


def test_bayesian_blocks_falls_back_to_cpu_without_a_gpu(monkeypatch, julia_test_data):
    # Unlike the PUC backend, a missing GPU here warns and falls back, since
    # both solvers select the same bin edges.
    import fastpidc.cuda

    monkeypatch.setattr(fastpidc.cuda, "cuda_available", lambda: False)
    path = str(julia_test_data / "yeast1_10_data.txt")

    with pytest.warns(RuntimeWarning, match="Falling back to the CPU reference solver"):
        fallback_nodes = get_nodes(path, bb_backend="cuda")

    cpu_nodes = get_nodes(path, bb_backend="cpu")
    for from_fallback, from_cpu in zip(fallback_nodes, cpu_nodes):
        np.testing.assert_array_equal(from_fallback.binned_values, from_cpu.binned_values)
