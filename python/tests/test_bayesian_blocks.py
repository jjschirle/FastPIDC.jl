"""Bayesian-blocks discretization: solver behavior and FastPIDC.jl parity.

FastPIDC.jl is the behavioral reference. Within a language the solvers are
required to agree bit for bit, so those checks compare raw float bits.

Across languages, the selected partition - change points, and therefore bin
edges - must still match exactly, but the objective score is compared with a
tolerance: the per-endpoint prior involves `K ** -0.478`, and Julia's and
NumPy's `pow` differ by one ULP at a handful of endpoints (K = 591, 1810 and
1922 among the first 2000). That shifts a score by ~1e-15 without changing
which partition wins. FastPIDC.jl's own CPU/GPU tests use the same convention.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fastpidc.discretizers import (
    BayesianBlocksProblem,
    LinearDiscretizer,
    binedges_bayesian_blocks,
    prepare_bayesian_blocks,
    solve_bayesian_blocks_cpu,
)


def _pseudo_random_zero_inflated(seed: int, size: int = 64) -> np.ndarray:
    """Deterministic zero-inflated values, mirroring the generator used by
    FastPIDC.jl's `test/bayesian_blocks_tests.jl`."""
    mask = (1 << 64) - 1
    state = seed
    values = np.empty(size, dtype=np.float64)
    for i in range(size):
        state = (state * 6364136223846793005 + 1442695040888963407) & mask
        value = float(state % 2001 - 1000) / 1000
        values[i] = 0.0 if state % 5 == 0 else value
    return values


def _synthetic_cases() -> dict[str, np.ndarray]:
    cases = {
        "repeated_values": np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 8.0, 13.0]),
        "negatives_and_repeats": np.array([-4.0, -2.0, -2.0, -1.0, 0.0, 0.0, 0.25, 0.5, 2.0, 9.0]),
        "tiny_spacings": np.array([0.0, 0.0, 1.0e-12, 2.0e-12, 0.1, 0.1, 1.0, 10.0, 100.0]),
        "uniform_grid": np.linspace(-3.0, 5.0, 24),
        "mostly_constant_with_outlier": np.concatenate((np.zeros(20), [5.0])),
        "all_singletons": np.array([1.0, 2.0, 3.0]),
        "two_heavy_levels": np.concatenate((np.zeros(30), np.full(30, 7.0))),
        "constant": np.full(16, 2.5),
        "single_observation": np.array([1.5]),
        "clustered": np.concatenate((np.zeros(12), np.arange(0.5, 8.5, 0.5), np.full(12, 15.0))),
        "quadratic": np.array([i**2 / 17 for i in range(40)]),
    }
    for seed in (1, 2, 3, 7, 11):
        cases[f"zero_inflated_{seed}"] = _pseudo_random_zero_inflated(seed)
    return cases


SYNTHETIC_CASES = _synthetic_cases()

# An exactly-tied fitness pair is not reachable from raw observations (candidate
# blocks always differ in either their point count or their width), so the
# tie-breaking rule is exercised through a hand-built problem instead. At
# endpoint K = 2 the two candidate starts below score identically, bit for bit.
TIE_PROBLEM = BayesianBlocksProblem(
    edges=np.array([0.0, 31.85815406716697, 38.74613546065674]),
    prefix_counts=np.array([27.0, 28.0]),
)


def _legacy_binedges(data: np.ndarray) -> np.ndarray:
    """Frozen copy of the incremental count-vector solver used before the
    prefix-count rewrite, kept so the new implementation can be required to
    reproduce the previous edges bit for bit (as FastPIDC.jl does for its own
    frozen reference)."""
    sorted_data = np.sort(np.ravel(np.asarray(data, dtype=np.float64)))
    unique_data, counts = np.unique(sorted_data, return_counts=True)
    n = unique_data.size
    nn_vec = counts.astype(np.float64)

    edges = np.empty(n + 1, dtype=np.float64)
    edges[0] = unique_data[0]
    edges[1:-1] = 0.5 * (unique_data[:-1] + unique_data[1:])
    edges[-1] = unique_data[-1]
    block_length = unique_data[-1] - edges

    count_vec = np.zeros(n, dtype=np.float64)
    best = np.zeros(n, dtype=np.float64)
    last = np.zeros(n, dtype=np.int64)

    for k in range(n):
        widths = block_length[: k + 1] - block_length[k + 1]
        count_vec[: k + 1] += nn_vec[k]
        prior = 4 - np.log(73.53 * 0.05 * ((k + 1) ** -0.478))
        with np.errstate(divide="ignore", invalid="ignore"):
            fit_vec = count_vec[: k + 1] * np.log(count_vec[: k + 1] / widths) - prior
        if k > 0:
            fit_vec[1:] += best[:k]
        i_max = int(np.argmax(fit_vec))
        last[k] = i_max
        best[k] = fit_vec[i_max]

    change_points = []
    ind = n
    while True:
        change_points.append(ind)
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points.reverse()
    return edges[change_points]


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


# --- Fixtures --------------------------------------------------------------


@pytest.fixture(scope="session")
def repository_genes(julia_test_data) -> list[tuple[str, np.ndarray]]:
    """A deterministic spread of real genes from the repository's test data.

    Includes the genes with the fewest and the most distinct values (where
    Bayesian blocks is most sensitive) plus an evenly spaced sample.
    """
    rows: list[tuple[str, np.ndarray]] = []
    with open(julia_test_data / "toy_small_200.txt") as fh:
        next(fh)
        for line in fh:
            fields = line.split()
            if fields:
                rows.append((fields[0], np.array(fields[1:], dtype=np.float64)))

    by_uniqueness = sorted(range(len(rows)), key=lambda i: np.unique(rows[i][1]).size)
    selected = {by_uniqueness[0], by_uniqueness[-1], *range(0, len(rows), 20)}
    return [rows[i] for i in sorted(selected)]


@pytest.fixture(scope="session")
def julia_bb_results(run_julia, repository_genes, tmp_path_factory):
    """Bayesian-block edges, change points and scores from FastPIDC.jl.

    Julia is invoked once per session (its startup dominates the solve
    itself) for both the synthetic cases and the real repository genes.
    """
    work_dir = tmp_path_factory.mktemp("julia_bb")
    cases = dict(SYNTHETIC_CASES)
    cases.update({f"gene::{label}": values for label, values in repository_genes})

    names = sorted(cases)
    for index, name in enumerate(names):
        np.save(work_dir / f"in_{index}.npy", np.asarray(cases[name], dtype=np.float64))

    run_julia(
        "using FastPIDC, NPZ\n"
        f'dir = raw"{work_dir}"\n'
        f"for index = 0:{len(names) - 1}\n"
        '    values = npzread(joinpath(dir, "in_$(index).npy"))\n'
        "    problem = FastPIDC.prepare_bayesian_blocks(values)\n"
        "    solution = FastPIDC.solve_bayesian_blocks_cpu(problem)\n"
        '    npzwrite(joinpath(dir, "edges_$(index).npy"), problem.edges[solution.change_points])\n'
        '    npzwrite(joinpath(dir, "cps_$(index).npy"), solution.change_points .- 1)\n'
        '    npzwrite(joinpath(dir, "score_$(index).npy"), [solution.score])\n'
        "end\n"
    )

    return {
        name: (
            np.load(work_dir / f"edges_{index}.npy"),
            np.load(work_dir / f"cps_{index}.npy"),
            float(np.load(work_dir / f"score_{index}.npy")[0]),
        )
        for index, name in enumerate(names)
    }


# --- Preparation -----------------------------------------------------------


def test_prepare_collapses_repeated_values_into_prefix_counts():
    problem = prepare_bayesian_blocks(np.array([3.0, 1.0, 1.0, 1.0, 3.0, 5.0]))
    # Unique values 1, 3, 5 with multiplicities 3, 2, 1.
    np.testing.assert_array_equal(problem.prefix_counts, [3.0, 5.0, 6.0])
    # Outer edges are the extreme observations; interior edges are midpoints.
    np.testing.assert_allclose(problem.edges, [1.0, 2.0, 4.0, 5.0])


@pytest.mark.parametrize("name", sorted(SYNTHETIC_CASES))
def test_prepare_prefix_counts_match_cumulative_multiplicities(name):
    values = SYNTHETIC_CASES[name]
    problem = prepare_bayesian_blocks(values)
    unique_values, counts = np.unique(values, return_counts=True)

    np.testing.assert_array_equal(problem.prefix_counts, np.cumsum(counts))
    assert problem.edges.size == unique_values.size + 1
    assert problem.edges[0] == values.min()
    assert problem.edges[-1] == values.max()
    assert problem.prefix_counts[-1] == values.size


def test_prepare_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one observation"):
        prepare_bayesian_blocks(np.array([]))


# --- Solver behavior -------------------------------------------------------


def test_constant_input_returns_degenerate_two_edge_partition():
    problem = prepare_bayesian_blocks(np.full(7, 2.5))
    with warnings.catch_warnings():
        # The generic fitness expression would divide by a zero-width block
        # here; the solver must special-case it rather than compute infinities.
        warnings.simplefilter("error")
        solution = solve_bayesian_blocks_cpu(problem)

    np.testing.assert_array_equal(solution.change_points, [0, 1])
    assert solution.score == 0.0
    np.testing.assert_array_equal(problem.edges[solution.change_points], [2.5, 2.5])


def test_mostly_constant_with_outlier_uses_the_maximal_partition():
    # This is the all-singleton case: the optimum keeps every candidate edge,
    # so the backtracked path is one longer than the number of unique values.
    values = SYNTHETIC_CASES["mostly_constant_with_outlier"]
    problem = prepare_bayesian_blocks(values)
    solution = solve_bayesian_blocks_cpu(problem)

    n_unique = problem.prefix_counts.size
    assert solution.change_points.size == n_unique + 1
    np.testing.assert_array_equal(solution.change_points, np.arange(n_unique + 1))
    np.testing.assert_array_equal(binedges_bayesian_blocks(values), problem.edges)


@pytest.mark.parametrize("name", sorted(SYNTHETIC_CASES))
def test_change_points_are_a_valid_increasing_edge_path(name):
    problem = prepare_bayesian_blocks(SYNTHETIC_CASES[name])
    solution = solve_bayesian_blocks_cpu(problem)

    assert solution.change_points[0] == 0
    assert solution.change_points[-1] == problem.prefix_counts.size
    assert np.all(np.diff(solution.change_points) > 0)


@pytest.mark.parametrize("name", sorted(SYNTHETIC_CASES))
def test_matches_frozen_count_vector_solver(name):
    values = SYNTHETIC_CASES[name]
    assert _same_bits(binedges_bayesian_blocks(values), _legacy_binedges(values))


@pytest.mark.parametrize("name", sorted(SYNTHETIC_CASES))
def test_solver_is_deterministic(name):
    problem = prepare_bayesian_blocks(SYNTHETIC_CASES[name])
    first = solve_bayesian_blocks_cpu(problem)
    second = solve_bayesian_blocks_cpu(problem)
    np.testing.assert_array_equal(first.change_points, second.change_points)
    assert _same_bits(np.array([first.score]), np.array([second.score]))


def test_exact_fitness_tie_selects_the_earlier_candidate():
    solution = solve_bayesian_blocks_cpu(TIE_PROBLEM)
    # Candidate starts 0 and 1 score identically at the final endpoint; the
    # first maximum wins, so the single-block partition is kept.
    np.testing.assert_array_equal(solution.change_points, [0, 2])


def test_repository_genes_produce_usable_bin_edges(repository_genes):
    constant_genes = 0
    for label, values in repository_genes:
        edges = binedges_bayesian_blocks(values)
        assert edges.size >= 2, label
        assert edges[0] == values.min(), label
        assert edges[-1] == values.max(), label

        if np.unique(values).size == 1:
            # A constant gene collapses to a zero-width interval, which is not
            # a usable discretizer; `get_bin_ids` short-circuits it to one bin
            # before Bayesian blocks is ever reached.
            constant_genes += 1
            np.testing.assert_array_equal(edges, [values[0], values[0]])
            continue

        assert np.all(np.diff(edges) > 0), label
        assert LinearDiscretizer(edges).encode(values).max() == edges.size - 2, label

    # The sampled genes deliberately include the least-varying gene in the
    # repository data, which is constant.
    assert constant_genes == 1


# --- FastPIDC.jl parity ----------------------------------------------------


def _assert_matches_julia(solution, problem, julia_result, label: str = "") -> None:
    edges, change_points, score = julia_result
    np.testing.assert_array_equal(solution.change_points, change_points, err_msg=label)
    assert _same_bits(problem.edges[solution.change_points], edges), label
    assert solution.score == pytest.approx(score, abs=1e-9, rel=1e-12), label


@pytest.mark.julia
@pytest.mark.parametrize("name", sorted(SYNTHETIC_CASES))
def test_synthetic_cases_match_julia(name, julia_bb_results):
    problem = prepare_bayesian_blocks(SYNTHETIC_CASES[name])
    _assert_matches_julia(solve_bayesian_blocks_cpu(problem), problem, julia_bb_results[name], name)


@pytest.mark.julia
def test_repository_genes_match_julia(repository_genes, julia_bb_results):
    for label, values in repository_genes:
        problem = prepare_bayesian_blocks(values)
        _assert_matches_julia(solve_bayesian_blocks_cpu(problem), problem, julia_bb_results[f"gene::{label}"], label)


@pytest.mark.julia
def test_gpu_solver_matches_julia(repository_genes, julia_bb_results):
    """Both languages drive the same shared CUDA kernel, so this pins the two
    GPU paths against each other (via Julia's CPU reference)."""
    from fastpidc.cuda import cuda_available, solve_bayesian_blocks_cuda

    if not cuda_available():
        pytest.skip("no functional GPU / cupy backend available")

    names = sorted(SYNTHETIC_CASES)
    cases = [(name, SYNTHETIC_CASES[name]) for name in names]
    cases += [(f"gene::{label}", values) for label, values in repository_genes]

    problems = [prepare_bayesian_blocks(values) for _, values in cases]
    # A constant node has no partition to search and is never batched to the GPU.
    solvable = [i for i, p in enumerate(problems) if p.prefix_counts.size > 1]
    solutions = solve_bayesian_blocks_cuda([problems[i] for i in solvable])

    for i, solution in zip(solvable, solutions):
        name = cases[i][0]
        _assert_matches_julia(solution, problems[i], julia_bb_results[name], name)


@pytest.mark.julia
def test_exact_fitness_tie_matches_julia(run_julia):
    edges = ", ".join(repr(float(e)) for e in TIE_PROBLEM.edges)
    prefix_counts = ", ".join(repr(float(c)) for c in TIE_PROBLEM.prefix_counts)
    output = run_julia(
        "using FastPIDC\n"
        f"problem = FastPIDC.BayesianBlocksProblem([{edges}], [{prefix_counts}])\n"
        "solution = FastPIDC.solve_bayesian_blocks_cpu(problem)\n"
        'println("CPS:", join(solution.change_points .- 1, ","))'
    )
    julia_change_points = [int(x) for x in _tagged_line(output, "CPS:").split(",")]
    solution = solve_bayesian_blocks_cpu(TIE_PROBLEM)
    np.testing.assert_array_equal(solution.change_points, julia_change_points)


def _tagged_line(output: str, tag: str) -> str:
    return next(line for line in output.splitlines() if line.startswith(tag)).removeprefix(tag)
