"""Discretization algorithms for continuous data.

Ported from FastPIDC.jl's ``src/discretizers.jl`` (itself vendored from
`Discretizers.jl <https://github.com/sisl/Discretizers.jl>`_), trimmed to the
uniform-width, uniform-count and Bayesian blocks binning used by FastPIDC.

Unlike the Julia implementation, bin ids here are **0-indexed**, to match
Python/NumPy conventions. Bayesian-block change points are likewise
0-indexed into :attr:`BayesianBlocksProblem.edges`.

This module holds the CPU Bayesian-block solver, which is the numerical
reference for both packages. The GPU solver lives in :mod:`fastpidc.cuda`,
which drives the same shared CUDA kernel FastPIDC.jl uses.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BayesianBlocksProblem",
    "BayesianBlocksSolution",
    "LinearDiscretizer",
    "binedges_bayesian_blocks",
    "binedges_uniform_count",
    "binedges_uniform_width",
    "get_bin_ids",
    "prepare_bayesian_blocks",
    "solve_bayesian_blocks_cpu",
]


class LinearDiscretizer:
    """Encode values into bins defined by a sorted list of edges.

    A value ``v`` is encoded into bin ``b`` (0-indexed) if
    ``binedges[b] <= v < binedges[b + 1]`` (the last bin is closed on both
    ends). Values outside the edges are clamped to the nearest end bin.
    """

    __slots__ = ("binedges", "nbins")

    def __init__(self, binedges: np.ndarray):
        binedges = np.asarray(binedges, dtype=np.float64)
        if binedges.size < 2:
            raise ValueError("bin edges must contain at least 2 values")
        if np.any(np.diff(binedges) <= 0):
            raise ValueError("Bin edges must be sorted in increasing order")
        self.binedges = binedges
        self.nbins = binedges.size - 1

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Return the (0-indexed) bin id(s) that ``x`` falls into."""
        x = np.asarray(x, dtype=np.float64)
        if np.any(np.isnan(x)):
            raise ValueError("cannot encode NaN values")
        # searchsorted(edges, x, side="right") - 1 gives the bin such that
        # edges[b] <= x < edges[b+1], matching the bisection search used by
        # the Julia implementation.
        ids = np.searchsorted(self.binedges, x, side="right") - 1
        ids = np.clip(ids, 0, self.nbins - 1)
        return ids


def binedges_uniform_width(data: np.ndarray, nbins: int) -> np.ndarray:
    """``nbins + 1`` edges spaced evenly across ``data``'s range."""
    lo, hi = np.min(data), np.max(data)
    if not hi > lo:
        raise ValueError("data must contain more than one distinct value")
    return np.linspace(lo, hi, nbins + 1)


def binedges_uniform_count(data: np.ndarray, nbins: int) -> np.ndarray:
    """``nbins + 1`` edges such that each bin holds an (approximately) equal
    number of sorted data points.

    Raises ``ValueError`` if there are fewer points than bins, or if any two
    resulting edges coincide (non-unique bin edges).
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.size
    if n < nbins:
        raise ValueError("too many bins requested")

    order = np.argsort(data, kind="stable")
    sorted_data = data[order]

    counts_per_bin, remainder = divmod(n, nbins)
    edges = np.empty(nbins + 1, dtype=np.float64)
    edges[0] = sorted_data[0]
    edges[-1] = sorted_data[-1]

    ind = 0
    for i in range(1, nbins):
        counts = counts_per_bin + (1 if remainder > 0 else 0)
        remainder -= 1
        ind += counts
        edges[i] = (sorted_data[ind - 1] + sorted_data[ind]) / 2.0
        if edges[i - 1] == edges[i]:
            raise ValueError("binedges non-unique")

    return edges


# --- Bayesian blocks -------------------------------------------------------
#
# Histogram variant of the algorithm in Scargle (2012): event data are sorted,
# then binned by maximizing a fitness function via dynamic programming, so both
# the number and the placement of bins are chosen adaptively. Ported from the
# vendored implementation in FastPIDC.jl's ``discretizers.jl`` (originally by
# Michael P.H. Stumpf and T. Chan, based on Jake Vanderplas' Python code).
#
# Like the Julia implementation, preparation (sorting and unique-value
# compression, which is backend-independent) is separated from the dynamic
# program itself, so the two stages can be tested and compared in isolation.
#
# References
# ----------
# Scargle 2012: http://adsabs.harvard.edu/abs/2012arXiv1207.5578S


@dataclass(frozen=True, slots=True)
class BayesianBlocksProblem:
    """Prepared inputs for the Bayesian-block dynamic program.

    Attributes
    ----------
    edges : ``(n_unique + 1,)`` candidate bin edges: the smallest observed
        value, the midpoints between consecutive unique values, and the
        largest observed value.
    prefix_counts : ``(n_unique,)`` cumulative multiplicities, so
        ``prefix_counts[i]`` is how many observations are ``<=`` the
        ``i``-th unique value. Stored as float64 (they are exact integers
        well below 2**53) so the dynamic program needs no casts.
    """

    edges: np.ndarray
    prefix_counts: np.ndarray


@dataclass(frozen=True, slots=True)
class BayesianBlocksSolution:
    """Selected change points and the final dynamic-program objective value.

    Attributes
    ----------
    change_points : 0-indexed positions into :attr:`BayesianBlocksProblem.edges`.
    score : the dynamic program's objective value at the last endpoint.
    """

    change_points: np.ndarray
    score: float


def prepare_bayesian_blocks(data: np.ndarray) -> BayesianBlocksProblem:
    """Sort one node's observations, collapse repeated values, and build the
    prefix-count representation used by the dynamic program.

    Raises
    ------
    ValueError
        If ``data`` is empty.
    """
    sorted_data = np.sort(np.ravel(np.asarray(data, dtype=np.float64)))
    m = sorted_data.size
    if m == 0:
        raise ValueError("Bayesian blocks requires at least one observation")

    # One pass over the sorted values yields both the unique values and, for
    # free, their cumulative multiplicities: each run boundary is exactly the
    # number of observations at or below the preceding unique value.
    boundaries = np.flatnonzero(sorted_data[1:] != sorted_data[:-1]) + 1
    run_starts = np.concatenate(([0], boundaries))
    unique_data = sorted_data[run_starts]
    prefix_counts = np.concatenate((boundaries, [m])).astype(np.float64)

    n = unique_data.size
    edges = np.empty(n + 1, dtype=np.float64)
    edges[0] = unique_data[0]
    edges[1:-1] = 0.5 * (unique_data[:-1] + unique_data[1:])
    edges[-1] = unique_data[-1]

    return BayesianBlocksProblem(edges, prefix_counts)


def solve_bayesian_blocks_cpu(problem: BayesianBlocksProblem) -> BayesianBlocksSolution:
    """Solve one prepared Bayesian-block problem with the exact prefix-count
    dynamic program.

    Prefix counts let each candidate block's observation count be recovered in
    O(1), instead of maintaining a running count vector. The counts are exact
    integers in float64, so the fitness values - and the first-maximum
    tie-breaking that follows from them - are bit-identical to FastPIDC.jl's
    reference solver.
    """
    prefix_counts = problem.prefix_counts
    n = prefix_counts.size

    if n == 1:
        # A constant node: its two outer edges coincide, so the generic
        # fitness expression would divide by a zero-width block. Return the
        # degenerate two-edge partition explicitly, as the Julia reference does.
        return BayesianBlocksSolution(np.array([0, 1], dtype=np.int64), 0.0)

    block_length = problem.edges[-1] - problem.edges
    # counts_before[i] is the cumulative count strictly before candidate start
    # i, so a block [i, K] holds prefix_counts[K] - counts_before[i] points.
    counts_before = np.empty(n, dtype=np.float64)
    counts_before[0] = 0.0
    counts_before[1:] = prefix_counts[:-1]

    best = np.zeros(n, dtype=np.float64)
    last = np.zeros(n, dtype=np.int64)

    for k in range(n):  # endpoint K = k + 1 in the reference's 1-indexed terms
        counts = prefix_counts[k] - counts_before[: k + 1]
        widths = block_length[: k + 1] - block_length[k + 1]

        # Prior (eq. 21) and fitness function (eq. 19) from Scargle 2012.
        prior = 4 - np.log(73.53 * 0.05 * ((k + 1) ** -0.478))
        fitness = counts * np.log(counts / widths) - prior
        fitness[1:] += best[:k]

        # np.argmax returns the first maximum, matching the reference scan's
        # strict `>` comparison.
        i_max = int(np.argmax(fitness))
        last[k] = i_max
        best[k] = fitness[i_max]

    # The maximal partition places every unique value in its own block, so the
    # backtracked path can hold up to n + 1 edge indices; reserve that many and
    # fill from the end.
    change_points = np.empty(n + 1, dtype=np.int64)
    cursor = n + 1
    ind = n
    while True:
        cursor -= 1
        change_points[cursor] = ind
        if ind == 0:
            break
        ind = int(last[ind - 1])

    return BayesianBlocksSolution(change_points[cursor:], float(best[-1]))


def binedges_bayesian_blocks(data: np.ndarray) -> np.ndarray:
    """Bayesian-blocks bin edges for ``data``.

    Convenience wrapper around :func:`prepare_bayesian_blocks` and
    :func:`solve_bayesian_blocks_cpu`.
    """
    problem = prepare_bayesian_blocks(data)
    solution = solve_bayesian_blocks_cpu(problem)
    return problem.edges[solution.change_points]


def _encode_uniform_width(values: np.ndarray, number_of_bins: int) -> tuple[np.ndarray, int]:
    edges = binedges_uniform_width(values, number_of_bins)
    return LinearDiscretizer(edges).encode(values), number_of_bins


def get_bin_ids_zero_as_own_bin(values: np.ndarray, number_of_bins: int) -> tuple[np.ndarray, int]:
    """Discretize ``values`` giving exact zeros their own dedicated bin (bin
    0), then equal-frequency ("uniform_count") binning the nonzero values
    into the remaining ``number_of_bins - 1`` bins.

    Motivated by dropout-dominated single-cell count/expression data, where
    "detected vs. not detected" is itself informative and a naive
    equal-frequency discretizer run on all values (zero and nonzero mixed)
    would otherwise split the zero mass arbitrarily across multiple bins
    whenever zeros exceed one bin's worth of the data, or absorb nonzero
    values into a zero-dominated bin -- either way conflating "not detected"
    with "detected but low" in a way that depends on the zero fraction
    rather than reflecting a real difference in expression level.

    Falls back to a plain two-bin zero/nonzero split if ``number_of_bins`` is
    too small to also split the nonzero values, or if there are no nonzero
    values at all (equivalent then to the ``get_bin_ids`` ``lo == hi`` case).
    """
    values = np.asarray(values, dtype=np.float64)
    zero_mask = values == 0
    nonzero = values[~zero_mask]

    if nonzero.size == 0:
        return np.zeros(values.shape, dtype=np.int64), 1

    if number_of_bins <= 1 or nonzero.size < number_of_bins - 1:
        # Not enough room/data for equal-frequency splitting of the nonzero
        # values; fall back to a single "nonzero" bin alongside the zero bin.
        bin_ids = zero_mask.astype(np.int64) ^ 1  # 0 if zero, 1 if nonzero
        return bin_ids, 2

    nonzero_ids, nonzero_nbins = get_bin_ids(nonzero, "uniform_count", number_of_bins - 1)

    bin_ids = np.zeros(values.shape, dtype=np.int64)
    bin_ids[~zero_mask] = nonzero_ids + 1
    return bin_ids, nonzero_nbins + 1


def get_bin_ids(values: np.ndarray, mode: str, number_of_bins: int) -> tuple[np.ndarray, int]:
    """Discretize ``values`` into bin ids using discretization method ``mode``.

    Parameters
    ----------
    values : array of raw (continuous) data values.
    mode : one of ``"bayesian_blocks"``, ``"uniform_width"``,
        ``"uniform_count"`` or ``"binarize"``. Falls back to
        ``"uniform_width"`` (with a :class:`RuntimeWarning`) if ``mode`` is
        unrecognized, or if the requested method fails on this data.
    number_of_bins : number of bins to use; ignored (and overwritten in the
        return value) when ``mode == "bayesian_blocks"``.

    Returns
    -------
    (bin_ids, number_of_bins) : the (0-indexed) bin id of each value, and the
        actual number of bins used.
    """
    values = np.asarray(values, dtype=np.float64)
    lo, hi = np.min(values), np.max(values)

    if lo == hi:
        return np.zeros(values.shape, dtype=np.int64), 1

    if mode == "binarize":
        return np.where(values == 0, 0, 1).astype(np.int64), 2

    if mode == "uniform_width":
        return _encode_uniform_width(values, number_of_bins)

    if mode == "uniform_count":
        try:
            edges = binedges_uniform_count(values, number_of_bins)
        except ValueError:
            warnings.warn("Uniform count failed, fell back to uniform width", RuntimeWarning, stacklevel=2)
            return _encode_uniform_width(values, number_of_bins)
        return LinearDiscretizer(edges).encode(values), number_of_bins

    if mode == "zero_as_own_bin":
        return get_bin_ids_zero_as_own_bin(values, number_of_bins)

    if mode == "bayesian_blocks":
        try:
            edges = binedges_bayesian_blocks(values)
            discretizer = LinearDiscretizer(edges)
        except ValueError:
            warnings.warn("Bayesian blocks failed, fell back to uniform width", RuntimeWarning, stacklevel=2)
            return _encode_uniform_width(values, number_of_bins)
        return discretizer.encode(values), discretizer.nbins

    warnings.warn(f"Discretizer {mode!r} doesn't exist, fell back to uniform width", RuntimeWarning, stacklevel=2)
    return _encode_uniform_width(values, number_of_bins)
