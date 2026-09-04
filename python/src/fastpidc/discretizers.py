"""Discretization algorithms for continuous data.

Ported from FastPIDC.jl's ``src/discretizers.jl`` (itself vendored from
`Discretizers.jl <https://github.com/sisl/Discretizers.jl>`_), trimmed to the
uniform-width, uniform-count and Bayesian blocks binning used by FastPIDC.

Unlike the Julia implementation, bin ids here are **0-indexed**, to match
Python/NumPy conventions.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "LinearDiscretizer",
    "binedges_bayesian_blocks",
    "binedges_uniform_count",
    "binedges_uniform_width",
    "get_bin_ids",
    "get_bin_ids_zero_as_own_bin",
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


def binedges_bayesian_blocks(data: np.ndarray) -> np.ndarray:
    """Bayesian-blocks bin edges for ``data``.

    Follows the histogram variant of the algorithm in Scargle (2012): event
    data are sorted, then binned by maximizing a fitness function via
    dynamic programming. The number and placement of bins is chosen
    adaptively. Ported from the vendored implementation in
    ``discretizers.jl`` (originally by Michael P.H. Stumpf and T. Chan,
    based on the Python implementation of Jake Vanderplas).

    References
    ----------
    Scargle 2012: http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
    """
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

    for k in range(n):  # K = k + 1 in 1-indexed terms
        block_length_k1 = block_length[k + 1]
        widths = block_length[: k + 1] - block_length_k1
        count_vec[: k + 1] += nn_vec[k]

        # Prior (eq. 21) and fitness function (eq. 19) from Scargle 2012.
        prior = 4 - np.log(73.53 * 0.05 * ((k + 1) ** -0.478))
        fit_vec = count_vec[: k + 1] * np.log(count_vec[: k + 1] / widths) - prior
        if k > 0:
            fit_vec[1:] += best[:k]

        i_max = int(np.argmax(fit_vec))
        last[k] = i_max  # 0-indexed predecessor
        best[k] = fit_vec[i_max]

    change_points = []
    ind = n  # 1-indexed "n + 1" position, tracked as an exclusive index
    while True:
        change_points.append(ind)
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points.reverse()

    return edges[change_points]


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
        ``"uniform_count"``, ``"zero_as_own_bin"`` or ``"binarize"``. Falls
        back to ``"uniform_width"`` if ``mode`` is unrecognized, or if the
        requested method fails on this data.
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
        edges = binedges_uniform_width(values, number_of_bins)
        return LinearDiscretizer(edges).encode(values), number_of_bins

    if mode == "uniform_count":
        try:
            edges = binedges_uniform_count(values, number_of_bins)
            return LinearDiscretizer(edges).encode(values), number_of_bins
        except ValueError:
            edges = binedges_uniform_width(values, number_of_bins)
            return LinearDiscretizer(edges).encode(values), number_of_bins

    if mode == "zero_as_own_bin":
        return get_bin_ids_zero_as_own_bin(values, number_of_bins)

    if mode == "bayesian_blocks":
        try:
            edges = binedges_bayesian_blocks(values)
            return LinearDiscretizer(edges).encode(values), edges.size - 1
        except ValueError:
            edges = binedges_uniform_width(values, number_of_bins)
            return LinearDiscretizer(edges).encode(values), number_of_bins

    edges = binedges_uniform_width(values, number_of_bins)
    return LinearDiscretizer(edges).encode(values), number_of_bins
