"""Probability estimators and information-theoretic formulae.

Ported from FastPIDC.jl's ``src/information_measures.jl`` (itself vendored
from `InformationMeasures.jl
<https://github.com/Tchanders/InformationMeasures.jl>`_), trimmed to the
probability estimation and information-measure formulae used by FastPIDC.

References
----------
Hausser, J.; Strimmer, K. (2009). "Entropy Inference and the James-Stein
Estimator, with Application to Nonlinear Gene Association Networks".
https://arxiv.org/abs/0811.3579

Timme, N.; Alford, W.; Flecker, B.; Beggs, J.M. (2013). "Synergy, redundancy,
and multivariate information measures: an experimentalist's perspective".
Journal of Computational Neuroscience 36(2):119-140.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply_mutual_information_formula",
    "apply_redundancy_formula",
    "apply_specific_information_formula",
    "get_frequencies_from_bin_ids",
    "get_joint_frequencies_from_bin_ids",
    "get_lambda",
    "get_probabilities",
]


def get_frequencies_from_bin_ids(bin_ids: np.ndarray, number_of_bins: int) -> np.ndarray:
    """Count how many values fall into each of ``number_of_bins`` bins."""
    return np.bincount(bin_ids, minlength=number_of_bins).astype(np.float64)


def get_joint_frequencies_from_bin_ids(
    bin_ids_x: np.ndarray,
    bin_ids_y: np.ndarray,
    number_of_bins_x: int,
    number_of_bins_y: int,
) -> np.ndarray:
    """Joint frequency of each ``(bin_ids_x, bin_ids_y)`` pair, as a
    ``number_of_bins_x`` by ``number_of_bins_y`` matrix of counts."""
    flat_index = bin_ids_x.astype(np.int64) * number_of_bins_y + bin_ids_y.astype(np.int64)
    counts = np.bincount(flat_index, minlength=number_of_bins_x * number_of_bins_y)
    return counts.reshape(number_of_bins_x, number_of_bins_y).astype(np.float64)


# --- Probability estimators -------------------------------------------------


def _probabilities_maximum_likelihood(frequencies: np.ndarray) -> np.ndarray:
    return frequencies / frequencies.sum()


def _probabilities_dirichlet(frequencies: np.ndarray, prior: float) -> np.ndarray:
    prior_frequencies = np.full(frequencies.shape, prior, dtype=np.float64)
    return (frequencies + prior_frequencies) / (frequencies.sum() + prior_frequencies.sum())


def get_lambda(normalized_frequencies: np.ndarray, target: float | np.ndarray, n: float) -> float:
    """James-Stein shrinkage intensity given normalized frequencies, a
    ``target`` distribution and sample size ``n``. Clamped to ``[0, 1]``."""
    if n == 0 or n == 1:
        return 1.0
    varu = normalized_frequencies * (1 - normalized_frequencies) / (n - 1)
    msp = float(np.sum((normalized_frequencies - target) ** 2))
    lam = 1.0 if msp == 0 else float(np.sum(varu) / msp)
    return min(max(lam, 0.0), 1.0)


def _probabilities_shrinkage(frequencies: np.ndarray, lam: float | None) -> np.ndarray:
    target = 1.0 / frequencies.size
    n = frequencies.sum()
    normalized_frequencies = frequencies / n
    if lam is None:
        lam = get_lambda(normalized_frequencies, target, n)
    return lam * target + (1 - lam) * normalized_frequencies


def get_probabilities(
    estimator: str,
    frequencies: np.ndarray,
    *,
    lam: float | None = None,
    prior: float = 1.0,
) -> np.ndarray:
    """Estimate probabilities from bin ``frequencies``.

    Parameters
    ----------
    estimator : ``"maximum_likelihood"`` (or ``"miller_madow"``, treated the
        same), ``"shrinkage"`` or ``"dirichlet"``.
    lam : shrinkage intensity, only used if ``estimator == "shrinkage"``.
        Estimated automatically (via :func:`get_lambda`) if ``None``.
    prior : Dirichlet prior, only used if ``estimator == "dirichlet"``.
    """
    if estimator in ("maximum_likelihood", "miller_madow"):
        return _probabilities_maximum_likelihood(frequencies)
    if estimator == "shrinkage":
        return _probabilities_shrinkage(frequencies, lam)
    if estimator == "dirichlet":
        return _probabilities_dirichlet(frequencies, prior)
    raise ValueError(f"unknown estimator: {estimator!r}")


# --- Formulae ----------------------------------------------------------


def _remove_non_finite(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, 0.0)


def apply_mutual_information_formula(p_xy: np.ndarray, p_x: np.ndarray, p_y: np.ndarray, base: float) -> float:
    """Mutual information ``sum(p_xy * log_base(p_xy / (p_x * p_y)))``
    between two variables. Non-finite terms (from zero probabilities) are
    treated as zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p_xy * (np.log(p_xy / (p_x * p_y)) / np.log(base))
    return float(np.sum(_remove_non_finite(terms)))


def apply_specific_information_formula(
    p_xz: np.ndarray, p_x: np.ndarray, p_z: np.ndarray, axis: int, base: float
) -> np.ndarray:
    """Specific information of a source variable with respect to a target
    variable. Summation is performed along ``axis`` of ``p_xz`` (the
    source's axis), so the result has one value per target bin."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (p_xz / p_z) * (np.log(p_xz / (p_x * p_z)) / np.log(base))
    return np.sum(_remove_non_finite(terms), axis=axis)


def apply_redundancy_formula(p_z: np.ndarray, si1: np.ndarray, si2: np.ndarray, base: float) -> float:
    """Redundancy between two source variables with respect to a common
    target: the expectation (over ``p_z``) of ``min(si1, si2)``. ``base`` is
    accepted for a consistent signature but unused (``si1``/``si2`` already
    encode the log base they were computed with)."""
    del base
    return float(np.sum(np.ravel(p_z) * np.minimum(np.ravel(si1), np.ravel(si2))))
