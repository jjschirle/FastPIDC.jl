"""Triple partial information decomposition (PID): redundancy, unique and
synergistic information that two source variables carry about a target.

This is the primitive the interventional plan calls ``pid_triple`` (see
``Interventional PID for GRN inference...md`` and ``LOG.md``'s 2026-09-04
entry #5): given an arbitrary triple ``(source1, source2, target)``, decompose
``MI((source1, source2); target)`` into

    MI_joint = redundancy + unique1 + unique2 + synergy

following the Williams & Beer (2010) lattice with the ``I_min`` redundancy
measure already implemented in :mod:`fastpidc.information`
(:func:`apply_redundancy_formula` = ``E[min(SI_1, SI_2)]``) -- this package
does not implement any other redundancy measure (no BROJA, no ``I_ccs``); see
the interventional plan's own Checkpoint 4 caveat about ``I_min`` being the
weakest link in this approach.

Built entirely from existing building blocks (:func:`fastpidc.network.get_mi_and_si`
and :func:`fastpidc.information.apply_redundancy_formula`); no new information
formula is introduced here. ``redundancy`` and ``unique*`` come directly from
those; ``synergy`` additionally needs the joint mutual information of a
"combined" source built by joint-binning ``source1`` and ``source2``
(:func:`combined_node`), a step this package's discretizers don't do for you
since the two sources are typically bin-compatible pre-existing nodes, not raw
values to be jointly rediscretized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .information import apply_redundancy_formula, get_frequencies_from_bin_ids, get_probabilities
from .network import get_mi_and_si
from .types import Node

__all__ = ["PIDTriple", "combined_node", "pid_triple"]


def combined_node(node1: Node, node2: Node, estimator: str, label: str | None = None) -> Node:
    """Joint-bin ``node1`` and ``node2`` into a single :class:`Node` whose bin
    id encodes both nodes' bin ids (``node1_bin * node2.number_of_bins +
    node2_bin``), for use as the "combined source" in :func:`pid_triple`'s
    synergy term. The combined node's marginal probabilities are estimated
    with ``estimator``, consistent with how :meth:`Node.from_raw_values`
    would compute them for a single discretized variable.
    """
    n2 = node2.number_of_bins
    combined_bins = node1.binned_values.astype(np.int64) * n2 + node2.binned_values.astype(np.int64)
    number_of_bins = node1.number_of_bins * n2
    frequencies = get_frequencies_from_bin_ids(combined_bins, number_of_bins)
    probabilities = get_probabilities(estimator, frequencies)
    return Node(label or f"({node1.label},{node2.label})", combined_bins, number_of_bins, probabilities)


@dataclass(frozen=True, slots=True)
class PIDTriple:
    """Partial information decomposition of ``MI((source1, source2); target)``.

    ``mi_joint == redundancy + unique1 + unique2 + synergy`` (up to floating
    point error) by construction.
    """

    redundancy: float
    unique1: float
    """Information about ``target`` unique to ``source1`` (not available from ``source2``)."""
    unique2: float
    """Information about ``target`` unique to ``source2`` (not available from ``source1``)."""
    synergy: float
    mi1: float
    """``MI(source1, target)``."""
    mi2: float
    """``MI(source2, target)``."""
    mi_joint: float
    """``MI((source1, source2), target)``, the quantity being decomposed."""


def pid_triple(
    source1: Node,
    source2: Node,
    target: Node,
    *,
    estimator: str = "maximum_likelihood",
    base: float = 2.0,
) -> PIDTriple:
    """Decompose the information ``source1`` and ``source2`` jointly carry
    about ``target`` into redundant, unique and synergistic components.

    ``target.probabilities`` is used directly as the target's marginal
    distribution for the redundancy formula (rather than recomputing it), so
    ``target`` should have been constructed (e.g. via
    :meth:`Node.from_raw_values`) with an estimator matching ``estimator``
    here -- consistent with how :func:`fastpidc.network.infer_network_from_nodes`
    itself only supports ``estimator="maximum_likelihood"`` for PUC/PIDC
    without violating shared-marginal assumptions (see its docstring).
    """
    mi1, si1, _ = get_mi_and_si(source1, target, estimator, base)
    mi2, si2, _ = get_mi_and_si(source2, target, estimator, base)
    redundancy = apply_redundancy_formula(target.probabilities, si1, si2, base)
    unique1 = mi1 - redundancy
    unique2 = mi2 - redundancy

    joint = combined_node(source1, source2, estimator)
    mi_joint, _, _ = get_mi_and_si(joint, target, estimator, base)
    synergy = mi_joint - redundancy - unique1 - unique2

    return PIDTriple(redundancy, unique1, unique2, synergy, mi1, mi2, mi_joint)
