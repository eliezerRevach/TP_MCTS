"""
Probabilistic Temporal RPG Heuristic — admissible upper-bound version.

Heuristic name (experiments.ipynb / strategy key): ``baseline_admissible``.

Status: ADMISSIBLE. Every combination operator below is a provable UPPER BOUND on
the relaxed probability that a fact is achieved by a given time, so the heuristic
never under-estimates reachability under the delete-relaxed temporal RPG envelope.
It is deliberately VERY OPTIMISTIC: the bounds are loose (Frechet min for
conjunctions, union bound for disjunctions) but always safe — unlike the default
``baseline`` strategy, which uses the *independence product* (AND) and *noisy-OR*
(OR). Those two are tighter but NOT admissible in general (the product breaks under
positively-correlated preconditions; noisy-OR breaks under negatively-correlated /
mutex achievers). This strategy trades that tightness for an always-valid bound.

Reference: PTRPG_Cleaned.docx ("Probabilistic Temporal RPG Heuristic — Starting
Point"). The structure follows Hoffmann & Nebel (2001) FF relaxed planning graphs,
the probabilistic planning-graph line (Bryce/Kambhampati/Smith 2006; Little &
Thiebaux 2006; Domshlak & Hoffmann 2007), and temporal RPG layers (Coles et al.
2008), specialised to the TP-MCTS deadline-reward setting (Berman, Brafman &
Karpas 2025).

------------------------------------------------------------------------------
Quantities stored per fact ``f`` and represented time ``t_i`` (doc Section 3)
------------------------------------------------------------------------------
- ``P_f(t_i)`` : admissible UPPER bound on the probability that ``f`` has been
                 achieved by time ``t_i`` (cumulative).
- ``H_f(t_i)`` : upper bound on the conditional probability that ``f`` is *newly*
                 achieved exactly at layer ``t_i`` given it was not achieved
                 before (the one-layer hazard).

An achiever is ``e = <a_e, tau_e, C_e, q_e, f_e>`` (doc 3.1): action ``a_e``, a
positive effect offset ``tau_e > 0``, required facts ``C_e``, success probability
``q_e in [0, 1]`` once the requirements hold, and produced fact ``f_e``. If the
effect lands at ``t_i`` the start time is ``s_e(t_i) = t_i - tau_e`` (doc 3.2).

------------------------------------------------------------------------------
Core recursion (doc Section 4)
------------------------------------------------------------------------------
- Initial layer:  ``P_f(t_0) = 1`` if ``f`` holds in the state, else ``0``.
- Later layers:   ``P_f(t_i) = P_f(t_{i-1}) + (1 - P_f(t_{i-1})) * H_f(t_i)``.

The update ``g(p, h) = p + (1 - p) * h`` is monotone non-decreasing in both ``p``
and ``h`` on ``[0, 1]^2`` (doc Section 9), which is what carries the
upper-bound property through the induction over sorted layers.

------------------------------------------------------------------------------
AND layer — requirements of one achiever (doc Section 5)
------------------------------------------------------------------------------
``R_e(t_i) = min_{f in C_e} P_f(s_e(t_i))``   (Frechet conjunction upper bound)

This is the probability-scale analogue of ``h_max``: a conjunction is bounded by
its weakest (least probable) requirement. Preconditions are NOT multiplied — the
product (``baseline``) is only valid under proven independence (doc Section 7.1),
which the admissible version does not assume.

Single-achiever contribution:  ``B_e(t_i) = q_e * R_e(t_i)``  (doc 5.2).

------------------------------------------------------------------------------
OR layer — alternative achievers of the same fact (doc Section 6)
------------------------------------------------------------------------------
``H_f(t_i) = min(1, sum_{e in Ach_f(t_i)} B_e(t_i))``   (union bound)

The probability that at least one achiever succeeds cannot exceed the sum of the
individual success probabilities, so this is always optimistic. Noisy-OR
(``1 - prod(1 - B_e)``, used by ``baseline``) is tighter but only admissible when
achiever-success events are non-negatively correlated (doc Section 7.2).

------------------------------------------------------------------------------
Time layers (doc Section 2)
------------------------------------------------------------------------------
The doc builds the represented layer set ``T`` by relaxed forward expansion: a
layer ``u + tau_e`` is added only when some achiever ``e`` whose requirements are
already reachable by ``u`` can fire, with local deltas ``Delta_i = t_i - t_{i-1}``.
On the integer grid used by the rest of this codebase (effect offsets discretised
to ``effect_delay_steps``), the dense layers ``0, 1, ..., depth`` are a superset of
``T`` and produce identical ``P_f`` at every represented layer (unrepresented
layers contribute no new arrivals), so :func:`propagate_admissible_temporal_rpg`
and the ``baseline_admissible`` strategy iterate the dense grid for simplicity.

The pure operators below are the single source of truth for the admissible
combination rules; the ``baseline_admissible`` strategy in
``temporal_probabilistic_rpg.py`` calls them inside the shared forward DP.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, Mapping, Sequence, Set


Fact = Hashable


def _clamp_probability(value: float) -> float:
    """Clamp a probability into ``[0, 1]`` (doc Section 9 domain guard)."""
    return max(0.0, min(1.0, float(value)))


def admissible_and_support(
    preconditions: Iterable[Fact],
    fact_probs: Mapping[Fact, float],
) -> float:
    """
    AND layer (doc 5.1): admissible upper bound on the joint probability that all
    required facts in ``C_e`` hold, via the Frechet min bound.

        R_e = min_{f in C_e} P_f(s_e)

    An empty requirement set is vacuously satisfied -> ``1.0``. This is always
    >= the independence product used by ``baseline`` (since ``min >= prod`` for
    probabilities in ``[0, 1]``), so it can only make the heuristic more
    optimistic, never break admissibility.
    """
    requirements = list(preconditions)
    if not requirements:
        return 1.0
    return _clamp_probability(
        min(_clamp_probability(fact_probs.get(fact, 0.0)) for fact in requirements)
    )


def union_bound_or_hazard(success_probs: Iterable[float]) -> float:
    """
    OR layer (doc 6.2): admissible upper bound on the per-layer hazard that some
    achiever produces the fact, via the union bound.

        H_f = min(1, sum_e B_e)

    Always >= the noisy-OR ``1 - prod(1 - B_e)`` used by ``baseline`` (by the
    union bound), so the combined estimate is an upper bound.
    """
    total = 0.0
    for probability in success_probs:
        total += _clamp_probability(probability)
    return _clamp_probability(min(1.0, total))


def cumulative_retry_update(previous_probability: float, hazard: float) -> float:
    """
    Core recursion (doc 4.2 / 9.1): cumulative update ``g(p, h) = p + (1 - p) h``.

    ``f`` is true by ``t_i`` either because it was already true by the previous
    represented layer, or because it was newly achieved at ``t_i``. Monotone
    non-decreasing in both arguments on ``[0, 1]^2``.
    """
    p = _clamp_probability(previous_probability)
    h = _clamp_probability(hazard)
    return _clamp_probability(p + (1.0 - p) * h)


def propagate_admissible_temporal_rpg(
    action_models: Sequence[object],
    facts: Iterable[Fact],
    state_facts: Iterable[Fact],
    depth: int,
) -> Dict[int, Dict[Fact, float]]:
    """
    Standalone reference implementation of the admissible PTRPG forward DP.

    Duck-typed ``action_models`` need ``.name``, ``.preconditions`` (iterable of
    facts ``C_e``), ``.add_probabilities`` (mapping ``f_e -> q_e``), and
    ``.effect_delay_steps`` (discretised ``tau_e``) — i.e. the
    ``TemporalRelaxedActionModel`` shape used by the temporal heuristic.

    Returns ``{layer: {fact: P_f(layer)}}`` for ``layer in 0..depth``. This is the
    canonical, dependency-light version used by the tests; the production strategy
    reuses the shared DP plumbing (caching, traces, action-support profiles) but
    applies the very same :func:`admissible_and_support` and
    :func:`union_bound_or_hazard` operators.
    """
    depth = max(0, int(depth))
    state_set: Set[Fact] = set(state_facts)
    all_facts: Set[Fact] = set(facts) | state_set
    for model in action_models:
        all_facts.update(getattr(model, "preconditions", ()))
        all_facts.update(getattr(model, "add_probabilities", {}).keys())

    probabilities_by_layer: Dict[int, Dict[Fact, float]] = {
        t: {fact: 0.0 for fact in all_facts} for t in range(depth + 1)
    }
    for fact in all_facts:
        probabilities_by_layer[0][fact] = 1.0 if fact in state_set else 0.0

    # (arrival_layer, fact) -> list of single-achiever contributions B_e.
    pending: Dict[tuple, list] = {}

    for layer in range(depth + 1):
        # Persistence: facts stay achieved (delete-relaxed envelope).
        if layer > 0:
            for fact in all_facts:
                probabilities_by_layer[layer][fact] = max(
                    probabilities_by_layer[layer][fact],
                    probabilities_by_layer[layer - 1][fact],
                )

        # OR layer + cumulative update for arrivals landing at this layer.
        for fact in all_facts:
            contributions = pending.get((layer, fact))
            if not contributions:
                continue
            hazard = union_bound_or_hazard(contributions)
            current = probabilities_by_layer[layer][fact]
            probabilities_by_layer[layer][fact] = max(
                current, cumulative_retry_update(current, hazard)
            )

        if layer == depth:
            break

        # AND layer: schedule each achiever's contribution to its arrival layer.
        for model in action_models:
            preconditions = getattr(model, "preconditions", ())
            support = admissible_and_support(
                preconditions, probabilities_by_layer[layer]
            )
            if support <= 0.0:
                continue
            arrival_layer = layer + int(getattr(model, "effect_delay_steps", 1))
            if arrival_layer > depth:
                continue
            for fact, add_prob in getattr(model, "add_probabilities", {}).items():
                contribution = _clamp_probability(support * _clamp_probability(add_prob))
                pending.setdefault((arrival_layer, fact), []).append(contribution)

    return probabilities_by_layer
