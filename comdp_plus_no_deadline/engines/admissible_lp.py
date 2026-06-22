"""
Marginal-consistent LP upper bound for the OR-of-AND achiever layer.

Heuristic name (experiments.ipynb / strategy key): ``baseline_admissible_lp``.
This module is the OR-layer tightening described in Section 9.3 of
``PTRPG_Cleaned_13.docx`` ("Marginal-Consistent LP Bound for the OR Layer").

Status: ADMISSIBLE (upper bound). The plain ``baseline_admissible`` strategy
combines the per-achiever contributions ``B_e`` by the union bound
``H_f = min(1, sum_e B_e)`` (doc Section 8). That is always safe, but it is
locally *incoherent*: the Frechet/min AND bound silently assumes maximal overlap
between the preconditions of one achiever, while the union OR bound assumes
minimal overlap between the achiever events. When several achievers *share*
preconditions, the union bound double-counts the same probability mass.

This module replaces the OR-layer hazard with the maximum probability of the full
OR-of-AND achiever formula over *all* local joint distributions that are
consistent with the stored marginal upper bounds ``P(x_l) = p_l``. Because the
true (unknown) local joint is one feasible point of that set, the maximum is still
an upper bound — admissibility is preserved — but it is never looser than the
union bound and can be strictly tighter when achievers share facts.

------------------------------------------------------------------------------
Local LP (doc Section 9.3)
------------------------------------------------------------------------------
For a target fact ``f`` at layer ``t_i`` let ``E_f(t_i) = {e_1, ..., e_k}`` be the
active achievers. Achiever ``e_j`` has preconditions ``C_j`` and success
probability ``q_j`` (the action's add-probability). Let
``U = union_j C_j = {x_1, ..., x_m}`` be the local facts. The heuristic stores
only the marginal upper bounds ``P(x_l) = p_l``.

Introduce one variable ``z_omega`` per local world ``omega in {0,1}^m``:

    maximize    sum_omega z_omega * v_omega
    subject to  z_omega >= 0
                sum_omega z_omega = 1
                sum_{omega: x_l = 1} z_omega <= p_l     for every local fact x_l

The constraint uses ``<=`` because the stored ``p_l`` are upper bounds, not exact
marginals. The per-world value uses the *enabled* achievers
``Enabled(omega) = {e_j : C_j subseteq omega}``:

  - ``value_mode="union"``        : v_omega = min(1, sum_{e in Enabled} q_e)
                                    (the safe conditional-union version; default).
  - ``value_mode="independent"``  : v_omega = 1 - prod_{e in Enabled} (1 - q_e)
                                    (tighter, only admissible if the action-outcome
                                    noise is assumed independent across achievers).

The optimal value replaces ``H_f(t_i)``.

------------------------------------------------------------------------------
Implementation (doc Section 9.3, points 8-9)
------------------------------------------------------------------------------
Brute force: enumerate the ``2^|U|`` local worlds when ``|U| <= M`` (default
``M = 8``), build the LP, and solve it with ``scipy.optimize.linprog`` (HiGHS).
When ``|U| > M`` — or SciPy is unavailable — fall back to the Frechet/min + capped
union bound (so the heuristic stays admissible and dependency-light). The local
formula structure (fact list, achiever precondition sets, success probabilities,
world enumeration, objective vector, constraint matrix) usually repeats across
layers while only the marginals ``p_l`` change, so it is cached by formula
signature; each later call with the same signature only rebuilds the RHS vector.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

Fact = Hashable

# Default cap on the number of distinct local facts |U|. Above this, the 2^|U|
# world enumeration is skipped and the caller falls back to the union bound.
DEFAULT_MAX_LOCAL_FACTS = 8

# Achiever passed to the LP: (preconditions, success probability q_e).
Achiever = Tuple[Iterable[Fact], float]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _fact_sort_key(fact: Fact):
    """Stable order for arbitrary (possibly non-comparable) fact objects."""
    return (type(fact).__name__, repr(fact))


def _has_shared_fact(achievers: Sequence[Tuple[frozenset, float]]) -> bool:
    """True iff some fact appears in the preconditions of two distinct achievers.

    When this is False the achiever events can be realised disjointly, so the
    capped union bound already equals the LP optimum and the LP would only add
    cost. (A single achiever with several preconditions is *not* sharing: its LP
    value is exactly the Frechet B_e.)
    """
    seen = set()
    for pre, _q in achievers:
        for fact in pre:
            if fact in seen:
                return True
            seen.add(fact)
    return False


class _PreparedFormula:
    """Structure-only part of one local LP, reused across layers (RHS changes).

    Holds everything that depends on the *formula signature* but not on the
    marginal values: the local fact order ``U``, the objective vector
    ``c[omega] = v_omega``, and a boolean membership matrix ``A_ub`` with one row
    per local fact (entry 1 where that fact is true in the world). Solving a new
    instance only needs a fresh RHS ``b_ub = [p_l for x_l in U]``.
    """

    __slots__ = ("facts", "c", "a_ub", "a_eq", "b_eq")

    def __init__(self, facts: Tuple[Fact, ...], achievers: Sequence[Tuple[frozenset, float]], value_mode: str):
        import numpy as np

        self.facts = facts
        m = len(facts)
        index = {fact: bit for bit, fact in enumerate(facts)}
        world_count = 1 << m

        # Precompute each achiever's precondition bitmask over U.
        masks: List[Tuple[int, float]] = []
        for preconditions, q in achievers:
            mask = 0
            for fact in preconditions:
                mask |= 1 << index[fact]
            masks.append((mask, q))

        c = np.zeros(world_count, dtype=float)
        for world in range(world_count):
            enabled = [q for mask, q in masks if (world & mask) == mask]
            c[world] = _world_value(enabled, value_mode)
        self.c = c

        # One membership row per local fact: 1 where that fact's bit is set.
        a_ub = np.zeros((m, world_count), dtype=float)
        for bit in range(m):
            for world in range(world_count):
                if world & (1 << bit):
                    a_ub[bit, world] = 1.0
        self.a_ub = a_ub
        self.a_eq = np.ones((1, world_count), dtype=float)
        self.b_eq = np.ones(1, dtype=float)


def _world_value(enabled_qs: Sequence[float], value_mode: str) -> float:
    """Per-world value ``v_omega`` from the enabled achievers' success probs."""
    if not enabled_qs:
        return 0.0
    if value_mode == "independent":
        survive = 1.0
        for q in enabled_qs:
            survive *= 1.0 - _clamp01(q)
        return _clamp01(1.0 - survive)
    # Default: safe conditional-union version.
    return _clamp01(min(1.0, sum(_clamp01(q) for q in enabled_qs)))


class MarginalConsistentORBound:
    """Reusable marginal-consistent OR-layer bound with formula-signature cache.

    One instance is kept per heuristic so the prepared formulas (and the lazy
    SciPy import) are shared across layers and queries. Thread-unfriendly by
    design — the heuristic is single-threaded per worker.
    """

    def __init__(
        self,
        max_local_facts: int = DEFAULT_MAX_LOCAL_FACTS,
        value_mode: str = "union",
    ):
        self.max_local_facts = int(max_local_facts)
        self.value_mode = "independent" if str(value_mode).lower() == "independent" else "union"
        self._prepared: Dict[Tuple, _PreparedFormula] = {}
        # Memoised optima keyed by (formula signature, rounded marginals). Many
        # layers reuse the same formula with identical (often saturated)
        # marginals, so this avoids re-solving the same LP.
        self._result_cache: Dict[Tuple, float] = {}
        # Marginal rounding for the result-cache key (5 decimals is well below the
        # heuristic's own discrimination scale).
        self._marginal_round = 5
        # Lazy SciPy handle: None = not tried, False = unavailable.
        self._linprog = None

    def _ensure_linprog(self):
        if self._linprog is None:
            try:
                from scipy.optimize import linprog

                self._linprog = linprog
            except Exception:
                self._linprog = False
        return self._linprog

    @staticmethod
    def _normalise(achievers: Sequence[Achiever]):
        """Drop zero-q achievers; return (norm, sorted local facts)."""
        norm: List[Tuple[frozenset, float]] = []
        fact_set = set()
        for preconditions, q in achievers:
            qv = _clamp01(q)
            if qv <= 0.0:
                continue
            pre = frozenset(preconditions)
            norm.append((pre, qv))
            fact_set.update(pre)
        return norm, tuple(sorted(fact_set, key=_fact_sort_key))

    def or_hazard(
        self,
        achievers: Sequence[Achiever],
        marginals: Dict[Fact, float],
    ) -> Optional[float]:
        """Return the LP OR-layer hazard, or ``None`` to signal "use the fallback".

        ``None`` is returned whenever the LP cannot tighten the union bound or
        cannot be solved, so the caller keeps the always-admissible union bound:

          - the active achievers share no precondition (union is already exact);
          - too many local facts (``|U| > max_local_facts``);
          - SciPy missing or a solver failure.

        A returned float is an admissible upper bound on the OR-of-AND achiever
        probability and is never looser than the union bound.
        """
        norm, facts = self._normalise(achievers)
        if not norm:
            return 0.0
        if len(facts) > self.max_local_facts:
            return None
        # Fast exit: with no shared fact the achiever events can be laid out
        # disjointly, so the capped union bound already equals the LP optimum and
        # the solve would only add cost. This gate is what keeps the strategy
        # affordable -- only genuinely-shared OR-of-AND structures reach the LP.
        if not _has_shared_fact(norm):
            return None
        return self._solve(facts, norm, marginals)

    def or_hazard_ungated(
        self,
        achievers: Sequence[Achiever],
        marginals: Dict[Fact, float],
    ) -> Optional[float]:
        """Like :meth:`or_hazard` but without the shared-precondition fast exit.

        Always solves the LP (subject to the size / SciPy guards). Used to verify
        that the LP reproduces the union bound on disjoint achievers; production
        code should call :meth:`or_hazard` so the cheap cases are short-circuited.
        """
        norm, facts = self._normalise(achievers)
        if not norm:
            return 0.0
        if len(facts) > self.max_local_facts:
            return None
        return self._solve(facts, norm, marginals)

    def _solve(
        self,
        facts: Tuple[Fact, ...],
        norm: Sequence[Tuple[frozenset, float]],
        marginals: Dict[Fact, float],
    ) -> Optional[float]:
        linprog = self._ensure_linprog()
        if not linprog:
            return None

        signature, prepared = self._get_prepared(facts, norm)

        import numpy as np

        b_vals = [_clamp01(marginals.get(fact, 1.0)) for fact in prepared.facts]
        cache_key = (signature, tuple(round(v, self._marginal_round) for v in b_vals))
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return cached
        b_ub = np.array(b_vals, dtype=float)
        try:
            result = linprog(
                c=-prepared.c,  # maximise sum z_omega v_omega
                A_ub=prepared.a_ub if prepared.a_ub.size else None,
                b_ub=b_ub if prepared.a_ub.size else None,
                A_eq=prepared.a_eq,
                b_eq=prepared.b_eq,
                bounds=(0.0, 1.0),
                method="highs",
            )
        except Exception:
            return None
        if not getattr(result, "success", False):
            return None
        value = _clamp01(-float(result.fun))
        self._result_cache[cache_key] = value
        return value

    def _get_prepared(
        self,
        facts: Tuple[Fact, ...],
        achievers: Sequence[Tuple[frozenset, float]],
    ) -> Tuple[Tuple, _PreparedFormula]:
        # Signature: local fact order + each achiever's precondition set (as bit
        # positions into ``facts``) and its q, plus the value mode. Only the
        # marginals change between calls with the same signature.
        index = {fact: bit for bit, fact in enumerate(facts)}
        achiever_sig = tuple(
            sorted(
                (
                    tuple(sorted(index[f] for f in pre)),
                    round(float(q), 9),
                )
                for pre, q in achievers
            )
        )
        signature = (self.value_mode, facts, achiever_sig)
        prepared = self._prepared.get(signature)
        if prepared is None:
            prepared = _PreparedFormula(facts, achievers, self.value_mode)
            self._prepared[signature] = prepared
        return signature, prepared


def marginal_consistent_or_hazard(
    achievers: Sequence[Achiever],
    marginals: Dict[Fact, float],
    max_local_facts: int = DEFAULT_MAX_LOCAL_FACTS,
    value_mode: str = "union",
) -> Optional[float]:
    """One-shot convenience wrapper around :class:`MarginalConsistentORBound`.

    Builds a throwaway bound object (no cross-call caching). Prefer holding a
    :class:`MarginalConsistentORBound` instance when evaluating many layers.
    """
    return MarginalConsistentORBound(
        max_local_facts=max_local_facts, value_mode=value_mode
    ).or_hazard(achievers, marginals)
