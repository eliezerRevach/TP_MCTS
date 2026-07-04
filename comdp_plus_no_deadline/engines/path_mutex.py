"""
Temporal path-mutex tightening for the admissible PTRPG OR layer.

Heuristic name (experiments.ipynb / strategy key): ``baseline_admissible_paths``.

Design
------
This is a MINIMAL, PROVABLE generalization of ``baseline_admissible`` (see
``admissible_temporal_rpg.py``), not a separate algorithm. ``baseline_admissible``
is a forward layered DP:

    P_t(f) = cumulative_retry_update(P_{t-1}(f), H_t(f))         # g(p,h)=p+(1-p)h
    H_t(f) = union_bound_or_hazard(B_e(t) for e arriving at t)   # flat sum, capped

The ONLY thing this module changes is how ``H_t(f)`` — the per-layer OR-hazard
over this layer's arriving achiever contributions ``B_e(t)`` — is combined. Instead
of a flat capped sum, the contributions are folded into a small K-bounded TABLE
(:func:`table_or_hazard`, via :func:`insert_or_absorb`) of mutually-exclusive
alternatives, and ``H_t(f)`` is read as the ``max`` over that table. Everything
else — the AND-layer Frechet min, the cross-layer cumulative-retry recursion, the
persistence step — is IDENTICAL to ``baseline_admissible``.

Why this is provably ``<= baseline_admissible``
-------------------------------------------------
``table_or_hazard``'s value is always ``<= sum of the same contributions``: each
contribution, as it is inserted, either (a) sums into the running free total — a
genuine ``+=`` that cannot exceed what flat-summing would have given so far, or
(b) is merged into an existing row via ``max`` — which can only be ``<=`` what
summing it in would have given (``max(a, b) <= a + b`` for ``a, b >= 0``). By
induction over the insertions, ``max(table) <= sum(all contributions)``, i.e.
``H_t(f)_table <= H_t(f)_baseline``. Since ``cumulative_retry_update`` is monotone
non-decreasing in ``h``, a second induction over layers gives
``P_t(f)_table <= P_t(f)_baseline`` at every layer — the whole point of this
module. ``baseline_admissible`` is exactly the DEGENERATE case where ``mutex_fn``
never certifies a conflict: every insertion takes the free/sum branch, the table
never exceeds one row, and ``table_or_hazard`` reduces to the flat union sum.

Why this fixes the earlier double-counting bug
-------------------------------------------------
An action whose precondition persists in the delete-relaxed graph re-fires (with
positive support) at EVERY subsequent layer, contributing a fresh ``B_e(t)`` at
each one. A flat cross-layer SUM of these (the old, now-removed, path-carrying
design) treats every re-firing as an independent fresh achievement and blows past
the true value. The cumulative-retry recursion is exactly what discounts this
correctly: the ``(1 - P_{t-1}(f))`` factor shrinks each later layer's contribution
geometrically. Because this module reuses that EXACT recursion for the cross-layer
composition (instead of re-deriving it), the bug cannot recur — only the
WITHIN-layer combination of distinct achievers is changed.

What mutex this design captures (and what it does not)
-----------------------------------------------------------
Two achiever contributions are compared ONLY when they arrive at the SAME layer
(via :func:`table_or_hazard`); a self-mutex action firing again at a LATER,
overlapping layer (e.g. action ``a`` at ``[0,10]`` then again at ``[5,15]`` — the
car-reuse example this module is named for) is NOT caught, since by the time the
later arrival is processed the earlier one's contribution is already folded into
the scalar ``P_{t-1}(f)``.

A cross-layer "shadowing" mechanism (exclude a later arrival if it conflicted
with a still-active earlier one) was tried and REVERTED: it can permanently block
a much STRONGER later mutex-alternative behind a weak earlier one, because there
is no way to retroactively compare values once the earlier one is folded into the
scalar via ``g(p,h)`` — excluding is not the same as taking ``max``, and by the
time a conflict is detected the earlier value can no longer be compared, only
discarded-or-kept. Catching the cross-layer case correctly needs a different
mechanism (hold contested contributions until their window can no longer be
challenged, then take ``max`` over whichever overlapped) — not yet built.

Scope also deferred: this only compares achievers OF THE SAME FACT; two DIFFERENT
facts whose achievers share a mutex resource (e.g. two different goal
preconditions both consuming the same machine) are not directly compared at the
OR layer — the AND-layer's :func:`combine_precondition_footprints` (see below)
gives downstream facts partial visibility into this via inherited footprints, but
it does not change any probability, only footprint bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, FrozenSet, Hashable, List, Mapping, Optional, Sequence, Tuple


Action = Hashable
Fact = Hashable


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class Segment:
    """A timed occupation of ``action`` over the half-open window ``[start, end)``."""

    action: Action
    start: int
    end: int


def segments_overlap(a: Segment, b: Segment) -> bool:
    """True iff the half-open windows ``[start, end)`` intersect.

    Touching endpoints do NOT overlap (one ends exactly as the other starts), and
    a zero-width window (an instantaneous action, ``start == end``) overlaps
    nothing — instantaneous actions do not occupy a resource over time.
    """
    if a.start >= a.end or b.start >= b.end:
        return False  # zero-width (instantaneous) windows occupy no time.
    return a.start < b.end and b.start < a.end


def segments_conflict(a: Segment, b: Segment, mutex_fn: Callable[[Action, Action], bool]) -> bool:
    """The pair occupies overlapping time AND carries mutex actions.

    ``mutex_fn`` is consulted with both actions; for ``a.action == b.action`` it
    must answer the self-mutex question (resource occupation). The overlap test
    already prevents a single segment from conflicting with itself.
    """
    if not segments_overlap(a, b):
        return False
    return mutex_fn(a.action, b.action)


@dataclass
class Row:
    """One row of the K-bounded OR fact table: an abstract policy alternative.

    ``prob`` upper-bounds all concrete contributions the row stands for.
    ``footprint`` is the set of timed segments GUARANTEED present in every
    concrete contribution of the row — the only mutex evidence we are allowed to
    rely on. It only ever SHRINKS (intersection on a max-merge, erased to empty on
    a sum). An empty footprint means "free / unknown": not certified mutex with
    anything.

    ``alternatives`` (OPT-IN, used by the enhanced paths OR/AND layer) is the
    disjunctive-footprint model from the design: a row that stands for "one of
    these footprints happened". A challenger ``c`` is certified mutex with this
    row ONLY IF it conflicts with EVERY alternative (no realization escapes) —
    which is exactly what makes the mutex-preserving SUM sound (two free-summed
    paths stay jointly mutex to a third ``c`` iff both were). When empty, the row
    is single-alternative and ``footprint`` is that single alternative — so
    legacy single-footprint rows behave EXACTLY as before. ``complete`` is the
    admissibility flag: ``False`` means the alternative set is NOT exhaustive
    (some realization is unrepresented), so the row may never certify any mutex
    (uncertain -> free). Truncation sets ``complete=False``; it is the one thing
    that must stay honest or the bound turns inadmissible.
    """

    prob: float
    footprint: FrozenSet[Segment]
    alternatives: Tuple[FrozenSet[Segment], ...] = ()
    complete: bool = True
    origin: Optional[str] = None  # emitting action name: same origin = RETRY class

    def alts(self) -> Tuple[FrozenSet[Segment], ...]:
        """The disjunctive footprints — the single ``footprint`` if none stored."""
        return self.alternatives if self.alternatives else (self.footprint,)


def _footprints_conflict(
    a: FrozenSet[Segment],
    b: FrozenSet[Segment],
    mutex_fn: Callable[[Action, Action], bool],
) -> bool:
    """Some segment of ``a`` conflicts (temporal overlap AND action-mutex) with
    some DISTINCT segment of ``b``. Identical shared occurrences are skipped — a
    shared step is not a contention. Empty on either side -> no conflict (free)."""
    for s1 in a:
        for s2 in b:
            if s1 == s2:
                continue
            if segments_conflict(s1, s2, mutex_fn):
                return True
    return False


def guaranteed_mutex(r1: Row, r2: Row, mutex_fn: Callable[[Action, Action], bool]) -> bool:
    """Certified row-level mutex: EVERY alternative of ``r1`` conflicts with EVERY
    alternative of ``r2`` (so no pair of realizations can co-occur). For the legacy
    single-footprint row (no ``alternatives``) this is exactly "the two footprints
    conflict" — behaviour is unchanged. An incomplete row (``complete=False``) can
    never certify mutex: an unrepresented realization might escape, so uncertain
    returns False and the pair is treated as free (the safe, admissibility-
    preserving default). Same rule for an empty footprint (occupies nothing)."""
    if not (r1.complete and r2.complete):
        return False
    for a in r1.alts():
        for b in r2.alts():
            if not _footprints_conflict(a, b, mutex_fn):
                return False
    return True


def common_footprint(r1: Row, r2: Row) -> FrozenSet[Segment]:
    """Mutex evidence guaranteed by BOTH rows after a merge = the intersection."""
    return r1.footprint & r2.footprint


def insert_or_absorb(
    table: List[Row],
    r_new: Row,
    k: int,
    mutex_fn: Callable[[Action, Action], bool],
    counter: Optional[List[int]] = None,
) -> List[Row]:
    """Insert ``r_new`` into the K-bounded OR fact table, never dropping it.

    Optional ``counter`` is a 3-slot list ``[case1_mutex_add, case2_merge,
    case3_sum]`` of HIT counts: slot 0 increments when r_new is certified mutex
    with EVERY existing row and is added as a new mutex alternative (Case 1 with a
    non-empty M); slot 1 when r_new is mutex-merged into a row (Case 2); slot 2 on
    a free sum (Case 3). A "hit" is a real mutex event = slot0 + slot1.

    Invariant: ``table`` has at most ``k`` rows, certified-mutex pairwise; each row
    is a policy alternative whose ``prob`` upper-bounds its concrete paths and whose
    ``footprint`` holds only mutex info guaranteed by all of them. Exactly three
    cases (``max`` only with certified mutex, ``sum``/free otherwise; ``r_new.prob``
    counted once; footprints intersect on max, erase on sum):

      M = rows certified mutex with r_new;  F = the rest (free / uncertain).
      Case 1  F empty, room  -> add r_new as a new mutex alternative.
      Case 2  F empty, full  -> absorb r_new into the best existing row via max,
              keeping the common (intersected) footprint.
      Case 3  F non-empty    -> r_new is compatible with the F rows, so fold them
              and r_new into one summed (union-bound) row with an erased footprint.

    Old rows are never re-checked against each other; the invariant is preserved by
    construction. PROOF SKETCH (admissibility): every insertion either sums
    (preserves the running total of whatever it touches) or maxes into an existing
    row (``max(a,b) <= a+b``), so by induction ``max(table) <= sum(all inserted
    contributions)`` always holds — see the module docstring.
    """
    M = [r for r in table if guaranteed_mutex(r, r_new, mutex_fn)]
    F = [r for r in table if not guaranteed_mutex(r, r_new, mutex_fn)]

    # Case 3: r_new is free w.r.t. at least one row -> sum and free.
    if F:
        if counter is not None:
            counter[2] += 1
        total = _clamp01(min(1.0, r_new.prob + sum(r.prob for r in F)))
        r_sum = Row(prob=total, footprint=frozenset())  # EMPTY_OR_UNKNOWN
        table[:] = M + [r_sum]
        return table

    # F is empty: r_new is certified mutex with every existing row.
    # Case 1: there is room for another mutex alternative.
    if len(table) < k:
        if counter is not None and M:  # M non-empty => a genuine mutex add (not the first row)
            counter[0] += 1
        table.append(r_new)
        return table

    # Case 2: table full -> absorb r_new into the best row via max.
    if counter is not None:
        counter[1] += 1
    # Prefer the largest common footprint, then the larger max prob, then the
    # smaller (stable) index.
    best_idx = 0
    best_key: Optional[Tuple[int, float, int]] = None
    for idx, r in enumerate(table):
        key = (len(common_footprint(r, r_new)), max(r.prob, r_new.prob), -idx)
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx
    r_best = table[best_idx]
    merged_fp = common_footprint(r_best, r_new)
    r_best.prob = max(r_best.prob, r_new.prob)
    r_best.footprint = merged_fp
    return table


@dataclass
class TableORResult:
    value: float                # max over the K-bounded table = the layer's H_t(f).
    union_value: float          # min(1, flat sum of contributions) = baseline's H_t(f).
    n_paths: int                # number of competing achiever contributions this layer.
    n_rows: int                 # rows kept in the K-bounded table.
    tightened: bool             # value < union (mutex actually reduced the hazard).
    mutex_adds: int = 0         # Case 1 with non-empty M: a mutex row was ADDED.
    mutex_merges: int = 0       # Case 2: a mutex was MERGED into a row.


def table_or_hazard(
    supports: Sequence[Tuple[Action, float, FrozenSet[Segment]]],
    mutex_fn: Callable[[Action, Action], bool],
    k: int,
) -> TableORResult:
    """Per-layer OR-hazard ``H_t(f)`` via the K-bounded :func:`insert_or_absorb`
    table — the drop-in admissible-but-tighter replacement for
    ``union_bound_or_hazard`` inside ``baseline_admissible``'s forward loop.

    ``supports`` are this layer's arriving achiever contributions as
    ``(action_name, B_e(t), footprint)`` triples. ``footprint`` is the full set
    of segments this contribution's "path" carries — at minimum the firing
    action's own ``[start, end)`` window, optionally UNIONED with footprints
    transitively inherited from its preconditions (see
    :func:`combine_precondition_footprints`) — so genuine resource collisions,
    including ones inherited from upstream occurrences, are correctly detected
    via ``mutex_fn``.

    Each contribution becomes a fresh row; the K-bounded table is built by
    repeated :func:`insert_or_absorb`. The result is ``max`` over the table rows,
    which by construction is ``<= min(1, sum of all contributions)`` (the value
    ``union_bound_or_hazard`` would have given) — see the module docstring proof.
    """
    union = _clamp01(min(1.0, sum(p for _, p, _ in supports)))
    if not supports:
        return TableORResult(0.0, union, 0, 0, False)
    table: List[Row] = []
    counter = [0, 0, 0]
    ordered = sorted(supports, key=lambda s: -s[1])
    for name, prob, footprint in ordered:
        insert_or_absorb(table, Row(prob=_clamp01(prob), footprint=footprint), k, mutex_fn, counter)
    value = _clamp01(max(r.prob for r in table)) if table else 0.0
    return TableORResult(
        value=value,
        union_value=union,
        n_paths=len(supports),
        n_rows=len(table),
        tightened=value < union - 1e-12,
        mutex_adds=counter[0],
        mutex_merges=counter[1],
    )


def prune_expired(footprints: Sequence[Segment], current_layer: int) -> List[Segment]:
    """Drop registered segments that can no longer overlap anything new.

    A segment ``[s, e)`` cannot overlap any future segment ``[s', e')`` with
    ``s' >= current_layer >= e`` (its window has fully elapsed), so it is safe to
    discard — pure garbage collection, never changes any future answer.
    """
    return [s for s in footprints if s.end > current_layer]


def combine_precondition_footprints(
    preconditions: Sequence[Fact],
    active_footprints: Mapping[Fact, Sequence[Segment]],
    mutex_fn: Callable[[Action, Action], bool],
    *,
    per_precondition_cap: int = 2,
    combo_cap: int = 8,
) -> FrozenSet[Segment]:
    """Best-effort UNION of one representative recent footprint per precondition
    — the AND-layer's "remember all paths in pai" step.

    For action ``a`` needing ``f1, f2, ...``, the resulting achiever-row should
    carry not just ``a``'s own segment but the segments of WHATEVER occurrences
    made each precondition true, so a LATER mutex check (against some downstream
    fact) can see the full transitive resource usage, not just ``a``'s own. Tries
    up to ``per_precondition_cap`` recent candidates per precondition (most
    recent first) and up to ``combo_cap`` combinations, returning the UNION of
    the first combination found where every chosen segment is pairwise FREE
    (none mutex-and-overlapping with another). If no feasible combination is
    found (or a precondition has no tracked footprint), that precondition simply
    contributes nothing — this function NEVER returns an internally
    self-conflicting footprint, and on failure returns whatever subset it
    safely can (degrading gracefully, never raising).

    This is purely ADDITIVE bookkeeping: callers do not change the admissible
    AND-layer probability (still the Frechet-min over precondition scalars)
    based on this — only the footprint used for FUTURE mutex detection. Unioning
    two segments that are themselves mutex would FALSELY claim "both definitely
    happened" (an over-claim that risks under-counting later), which is exactly
    what the pairwise-feasibility search here prevents.
    """
    candidate_lists: List[List[Segment]] = [
        list(active_footprints.get(f, []))[-per_precondition_cap:][::-1] for f in preconditions
    ]
    candidate_lists = [c for c in candidate_lists if c]
    if not candidate_lists:
        return frozenset()

    def feasible(chosen: Sequence[Segment]) -> bool:
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if segments_conflict(chosen[i], chosen[j], mutex_fn):
                    return False
        return True

    tried = 0
    for combo in product(*candidate_lists):
        tried += 1
        if tried > combo_cap:
            break
        if feasible(combo):
            return frozenset(combo)
    return frozenset()


@dataclass
class PathMutexInstrumentation:
    """Accumulates the per-layer OR-hazard table HIT metrics."""

    or_nodes_total: int = 0           # layer/fact cells with >= 2 competing achievers.
    or_nodes_tightened: int = 0       # cells where the table value < union (mutex bit).
    or_nodes_multi_row: int = 0       # HIT: cells whose final table has > 1 row (mutex alternatives survived).
    mutex_adds: int = 0               # HIT: a mutex row was ADDED (insert_or_absorb Case 1, M non-empty).
    mutex_merges: int = 0             # HIT: a mutex was MERGED into a row (Case 2).
    max_rows_seen: int = 0            # largest table (mutex alternatives) seen.
    mass_shaved: float = 0.0          # total (union - value) removed by mutex.

    @property
    def hits(self) -> int:
        """Total mutex hits: a row added OR a mutex merged into a row."""
        return self.mutex_adds + self.mutex_merges

    def record_or(self, res: TableORResult) -> None:
        if res.n_paths < 2:
            return
        self.or_nodes_total += 1
        self.mutex_adds += res.mutex_adds
        self.mutex_merges += res.mutex_merges
        if res.n_rows > 1:
            self.or_nodes_multi_row += 1
        if res.tightened:
            self.or_nodes_tightened += 1
            self.mass_shaved += res.union_value - res.value
        if res.n_rows > self.max_rows_seen:
            self.max_rows_seen = res.n_rows

    @property
    def tighten_fraction(self) -> float:
        if self.or_nodes_total == 0:
            return 0.0
        return self.or_nodes_tightened / self.or_nodes_total

    def summary(self) -> str:
        return (
            "[pathmutex] OR-nodes(>=2 supports)={t} HITS={h} "
            "(mutex_adds={ma} mutex_merges={mm}) "
            "multi_row_nodes={mr} tightened_nodes={s} max_rows={m} "
            "mass_shaved={ms:.4f}".format(
                t=self.or_nodes_total,
                h=self.hits,
                ma=self.mutex_adds,
                mm=self.mutex_merges,
                mr=self.or_nodes_multi_row,
                s=self.or_nodes_tightened,
                m=self.max_rows_seen,
                ms=self.mass_shaved,
            )
        )


# =====================================================================
# Enhanced OR layer: dominance pruning + mutex-preserving sum
# ---------------------------------------------------------------------
# These are the OPT-IN upgrades from the AND-layer design session; the
# original insert_or_absorb / table_or_hazard above are UNCHANGED (their
# tests pin the erase-on-sum behaviour). The enhanced path lives in
# table_or_hazard_paths and is used by the kernelized AND strategy.
# =====================================================================


def dominates(a: Row, b: Row) -> bool:
    """Future-proof, single-alternative dominance: ``a`` makes ``b`` redundant
    forever. ``prob(a) >= prob(b)`` AND ``footprint(a) subset-of footprint(b)`` =>
    any future challenger that kills ``a`` (conflicts with a's segments) also kills
    ``b`` (b has a superset), and ``a`` scores at least as high. So keeping ``a``
    and dropping ``b`` never changes the OR max NOR ever over-certifies a future
    mutex. Restricted to single-alternative, complete rows — the disjunctive
    (summed) rows are never used as dominator/dominatee (safe: we prune less)."""
    if a.alternatives or b.alternatives:
        return False
    if not (a.complete and b.complete):
        return False
    return a.prob >= b.prob and a.footprint <= b.footprint


def dominance_prune(table: List[Row]) -> List[Row]:
    """Drop every row dominated by a sibling — keep the Pareto frontier over
    (prob, footprint). Value-preserving (max over probs unchanged). Index tie-break
    keeps exactly one of an equal pair. In place; returns the table."""
    survivors: List[Row] = []
    for i, b in enumerate(table):
        dominated = False
        for j, a in enumerate(table):
            if i == j:
                continue
            if dominates(a, b) and (a.prob > b.prob or a.footprint < b.footprint or j < i):
                dominated = True
                break
        if not dominated:
            survivors.append(b)
    table[:] = survivors
    return table


def _merge_alternatives(rows: Sequence[Row], cap: int) -> Tuple[Tuple[FrozenSet[Segment], ...], bool]:
    """Union the disjunctive footprints of ``rows`` (the mutex-preserving-sum
    bookkeeping). ``complete`` = all inputs complete AND no truncation. Over ``cap``
    alternatives -> drop to free/incomplete (``(), False``): the honest degrade."""
    alts: List[FrozenSet[Segment]] = []
    complete = True
    for r in rows:
        if not r.complete:
            complete = False
        for a in r.alts():
            if a not in alts:
                alts.append(a)
    if len(alts) > cap:
        return (), False
    return tuple(alts), complete


def insert_path(
    table: List[Row],
    r_new: Row,
    k: int,
    mutex_fn: Callable[[Action, Action], bool],
    counter: Optional[List[int]] = None,
) -> List[Row]:
    """Enhanced OR insert: same 3-case skeleton as :func:`insert_or_absorb`, but

      * SUM (Case 3) is MUTEX-PRESERVING — the summed row keeps the UNION of its
        members' alternatives instead of erasing the footprint, so it stays
        certified mutex with any ``c`` that all its members were mutex with
        (the "a,b free of each other but both mutex to c" case). ``prob`` is the
        same sum as before, so the OR value (max over probs) is UNCHANGED and the
        ``<= union`` invariant is preserved.
      * DOMINANCE pruning runs after every insert (keeps the Pareto frontier).

    Alternatives are capped at ``max(k, 4)``; overflow degrades to free/incomplete
    (never over-certifies)."""
    cap = max(k, 4)
    M = [r for r in table if guaranteed_mutex(r, r_new, mutex_fn)]
    F = [r for r in table if not guaranteed_mutex(r, r_new, mutex_fn)]

    if F:  # Case 3: r_new is free w.r.t. >=1 row -> sum, but KEEP shared mutex info
        if counter is not None:
            counter[2] += 1
        total = _clamp01(min(1.0, r_new.prob + sum(r.prob for r in F)))
        alts, complete = _merge_alternatives([r_new] + F, cap)
        r_sum = Row(prob=total, footprint=frozenset(), alternatives=alts, complete=complete)
        table[:] = M + [r_sum]
        return dominance_prune(table)

    if len(table) < k:  # Case 1: room for another mutex alternative
        if counter is not None and M:
            counter[0] += 1
        table.append(r_new)
        return dominance_prune(table)

    # Case 2: full -> absorb r_new into the best row via max
    if counter is not None:
        counter[1] += 1
    best_idx = 0
    best_key: Optional[Tuple[int, float, int]] = None
    for idx, r in enumerate(table):
        key = (len(common_footprint(r, r_new)), max(r.prob, r_new.prob), -idx)
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx
    r_best = table[best_idx]
    alts, complete = _merge_alternatives([r_best, r_new], cap)
    r_best.prob = max(r_best.prob, r_new.prob)
    r_best.footprint = common_footprint(r_best, r_new)
    r_best.alternatives = alts
    r_best.complete = complete
    return dominance_prune(table)


def table_or_hazard_paths(
    supports: Sequence[Tuple[Action, float, FrozenSet[Segment]]],
    mutex_fn: Callable[[Action, Action], bool],
    k: int,
) -> TableORResult:
    """Drop-in for :func:`table_or_hazard` using the enhanced :func:`insert_path`
    (mutex-preserving sum + dominance). Same ``<= union`` guarantee: every insert
    still sums or maxes-into-a-row, so ``max(table) <= sum(inputs)``."""
    union = _clamp01(min(1.0, sum(p for _, p, _ in supports)))
    if not supports:
        return TableORResult(0.0, union, 0, 0, False)
    table: List[Row] = []
    counter = [0, 0, 0]
    ordered = sorted(supports, key=lambda s: -s[1])
    for name, prob, footprint in ordered:
        insert_path(table, Row(prob=_clamp01(prob), footprint=footprint), k, mutex_fn, counter)
    value = _clamp01(max(r.prob for r in table)) if table else 0.0
    return TableORResult(
        value=value,
        union_value=union,
        n_paths=len(supports),
        n_rows=len(table),
        tightened=value < union - 1e-12,
        mutex_adds=counter[0],
        mutex_merges=counter[1],
    )


# =====================================================================
# AND layer: temporal-mutex kernelization
# ---------------------------------------------------------------------
# Computes  max over feasible selections (one row per fact, pairwise NON-mutex)
# of min(selected probs), via gate -> connected components -> exact-per-component.
# The component decomposition is EXACT (verified against brute force).
#
# !!! ADMISSIBILITY DOMAIN — READ BEFORE WIRING INTO A HEURISTIC !!!
# max-of-min is admissible PROVIDED each fact's rows satisfy the OR-table
# invariant: separate rows are MUTEX alternatives and free contributions are
# already SUMMED into a single row (which is what insert_path / table_or_hazard
# produce). Under that invariant P(fact) = max over its rows (mutex alternatives:
# you run one), so min-across-facts is the correct Frechet direction and the
# feasibility restriction only lowers it. Do NOT hand-build a fact table with two
# NON-mutex rows kept separate: that malformed input makes max credit one path
# and undercount the OR of the two — an input-validity bug, not an algorithm one.
# For sound ZEROING (declaring a conjunction infeasible) the rows must also be
# COMPLETE (cover every way to reach the fact) — an incomplete row can't certify
# mutex, so it stays a free fallback and the value can't drop below true.
# STILL OPEN for wiring: the per-fact rows at the AND layer must be the CUMULATIVE
# (by-time-t) achievement alternatives, but the DP keeps only the scalar P_t(f)
# built by the cross-layer g(p,h) retry fold (P_t(f) exceeds max-over-one-layer's
# rows). Maintaining cumulative, complete, mutex-alternative row tables across
# layers is the remaining piece — the row-level analogue of g(p,h).
# =====================================================================


@dataclass
class AndKernelResult:
    value: float          # tightened R_e (<= frechet_min always; >= true iff rows complete).
    frechet_min: float    # min over facts of max over rows = the baseline AND value.
    tightened: bool       # value < frechet_min (mutex actually bit).
    n_facts: int
    n_components: int      # connected components of the fact conflict graph.
    max_component: int     # largest multi-fact component = contention degree.


def _fact_max(rows: Sequence[Row]) -> float:
    return max((r.prob for r in rows), default=0.0)


def _facts_mutex(rows_a: Sequence[Row], rows_b: Sequence[Row],
                 mutex_fn: Callable[[Action, Action], bool]) -> bool:
    return any(guaranteed_mutex(r1, r2, mutex_fn) for r1 in rows_a for r2 in rows_b)


def and_has_mutex(fact_rows: Mapping[Fact, Sequence[Row]],
                  mutex_fn: Callable[[Action, Action], bool]) -> bool:
    """The gate: is there ANY cross-fact certified mutex? No -> the whole
    kernelization is a no-op and R_e = the plain Frechet min."""
    facts = list(fact_rows)
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            if _facts_mutex(fact_rows[facts[i]], fact_rows[facts[j]], mutex_fn):
                return True
    return False


def and_components(fact_rows: Mapping[Fact, Sequence[Row]],
                   mutex_fn: Callable[[Action, Action], bool]) -> List[List[Fact]]:
    """Connected components of the fact conflict graph (edge = some cross-fact row
    pair is mutex). Independent components can't constrain each other, so solving
    each separately and taking ``min`` is EXACT — this is the K^F -> sum(K^comp)
    win. Isolated facts are singleton components."""
    facts = list(fact_rows)
    parent = {f: f for f in facts}

    def find(x: Fact) -> Fact:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            if _facts_mutex(fact_rows[facts[i]], fact_rows[facts[j]], mutex_fn):
                parent[find(facts[i])] = find(facts[j])

    comps: dict = {}
    for f in facts:
        comps.setdefault(find(f), []).append(f)
    return list(comps.values())


def exact_component_value(component: Sequence[Fact],
                          fact_rows: Mapping[Fact, Sequence[Row]],
                          mutex_fn: Callable[[Action, Action], bool]) -> float:
    """max over feasible selections (one row per fact, pairwise NON-mutex) of
    min(selected probs). Brute force over the component's row product — sound
    reference and fast because components are machine-sized. A fact with no rows
    (unreachable) -> 0; no feasible selection at all (transitive dead-end /
    tight-deadline serialization) -> 0."""
    facts = list(component)
    if any(not fact_rows[f] for f in facts):
        return 0.0
    best = 0.0
    for combo in product(*(fact_rows[f] for f in facts)):
        feasible = True
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                if guaranteed_mutex(combo[i], combo[j], mutex_fn):
                    feasible = False
                    break
            if not feasible:
                break
        if feasible:
            m = min(r.prob for r in combo)
            if m > best:
                best = m
    return best


def and_support_kernelized(fact_rows: Mapping[Fact, Sequence[Row]],
                           mutex_fn: Callable[[Action, Action], bool]) -> AndKernelResult:
    """Full AND pipeline: gate -> components -> exact per component -> min.

    Returns the tightened R_e. Equals the plain Frechet min when no cross-fact
    mutex is certified; drops below it exactly when a genuine temporal conflict
    makes some conjunction infeasible (the machine_shop serialization signal)."""
    facts = list(fact_rows)
    if not facts:
        return AndKernelResult(1.0, 1.0, False, 0, 0, 0)  # vacuous conjunction
    frechet = min(_fact_max(fact_rows[f]) for f in facts)
    if not and_has_mutex(fact_rows, mutex_fn):
        return AndKernelResult(frechet, frechet, False, len(facts), len(facts), 1)
    comps = and_components(fact_rows, mutex_fn)
    value = frechet
    max_component = 1
    for comp in comps:
        v = exact_component_value(comp, fact_rows, mutex_fn)
        if v < value:
            value = v
        if len(comp) > 1:
            max_component = max(max_component, len(comp))
    return AndKernelResult(
        value=_clamp01(value),
        frechet_min=_clamp01(frechet),
        tightened=value < frechet - 1e-12,
        n_facts=len(facts),
        n_components=len(comps),
        max_component=max_component,
    )


# =====================================================================
# CUMULATIVE-table AND bound (the one used by the table-flowing strategy)
# ---------------------------------------------------------------------
# A fact's CUMULATIVE table (paths accumulated across layers) has FREE/SUMMED
# rows: they are time-separated achievement routes, so P(fact) = sum(row probs)
# = the fact's marginal V_f, NOT max. The AND therefore is NOT max-of-min. The
# admissible bound is a union bound over the COMPATIBLE cross-fact row tuples,
# capped by the Frechet min:
#     P(and f) <= min( min_f V_f ,  sum over compatible tuples of min(tuple) )
# Union bound: P(and f) = P(OR over compatible path-tuples of "all fire") <=
# sum_{compatible} min(probs); incompatible tuples are impossible (0). The min
# with Frechet keeps it <= baseline. Reduces to Frechet exactly when nothing is
# mutex. Admissible for ANY row set (no completeness needed for the UPPER bound;
# completeness only governs whether the drop reflects a real infeasibility).
# =====================================================================


def and_cumulative_bound(fact_rows: Mapping[Fact, Sequence[Row]],
                         marginals: Mapping[Fact, float],
                         mutex_fn: Callable[[Action, Action], bool]) -> float:
    """Admissible AND bound for cumulative (free-summed) per-fact tables — see the
    section header. ``marginals[f]`` is V_f = the fact's scalar achievement prob
    (kept by the DP); ``fact_rows[f]`` are its paths (probs need not sum to exactly
    V_f — the Frechet term uses the marginals, the tuple sum uses the row probs)."""
    facts = list(fact_rows)
    if not facts:
        return 1.0
    frechet = min(_clamp01(marginals.get(f, 0.0)) for f in facts)
    if not and_has_mutex(fact_rows, mutex_fn):
        return frechet
    bound = frechet
    for comp in and_components(fact_rows, mutex_fn):
        if len(comp) < 2:
            continue
        total = 0.0
        for combo in product(*(fact_rows[f] for f in comp)):
            feasible = True
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    if guaranteed_mutex(combo[i], combo[j], mutex_fn):
                        feasible = False
                        break
                if not feasible:
                    break
            if feasible:
                total += min(r.prob for r in combo)
        if total < bound:
            bound = total
    return _clamp01(bound)


def cumulative_merge_truncate(table: List[Row], k: int) -> List[Row]:
    """Sum-PRESERVING truncation for a cumulative (free-summed) table.

    Two passes, both merge-only (NEVER drop — dropping a route's mass from a
    cumulative table under-counts the union bound at the AND -> inadmissible):

      1. DEDUPE: rows with the IDENTICAL footprint merge by prob-sum (both
         route-classes guarantee the same segments, so the merged class does too;
         ``complete`` = AND).
      2. While over ``k`` rows, merge the two WEAKEST into one row: prob summed,
         footprint = INTERSECTION (segments guaranteed by BOTH classes stay
         guaranteed for the union of the classes — certified conflicts on the
         shared part survive; disagreements degrade to uncertain), ``complete`` =
         AND. Frechet per tuple stays valid for any grouping of routes, so the
         AND union bound is unaffected in soundness, only (possibly) loosened.

    Preserves the marginal V = sum(probs) exactly. In place; returns the table."""
    if len(table) > 1:
        by_fp: dict = {}
        for r in table:
            key = (r.footprint, r.alternatives, r.origin)
            prev = by_fp.get(key)
            if prev is None:
                by_fp[key] = Row(r.prob, r.footprint, alternatives=r.alternatives,
                                 complete=r.complete, origin=r.origin)
            else:
                prev.prob = _clamp01(min(1.0, prev.prob + r.prob))
                prev.complete = prev.complete and r.complete
        table[:] = list(by_fp.values())
    if len(table) <= k:
        return table
    table.sort(key=lambda r: r.prob)
    while len(table) > k:
        r1 = table.pop(0)
        r2 = table.pop(0)
        merged = Row(_clamp01(min(1.0, r1.prob + r2.prob)),
                     r1.footprint & r2.footprint,
                     alternatives=(), complete=r1.complete and r2.complete)
        # insert keeping the ascending order cheap.
        lo, hi = 0, len(table)
        while lo < hi:
            mid = (lo + hi) // 2
            if table[mid].prob < merged.prob:
                lo = mid + 1
            else:
                hi = mid
        table.insert(lo, merged)
    return table


# =====================================================================
# v2: transitive (chained) footprints + multi-row AND emission
# ---------------------------------------------------------------------
# The chaining rule from the design session: an achiever's emitted row carries
# not just its own [fire, arrival) segment but the segments of WHATEVER
# occurrences made its preconditions true — per feasible COMBO (one row per
# precondition), NEVER unioned across a fact's alternative rows (that would
# claim "both alternatives happened" and over-certify).
#
# Soundness of every footprint here = the GUARANTEED reading: a segment may be
# in a row's footprint only if EVERY realization of that row-class used it.
#   * own segment: guaranteed (this row-class is "the achiever fired there").
#   * a chosen combo row's footprint: guaranteed for realizations routed
#     through that row-class — which is exactly what the combo row stands for.
#   * a NON-contending precondition contributes its guaranteed REPRESENTATIVE:
#     the INTERSECTION of its rows' footprints (segments used by EVERY route;
#     incomplete rows contribute nothing, killing the intersection — honest).
#   * RESOURCE PROJECTION: only segments of actions that participate in >= 1
#     mutex pair are kept — provably lossless for certification (a non-resource
#     segment can never conflict with anything) and the thing that keeps combo
#     row-sets small and dedupe-able.
# =====================================================================


def retry_fold_arrivals(
    table: List[Row],
    arriving: Sequence[Row],
    alt_cap: int = 8,
) -> List[Row]:
    """Fold a layer's arriving aux rows into the cumulative table with RETRY
    semantics — the session rule "same-path re-firing -> g-discount, never sum".

    Same ``origin`` (action name) = the same route-class re-firing:
      * WITHIN the arriving batch (same layer): alternative concrete routes of
        the class -> probs SUM (union bound over routes), windows accumulate as
        alternatives.
      * ACROSS layers (batch row vs existing table row): a RETRY of the class ->
        probs fold by ``g(p, h) = p + (1-p) h`` (the DP's own cross-layer
        algebra — summing here is the classic re-firing double-count that blows
        a table row up to a free prob-1 mass and kills all tightening), windows
        accumulate as alternatives.

    The class row's ``alternatives`` are its possible windows; certification
    (:func:`guaranteed_mutex`) demands a conflict with EVERY window — a retry
    window that escapes the challenger correctly blocks the zero. ``footprint``
    degrades to the intersection of the alternatives (the always-guaranteed
    part). Over ``alt_cap`` windows the row keeps the first ``alt_cap`` but
    drops ``complete`` — it can then never certify (honest degrade, admissible
    direction). Rows without an origin never fold (safe: they only sum later).
    """
    batch: dict = {}
    loose: List[Row] = []
    for r in arriving:
        if r.origin is None:
            loose.append(r)
            continue
        prev = batch.get(r.origin)
        if prev is None:
            batch[r.origin] = Row(r.prob, r.footprint, alternatives=r.alternatives,
                                  complete=r.complete, origin=r.origin)
        else:
            prev.prob = _clamp01(min(1.0, prev.prob + r.prob))  # same layer: routes sum
            _accumulate_alternatives(prev, r, alt_cap)
    for r in batch.values():
        target = None
        for row in table:
            if row.origin is not None and row.origin == r.origin:
                target = row
                break
        if target is None:
            table.append(r)
        else:
            target.prob = _clamp01(target.prob + (1.0 - target.prob) * r.prob)  # retry: g
            _accumulate_alternatives(target, r, alt_cap)
    table.extend(loose)
    return table


def _accumulate_alternatives(row: Row, other: Row, alt_cap: int) -> None:
    """Merge ``other``'s windows into ``row`` as disjunctive alternatives."""
    alts = list(row.alts())
    for a in other.alts():
        if a not in alts:
            alts.append(a)
    if len(alts) > alt_cap:
        alts = alts[:alt_cap]
        row.complete = False
    else:
        row.complete = row.complete and other.complete
    row.alternatives = tuple(alts)
    inter = alts[0]
    for a in alts[1:]:
        inter = inter & a
    row.footprint = inter


def guaranteed_rep(rows: Sequence[Row]) -> FrozenSet[Segment]:
    """Segments guaranteed by EVERY route of the fact = intersection over its
    rows' footprints. An incomplete row (unknown routes) contributes nothing,
    so it erases the intersection — the honest degrade."""
    acc: Optional[FrozenSet[Segment]] = None
    for r in rows:
        fp = r.footprint if r.complete else frozenset()
        acc = fp if acc is None else (acc & fp)
        if not acc:
            return frozenset()
    return acc or frozenset()


def and_emit_rows(
    fact_rows: Mapping[Fact, Sequence[Row]],
    marginals: Mapping[Fact, float],
    support_value: float,
    own_segment: Segment,
    q: float,
    mutex_fn: Callable[[Action, Action], bool],
    project: Callable[[FrozenSet[Segment]], FrozenSet[Segment]],
    combo_cap: int = 64,
    out_cap: int = 5,
) -> List[Row]:
    """Multi-row AND emission: the achiever's contribution to its effect fact as
    one row PER surviving feasible combo of contending-precondition routes, each
    chaining the combo's (projected) footprints + the achiever's own segment.

    Non-contending preconditions (no certified cross-fact mutex) are not
    enumerated — they contribute their guaranteed representative footprint and
    their marginal as a Frechet floor. If there is no contention (or the combo
    product exceeds ``combo_cap``), degrades to ONE row with
    ``prob = q * support_value`` and the fully-guaranteed footprint — exactly the
    v1 emission, enriched with the guaranteed inherited segments.

    Every returned row's prob is a valid upper bound on its route-class (Frechet
    per combo), and together the rows COVER all routes (combos are enumerated,
    merged on overflow, never dropped) — the two conditions the cumulative AND
    union bound needs."""
    facts = list(fact_rows)
    own_fp = project(frozenset({own_segment}))
    reps = {f: project(guaranteed_rep(fact_rows[f])) for f in facts}
    all_reps: FrozenSet[Segment] = frozenset().union(own_fp, *reps.values()) if facts else own_fp

    def fallback() -> List[Row]:
        p = _clamp01(q * support_value)
        return [Row(p, all_reps, alternatives=(), complete=True)] if p > 0.0 else []

    contending = [f for comp in and_components(fact_rows, mutex_fn) if len(comp) > 1 for f in comp]
    if not contending:
        return fallback()
    size = 1
    for f in contending:
        size *= max(1, len(fact_rows[f]))
        if size > combo_cap:
            return fallback()
    if any(not fact_rows[f] for f in contending):
        return fallback()

    non_contending = [f for f in facts if f not in contending]
    floor = min((_clamp01(marginals.get(f, 0.0)) for f in non_contending), default=1.0)
    nc_fp: FrozenSet[Segment] = frozenset().union(own_fp, *(reps[f] for f in non_contending)) \
        if non_contending else own_fp

    out: List[Row] = []
    for combo in product(*(fact_rows[f] for f in contending)):
        feasible = True
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                if guaranteed_mutex(combo[i], combo[j], mutex_fn):
                    feasible = False
                    break
            if not feasible:
                break
        if not feasible:
            continue
        prob = _clamp01(q * min(min(r.prob for r in combo), floor))
        if prob <= 0.0:
            continue
        fp = frozenset().union(nc_fp, *(project(r.footprint) for r in combo))
        out.append(Row(prob, fp, alternatives=(),
                       complete=all(r.complete for r in combo)))
    cumulative_merge_truncate(out, out_cap)
    return out
