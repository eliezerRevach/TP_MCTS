"""
Path-mutex temporal tightening for the admissible PTRPG.

Heuristic name (experiments.ipynb / strategy key): ``baseline_admissible_paths``.

Motivation
----------
The per-cell ``baseline_admissible_kmutex`` bound only compares achievers landing
at the SAME fact at the SAME layer, and it never lets an action be mutex with
itself. That misses the dominant over-count in a durative/temporal setting: the
same resource (an action / agent) being reused across OVERLAPPING time windows.

Example (the car): path 1 drives [0, 15], path 2 drives [5, 20]. They share the
car during [5, 15], so the two paths are MUTEX even though structurally it is the
"same action". The cumulative-retry recursion of the scalar DP happily sums those
two as independent retries; this module does not.

What a "path" is
----------------
A :class:`Path` is the timed set of action SEGMENTS that achieves a fact, plus a
probability (product of the achievers' add-probabilities along it). A segment is
``(action_name, start, end)`` — the window the action occupies. We keep <= K
candidate paths per fact per layer.

Mutex between two paths (the whole idea)
----------------------------------------
``paths_mutex(P, Q)`` is True iff SOME segment of ``P`` and SOME segment of ``Q``
both (a) overlap in time and (b) carry mutex actions. This is "at least one mutex
breaks the parallel": one conflicting segment pair anywhere makes the two paths
mutually exclusive. Action-mutex now INCLUDES self-mutex: a resource-occupying
action (one that deletes a precondition it needs, ``del(a) & pre(a) != {}``) is
mutex with itself, so two overlapping uses of it conflict.

OR layer (MAX of SUMS)
----------------------
Alternative achiever-paths are grouped into COMPATIBLE-SET rows: each row is a set
of pairwise-free paths (a schedule that can be executed together) holding the
``sum`` of its members (the union bound for that schedule). A new path is summed
into every row it is free with, and skipped for rows it is mutex with. You can
execute only ONE schedule, so the achievable probability is bounded by the BEST
one: ``P(f) <= max_rows( sum_{path in row} prob(path) )`` — the MAX of SUMS.

This is admissible (``max`` only across mutually-exclusive schedules, ``sum`` only
within a compatible one) and, crucially, MONOTONE in the deadline: more time lets
overlapping paths be scheduled apart, so they become free, join the same row, and
the max rises. It reduces to the union bound when nothing is mutex (one row holds
everything). NOTE: it is max-of-sums, NOT sum-of-(per-clique max) — a new path
that is mutex with one row but free with others must SUM into those others (e.g.
``max(p1, p2+pnew, p3+pnew)``), never collapse to ``max(p1, pnew)``.

AND layer
---------
An action needing several preconditions builds a CONJUNCTIVE path: pick one path
per precondition, UNION their segments, add the action's own segment. The path is
feasible only if it is internally consistent (no two of its own segments are
mutex-overlapping) — if any precondition's chosen achiever conflicts with another,
the parallel cannot be executed and that combination is dropped ("in AND it is at
least one mutex that breaks the parallel").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, FrozenSet, Hashable, List, Optional, Sequence, Tuple


Action = Hashable


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


@dataclass(frozen=True)
class Path:
    """A timed achiever set (``segments``) with its success probability ``prob``."""

    segments: FrozenSet[Segment]
    prob: float

    def union_with(self, others: Sequence["Path"], extra: Segment, new_prob: float) -> "Path":
        segs = set(self.segments)
        for o in others:
            segs |= o.segments
        segs.add(extra)
        return Path(frozenset(segs), _clamp01(new_prob))


def path_internal_feasible(path: Path, mutex_fn: Callable[[Action, Action], bool]) -> bool:
    """No two distinct segments of ``path`` are mutex-overlapping.

    This is what enforces "at least one mutex breaks the parallel" when a
    conjunctive AND path tries to run several achievers concurrently.
    """
    segs = list(path.segments)
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if segments_conflict(segs[i], segs[j], mutex_fn):
                return False
    return True


def paths_mutex(p: Path, q: Path, mutex_fn: Callable[[Action, Action], bool]) -> bool:
    """True iff any segment of ``p`` conflicts with a DISTINCT segment of ``q``.

    Identical segments shared by both paths (same action, same window) are the
    SAME physical occurrence — a shared step, not two contending uses of a
    resource — so they are skipped. Only distinct segments (different windows, or
    different actions) can establish a temporal mutex. Without this, two retry
    paths that share a common prefix would falsely be mutex (the shared self-mutex
    segment overlaps itself), collapsing accumulating retries that should sum.
    """
    for s1 in p.segments:
        for s2 in q.segments:
            if s1 == s2:
                continue
            if segments_conflict(s1, s2, mutex_fn):
                return True
    return False


def _dedup_topk(paths: Sequence[Path], k: int) -> List[Path]:
    """Keep <= ``k`` highest-probability paths, deduped by segment set."""
    best: Dict[FrozenSet[Segment], float] = {}
    for p in paths:
        prev = best.get(p.segments)
        if prev is None or p.prob > prev:
            best[p.segments] = p.prob
    merged = [Path(segs, pr) for segs, pr in best.items()]
    merged.sort(key=lambda p: p.prob, reverse=True)
    return merged[: max(1, int(k))]


def _keep_for_or(paths: Sequence[Path], mutex_fn, k: int) -> List[Path]:
    """Retain <= ``k`` paths chosen for FREE diversity, not raw probability.

    The OR value comes from summing mutually-free (non-overlapping) paths, so when
    capping we must NOT keep ``k`` overlapping near-duplicates of the same attempt
    (which all collapse to one ``max``); we keep paths that are free with the ones
    already chosen, so waited re-uses (``a[0,5]`` then ``a[6,11]`` …) survive and
    accumulate. Greedy: take the highest-probability path, then repeatedly add the
    highest-probability path that is FREE w.r.t. every path chosen so far; if none
    is free, fall back to the highest-probability remaining (a genuine mutex
    alternative). Deterministic and O(k · n) pairwise tests.
    """
    best: Dict[FrozenSet[Segment], float] = {}
    for p in paths:
        prev = best.get(p.segments)
        if prev is None or p.prob > prev:
            best[p.segments] = p.prob
    distinct = [Path(segs, pr) for segs, pr in best.items()]
    if len(distinct) <= max(1, int(k)):
        return distinct
    distinct.sort(key=lambda p: -p.prob)
    selected: List[Path] = []
    remaining = list(distinct)
    while remaining and len(selected) < k:
        pick = None
        for p in remaining:
            if all(not paths_mutex(p, s, mutex_fn) for s in selected):
                pick = p
                break
        if pick is None:
            pick = remaining[0]
        selected.append(pick)
        remaining.remove(pick)
    return selected


@dataclass
class Row:
    """One row of the K-bounded OR fact table: an abstract policy alternative.

    ``prob`` upper-bounds all concrete paths the row stands for. ``footprint`` is
    the set of timed segments GUARANTEED present in every concrete path of the row
    — the only mutex evidence we are allowed to rely on. It only ever SHRINKS
    (intersection on a max-merge, erased to empty on a sum). An empty footprint
    means "free / unknown": not certified mutex with anything.
    """

    prob: float
    footprint: FrozenSet[Segment]


def row_from_path(p: Path) -> Row:
    """A fresh single-path row: it guarantees exactly its own segments."""
    return Row(prob=_clamp01(p.prob), footprint=frozenset(p.segments))


def guaranteed_mutex(r1: Row, r2: Row, mutex_fn: Callable[[Action, Action], bool]) -> bool:
    """Certified row-level mutex: some guaranteed segment of each row conflicts
    (temporal overlap AND action-mutex). If either footprint is empty, or no pair
    conflicts, the pair is NOT certified mutex — uncertain returns False, so it is
    treated as free (the safe, admissibility-preserving default). Identical shared
    segments (same occurrence in both rows) are skipped — a shared step is not a
    contention."""
    for s1 in r1.footprint:
        for s2 in r2.footprint:
            if s1 == s2:
                continue
            if segments_conflict(s1, s2, mutex_fn):
                return True
    return False


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
    construction.
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
class ORAggregateResult:
    value: float
    union_value: float         # min(1, sum of all paths) — the loose union baseline.
    n_paths: int
    n_rows: int                # rows kept in the K-bounded table.
    tightened: bool            # value < union (mutex actually reduced the bound).
    mutex_adds: int = 0        # Case 1 with non-empty M: a mutex row was ADDED.
    mutex_merges: int = 0      # Case 2: a mutex was MERGED into a row.


def or_aggregate_paths(
    paths: Sequence[Path],
    mutex_fn: Callable[[Action, Action], bool],
    k: int,
) -> ORAggregateResult:
    """OR-combine achiever-paths via the K-bounded :func:`insert_or_absorb` table.

    Each path is inserted as a fresh row; the table keeps <= ``k`` rows
    (certified-mutex alternatives, plus at most one free/summed row). The fact's
    probability is the ``max`` over the table rows — you can execute only one
    policy alternative, so the achievable probability is bounded by the best one,
    while free (non-mutex) paths have already been summed into a single row. ``max``
    is used only across certified-mutex rows, so the bound stays admissible and
    reduces to the union when nothing is certified mutex.
    """
    distinct = _dedup_topk(paths, len(list(paths)) or 1)  # dedup identical, keep all
    if not distinct:
        return ORAggregateResult(0.0, 0.0, 0, 0, False)
    union = _clamp01(min(1.0, sum(p.prob for p in distinct)))
    table: List[Row] = []
    counter = [0, 0, 0]  # [case1_mutex_add, case2_merge, case3_sum]
    for p in sorted(distinct, key=lambda p: (-p.prob,)):
        insert_or_absorb(table, row_from_path(p), k, mutex_fn, counter)
    value = _clamp01(max(r.prob for r in table))
    return ORAggregateResult(
        value=value,
        union_value=union,
        mutex_adds=counter[0],
        mutex_merges=counter[1],
        n_paths=len(distinct),
        n_rows=len(table),
        tightened=value < union - 1e-12,
    )


def best_feasible_and(
    precondition_paths: Sequence[Sequence[Path]],
    action_segment: Segment,
    add_prob: float,
    mutex_fn: Callable[[Action, Action], bool],
    *,
    per_precondition_cap: int = 2,
    combo_cap: int = 16,
) -> Optional[Path]:
    """Build the best feasible conjunctive achiever path for one action.

    Picks one path per precondition (searching the top ``per_precondition_cap``
    of each, at most ``combo_cap`` combinations), unions their segments with the
    action's own ``action_segment``, and returns the highest-probability
    combination that is internally feasible. Returns ``None`` when every
    combination has a mutex-overlap among the chosen achievers (the parallel is
    impossible) — or simply ``Path({action_segment}, add_prob)`` when the action
    has no preconditions.
    """
    if not precondition_paths:
        seg_path = Path(frozenset({action_segment}), _clamp01(add_prob))
        return seg_path if path_internal_feasible(seg_path, mutex_fn) else None

    trimmed = [list(p[: max(1, per_precondition_cap)]) for p in precondition_paths]
    if any(not opts for opts in trimmed):
        return None

    combos = list(product(*trimmed))
    # Highest joint probability first so we return the best feasible combination.
    combos.sort(key=lambda c: _prod(p.prob for p in c), reverse=True)
    for combo in combos[: max(1, combo_cap)]:
        base = combo[0]
        joint_prob = _prod(p.prob for p in combo) * _clamp01(add_prob)
        candidate = base.union_with(combo[1:], action_segment, joint_prob)
        if path_internal_feasible(candidate, mutex_fn):
            return candidate
    return None


def _prod(values) -> float:
    out = 1.0
    for v in values:
        out *= float(v)
    return out


@dataclass
class PathMutexInstrumentation:
    """Accumulates the path-mutex survival + AND-feasibility metrics."""

    or_nodes_total: int = 0           # cells with >= 2 candidate paths.
    or_nodes_tightened: int = 0       # cells where the table value < union (mutex bit).
    or_nodes_multi_row: int = 0       # HIT: cells whose final table has > 1 row (mutex alternatives survived).
    mutex_adds: int = 0               # HIT: a mutex row was ADDED (insert_or_absorb Case 1, M non-empty).
    mutex_merges: int = 0             # HIT: a mutex was MERGED into a row (Case 2).
    max_rows_seen: int = 0            # largest table (mutex alternatives) seen.
    mass_shaved: float = 0.0          # total (union - value) removed by mutex.
    and_paths_built: int = 0          # conjunctive paths successfully built.
    and_paths_blocked: int = 0        # conjunctive combos dropped as infeasible.

    @property
    def hits(self) -> int:
        """Total mutex hits: a row added OR a mutex merged into a row."""
        return self.mutex_adds + self.mutex_merges

    def record_or(self, res: ORAggregateResult) -> None:
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

    def record_and(self, built: bool) -> None:
        if built:
            self.and_paths_built += 1
        else:
            self.and_paths_blocked += 1

    @property
    def tighten_fraction(self) -> float:
        if self.or_nodes_total == 0:
            return 0.0
        return self.or_nodes_tightened / self.or_nodes_total

    def summary(self) -> str:
        return (
            "[pathmutex] OR-nodes(>=2 paths)={t} HITS={h} "
            "(mutex_adds={ma} mutex_merges={mm}) multi_row_nodes={mr} "
            "tightened_nodes={s} max_rows={m} mass_shaved={ms:.4f} "
            "AND_built={ab} AND_blocked={bl}".format(
                t=self.or_nodes_total,
                h=self.hits,
                ma=self.mutex_adds,
                mm=self.mutex_merges,
                mr=self.or_nodes_multi_row,
                s=self.or_nodes_tightened,
                m=self.max_rows_seen,
                ms=self.mass_shaved,
                ab=self.and_paths_built,
                bl=self.and_paths_blocked,
            )
        )


def propagate_path_mutex(
    action_models: Sequence[object],
    state_facts,
    facts,
    depth: int,
    mutex_fn: Callable[[Action, Action], bool],
    k: int,
    instrumentation: Optional[PathMutexInstrumentation] = None,
    capture_paths: Optional[Dict[int, Dict[Action, List["Path"]]]] = None,
) -> Tuple[Dict[int, Dict[Action, float]], PathMutexInstrumentation]:
    """Bounded path-carrying forward DP.

    Maintains <= ``k`` timed achiever-paths per fact per layer, combining them
    through the AND layer (:func:`best_feasible_and`) and OR layer
    (:func:`or_aggregate_paths`). Returns ``{layer: {fact: P_layer(fact)}}`` (the
    OR-aggregated scalar, for the usual goal product) and the instrumentation.

    Duck-typed ``action_models`` need ``.name``, ``.preconditions``,
    ``.add_probabilities`` (mapping ``fact -> q``) and ``.effect_delay_steps``.
    """
    depth = max(0, int(depth))
    instr = instrumentation if instrumentation is not None else PathMutexInstrumentation()
    state_set = set(state_facts)
    all_facts = set(facts) | state_set
    for m in action_models:
        all_facts.update(getattr(m, "preconditions", ()))
        all_facts.update(getattr(m, "add_probabilities", {}).keys())

    # paths_by_layer[t][f] -> list[Path]; pending[(arrival, f)] -> list[Path].
    paths_by_layer: Dict[int, Dict[Action, List[Path]]] = {t: {} for t in range(depth + 1)}
    pending: Dict[Tuple[int, Action], List[Path]] = {}
    prob_by_layer: Dict[int, Dict[Action, float]] = {
        t: {f: 0.0 for f in all_facts} for t in range(depth + 1)
    }

    for f in state_set:
        paths_by_layer[0][f] = [Path(frozenset(), 1.0)]

    for t in range(depth + 1):
        layer_paths = paths_by_layer[t]
        # Persistence: a fact stays achieved (carry its paths forward).
        if t > 0:
            for f, plist in paths_by_layer[t - 1].items():
                layer_paths[f] = _keep_for_or(layer_paths.get(f, []) + plist, mutex_fn, k)
        # Arrivals scheduled to land at this layer.
        for (arrival, f), plist in pending.items():
            if arrival == t:
                layer_paths[f] = _keep_for_or(layer_paths.get(f, []) + plist, mutex_fn, k)

        # OR-aggregate every fact's paths into the scalar probability.
        for f, plist in layer_paths.items():
            res = or_aggregate_paths(plist, mutex_fn, k)
            prob_by_layer[t][f] = res.value
            instr.record_or(res)

        if t == depth:
            break

        # AND layer: fire each action whose preconditions all have a path by now.
        for model in action_models:
            d = max(0, int(getattr(model, "effect_delay_steps", 1)))
            arrival = t + d
            if arrival > depth:
                continue
            pres = list(getattr(model, "preconditions", ()))
            pre_paths: List[List[Path]] = []
            ok = True
            for pre in pres:
                opts = layer_paths.get(pre)
                if not opts:
                    ok = False
                    break
                pre_paths.append(opts)
            if not ok:
                continue
            action_segment = Segment(getattr(model, "name", repr(model)), t, t + d)
            support = best_feasible_and(pre_paths, action_segment, 1.0, mutex_fn)
            instr.record_and(support is not None)
            if support is None:
                continue
            for f_add, add_prob in getattr(model, "add_probabilities", {}).items():
                new_path = Path(support.segments, _clamp01(support.prob * _clamp01(add_prob)))
                pending.setdefault((arrival, f_add), []).append(new_path)

    if capture_paths is not None:
        for t, fmap in paths_by_layer.items():
            capture_paths[t] = {f: list(pl) for f, pl in fmap.items()}

    return prob_by_layer, instr
