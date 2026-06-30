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

OR layer
--------
Alternative achiever-paths of a fact are grouped into mutex cliques (connected
components of the path-mutex graph). Within a clique only one path can happen, so
we take ``max``; across cliques (mutually compatible) we ``sum`` (union bound).
``P(f) <= sum_components( max_{path in component} prob(path) )``.

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
    """True iff any segment of ``p`` conflicts with any segment of ``q``."""
    for s1 in p.segments:
        for s2 in q.segments:
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


def _mutex_components(paths: Sequence[Path], mutex_fn) -> List[List[int]]:
    """Connected components of the path-mutex graph (each is a mutex clique)."""
    n = len(paths)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if paths_mutex(paths[i], paths[j], mutex_fn):
                union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


@dataclass
class ORAggregateResult:
    value: float
    n_paths: int
    n_components: int
    clique_survived: bool      # a mutex component of size >= 2 (max-collapse happened).
    max_component_size: int


def or_aggregate_paths(
    paths: Sequence[Path],
    mutex_fn: Callable[[Action, Action], bool],
    k: int,
) -> ORAggregateResult:
    """OR-combine achiever-paths: ``sum`` across mutex components of their ``max``.

    Paths are first capped to the top ``k`` by probability. Mutex paths (any
    overlapping mutex segment pair) fall into one component and contribute a
    single ``max``; mutually-compatible paths sum (union bound).
    """
    kept = _dedup_topk(paths, k)
    if not kept:
        return ORAggregateResult(0.0, 0, 0, False, 0)
    components = _mutex_components(kept, mutex_fn)
    total = 0.0
    max_size = 0
    survived = False
    for comp in components:
        total += max(kept[i].prob for i in comp)
        if len(comp) > max_size:
            max_size = len(comp)
        if len(comp) >= 2:
            survived = True
    return ORAggregateResult(
        value=_clamp01(min(1.0, total)),
        n_paths=len(kept),
        n_components=len(components),
        clique_survived=survived,
        max_component_size=max_size,
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
    or_nodes_clique_survived: int = 0  # cells with a mutex component of size >= 2.
    max_component_size_seen: int = 0
    and_paths_built: int = 0          # conjunctive paths successfully built.
    and_paths_blocked: int = 0        # conjunctive combos dropped as infeasible.

    def record_or(self, res: ORAggregateResult) -> None:
        if res.n_paths < 2:
            return
        self.or_nodes_total += 1
        if res.clique_survived:
            self.or_nodes_clique_survived += 1
        if res.max_component_size > self.max_component_size_seen:
            self.max_component_size_seen = res.max_component_size

    def record_and(self, built: bool) -> None:
        if built:
            self.and_paths_built += 1
        else:
            self.and_paths_blocked += 1

    @property
    def clique_survival_fraction(self) -> float:
        if self.or_nodes_total == 0:
            return 0.0
        return self.or_nodes_clique_survived / self.or_nodes_total

    def summary(self) -> str:
        return (
            "[pathmutex] OR-nodes(>=2 paths)={t} clique>=2_survived={s} "
            "survival_fraction={f:.4f} max_component={m} "
            "AND_built={ab} AND_blocked={bl}".format(
                t=self.or_nodes_total,
                s=self.or_nodes_clique_survived,
                f=self.clique_survival_fraction,
                m=self.max_component_size_seen,
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
                layer_paths[f] = _dedup_topk(layer_paths.get(f, []) + plist, k)
        # Arrivals scheduled to land at this layer.
        for (arrival, f), plist in pending.items():
            if arrival == t:
                layer_paths[f] = _dedup_topk(layer_paths.get(f, []) + plist, k)

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

    return prob_by_layer, instr
