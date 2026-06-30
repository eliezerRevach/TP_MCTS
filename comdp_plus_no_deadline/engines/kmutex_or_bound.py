"""
Mutex-aware K-bounded OR-layer tightening for the admissible PTRPG.

Heuristic name (experiments.ipynb / strategy key): ``baseline_admissible_kmutex``.

Status: ADMISSIBLE w.r.t. the union-bound baseline. This is an OPT-IN tightening
of the OR/fact layer used by ``baseline_admissible`` (``union_bound_or_hazard``,
``H_f = min(1, sum_e B_e)``). The union bound stays available unchanged; when no
two achievers landing at a fact-layer cell are certified mutex this module returns
exactly that union bound, so the strategy degrades to ``baseline_admissible``.

------------------------------------------------------------------------------
Idea
------------------------------------------------------------------------------
The union bound double-counts achievers that cannot both fire — e.g. several
machine-shop operations that all need the same machine (``free(m)``, exclusive
achievers). If two achievers are certified MUTEX we may take ``max`` over them
instead of ``sum``; ``max <= sum`` so the bound can only get tighter (lower),
never rise above the union baseline (the "admissibility direction").

------------------------------------------------------------------------------
Per-cell rows (the whole mechanism)
------------------------------------------------------------------------------
For one fact-layer cell we keep <= K ROWS. Each row is ``(prob, footprint)``:

- ``prob``     : a SUM of the support contributions ``B_e`` folded into the row.
- ``footprint``: the set of support identities this row is *certified mutex with*.
                 ``footprint`` only ever SHRINKS — never grows.

Insertion / merge rule, processing the cell's supports:

- A MUTEX support (mutex with at least one other support in the cell) seeds its
  own row, with ``footprint`` = the identities it is mutex with.
- A FREE support (mutex with nothing in the cell) folds (SUMS) into one existing
  row and CLEARS that row's footprint permanently — once a free, non-mutex mass
  is mixed in, the row can no longer be certified mutex with anyone, so it must
  rejoin the union (sum) side. This is exactly why a single no-mutex achiever
  "forgets" a mutex collapse (the documented naive behaviour).
- Over K rows: merge the pair with the smallest footprint-disagreement
  (symmetric-difference) first; the merged row sums the probs (conservative,
  stays <= union) and intersects the footprints.

Aggregation (HARD INVARIANT: max-vs-sum is decided from ``footprint`` ALONE,
never by re-inspecting which elements a row holds):

    H_f <= ( sum of rows whose footprint is EMPTY )            # free / union side
         + ( max over rows whose footprint is NON-EMPTY )      # surviving mutex clique

All non-empty-footprint rows are treated as one surviving clique (the naive
single-clique model). A lone non-empty row contributes itself (``max`` of one),
so it can never under-count below its own mass.

Reductions:
- all supports mutex (one clique, no free mass) -> ``max`` over them.
- all supports free                              -> ``sum`` == union baseline.
- ``a, b`` mutex and ``c`` free                  -> ``c`` folds into one of the
  two rows and clears it, leaving a lone mutex row; result == union (``a+b+c``),
  i.e. NOT an under-count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, List, Optional, Sequence, Set, Tuple


Identity = Hashable


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class _Row:
    """One row of a fact-layer cell (see module docstring)."""

    prob: float
    footprint: Set[Identity]
    members: Set[Identity] = field(default_factory=set)


@dataclass
class KMutexORResult:
    """OR-layer hazard plus per-cell instrumentation for the survival metric."""

    value: float            # H_f: tightened hazard, always <= union_value.
    union_value: float      # baseline union bound min(1, sum B_e), for comparison.
    n_supports: int         # number of achiever contributions landing at the cell.
    n_rows: int             # rows kept after the K bound.
    n_mutex_rows: int       # rows with a non-empty footprint at aggregation.
    clique_survived: bool   # a pure mutex clique of size >= 2 survived.
    max_clique_size: int    # size of the surviving clique (== n_mutex_rows).


def _merge_closest_pair(rows: List[_Row]) -> None:
    """Collapse the two most-similar rows to respect the K bound.

    "Most similar" = smallest footprint symmetric-difference. The merged row
    SUMS the probs (conservative: keeps the mass on the union side, so the
    aggregate never drops below the true relaxed value more than the union bound
    already allows) and INTERSECTS the footprints (footprints only shrink).
    """
    best_i = 0
    best_j = 1
    best_cost: Optional[int] = None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            cost = len(rows[i].footprint ^ rows[j].footprint)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_i, best_j = i, j
    a = rows[best_i]
    b = rows[best_j]
    merged = _Row(
        prob=_clamp01(a.prob + b.prob),
        footprint=set(a.footprint & b.footprint),
        members=set(a.members | b.members),
    )
    # Remove the higher index first so the lower index stays valid.
    rows.pop(best_j)
    rows.pop(best_i)
    rows.append(merged)


def _fold_free_support(rows: List[_Row], identity: Identity, prob: float) -> None:
    """Fold a free (non-mutex) support into an existing row, clearing its footprint.

    Free supports never consume a slot. If any mutex (non-empty-footprint) row
    exists, fold into the SMALLEST-prob one and clear its footprint permanently —
    this is the "no-mutex action forgets the clique" behaviour, and it is what
    keeps the bound from under-counting when a free achiever coexists with a
    mutex pair. With no rows yet, the free support seeds the first (already-free)
    row; with only free rows, it sums into the smallest of them.
    """
    if not rows:
        rows.append(_Row(prob=_clamp01(prob), footprint=set(), members={identity}))
        return
    non_empty = [r for r in rows if r.footprint]
    if non_empty:
        target = min(non_empty, key=lambda r: (r.prob, len(r.footprint)))
        target.prob = _clamp01(target.prob + prob)
        target.footprint.clear()  # permanent: footprints only ever shrink.
        target.members.add(identity)
    else:
        target = min(rows, key=lambda r: r.prob)
        target.prob = _clamp01(target.prob + prob)
        target.members.add(identity)


def kmutex_or_hazard(
    supports: Sequence[Tuple[Identity, float]],
    mutex_fn: Callable[[Identity, Identity], bool],
    k: int,
) -> KMutexORResult:
    """Tightened OR-layer hazard for one fact-layer cell.

    ``supports`` are the per-achiever contributions ``(identity, B_e)`` arriving
    at the fact this layer; ``mutex_fn(id_a, id_b)`` certifies a structural mutex
    between two achievers; ``k`` caps the rows kept per cell. The returned
    ``value`` is always ``<=`` the union bound over the same supports.
    """
    k = max(1, int(k))
    n = len(supports)
    identities = [s[0] for s in supports]
    probs = [_clamp01(s[1]) for s in supports]
    union_value = _clamp01(min(1.0, sum(probs)))

    if n <= 1:
        return KMutexORResult(
            value=union_value,
            union_value=union_value,
            n_supports=n,
            n_rows=n,
            n_mutex_rows=0,
            clique_survived=False,
            max_clique_size=0,
        )

    # Pairwise mutex among the cell's supports (by index), symmetric by build.
    mutex_sets: List[Set[Identity]] = [set() for _ in range(n)]
    any_mutex = False
    for i in range(n):
        for j in range(i + 1, n):
            if mutex_fn(identities[i], identities[j]):
                mutex_sets[i].add(identities[j])
                mutex_sets[j].add(identities[i])
                any_mutex = True

    if not any_mutex:
        # No certified mutex anywhere -> exactly the union baseline.
        return KMutexORResult(
            value=union_value,
            union_value=union_value,
            n_supports=n,
            n_rows=n,
            n_mutex_rows=0,
            clique_survived=False,
            max_clique_size=0,
        )

    mutex_indices = [i for i in range(n) if mutex_sets[i]]
    free_indices = [i for i in range(n) if not mutex_sets[i]]

    # Seed one row per mutex support (largest contribution claims a slot first),
    # enforcing the K bound by merging the closest pair as we go.
    rows: List[_Row] = []
    for i in sorted(mutex_indices, key=lambda i: (-probs[i], str(identities[i]))):
        rows.append(
            _Row(
                prob=probs[i],
                footprint=set(mutex_sets[i]),
                members={identities[i]},
            )
        )
        while len(rows) > k:
            _merge_closest_pair(rows)

    # Fold the free supports in; each clears at most one mutex row's footprint.
    for i in sorted(free_indices, key=lambda i: (-probs[i], str(identities[i]))):
        _fold_free_support(rows, identities[i], probs[i])

    # Aggregate strictly from footprints: empty -> union side (sum), non-empty ->
    # the surviving mutex clique (max).
    free_sum = sum(r.prob for r in rows if not r.footprint)
    mutex_rows = [r for r in rows if r.footprint]
    clique_max = max((r.prob for r in mutex_rows), default=0.0)
    value = _clamp01(min(1.0, free_sum + clique_max))

    # Admissibility direction guard: the tightening must never exceed the union.
    if value > union_value:
        value = union_value

    return KMutexORResult(
        value=value,
        union_value=union_value,
        n_supports=n,
        n_rows=len(rows),
        n_mutex_rows=len(mutex_rows),
        clique_survived=len(mutex_rows) >= 2,
        max_clique_size=len(mutex_rows),
    )


@dataclass
class KMutexInstrumentation:
    """Accumulates the headline survival metric across many OR-node evaluations.

    The headline number is ``clique_survival_fraction``: of the OR-nodes that had
    at least two achievers (so a mutex collapse was even possible), the fraction
    where a pure mutex clique of size >= 2 actually survived to aggregation.
    """

    or_nodes_total: int = 0          # cells evaluated with >= 2 supports.
    or_nodes_with_mutex_pair: int = 0  # >= 1 certified mutex pair present.
    or_nodes_clique_survived: int = 0  # >= 2 non-empty-footprint rows survived.
    max_clique_size_seen: int = 0
    tightened_below_union: int = 0   # cells where value < union (strictly tighter).
    sum_union_minus_value: float = 0.0  # total mass shaved off the union bound.

    def record(self, result: KMutexORResult) -> None:
        if result.n_supports < 2:
            return
        self.or_nodes_total += 1
        if result.n_mutex_rows >= 1:
            self.or_nodes_with_mutex_pair += 1
        if result.clique_survived:
            self.or_nodes_clique_survived += 1
        if result.max_clique_size > self.max_clique_size_seen:
            self.max_clique_size_seen = result.max_clique_size
        if result.value < result.union_value - 1e-12:
            self.tightened_below_union += 1
            self.sum_union_minus_value += (result.union_value - result.value)

    @property
    def clique_survival_fraction(self) -> float:
        if self.or_nodes_total == 0:
            return 0.0
        return self.or_nodes_clique_survived / self.or_nodes_total

    def summary(self) -> str:
        return (
            "[kmutex] OR-nodes(>=2 supports)={total} "
            "with_mutex_pair={pair} clique>=2_survived={surv} "
            "survival_fraction={frac:.4f} max_clique={maxc} "
            "tightened<union={tight} mass_shaved={mass:.4f}".format(
                total=self.or_nodes_total,
                pair=self.or_nodes_with_mutex_pair,
                surv=self.or_nodes_clique_survived,
                frac=self.clique_survival_fraction,
                maxc=self.max_clique_size_seen,
                tight=self.tightened_below_union,
                mass=self.sum_union_minus_value,
            )
        )
