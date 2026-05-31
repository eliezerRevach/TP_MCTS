"""
AND-layer gamma correction for the PTRPG / survival heuristic.

The relaxed RPG approximates an action's precondition support (the AND layer) by
assuming the preconditions are *independent*:

    R(a) = Π_{f in Pre(a)} P_{t-1}(f)

That independence assumption is the dominant ranking bias of the AND layer:
preconditions that share an achiever (positive correlation) or that delete /
contradict each other (negative correlation) are mis-estimated. This module
replaces the flat product with a *component-wise* corrected estimate:

    R_gamma(a) = Π_{C in Components(Pre(a))} [ gamma(key(C)) * Π_{f in C} P_{t-1}(f) ]

where ``Components(Pre(a))`` are the separable dependency components of the
preconditions (connected components of a cheap static-dependency graph), and
``gamma(key(C))`` is a per-component-type multiplicative correction. Because the
inner products over the components multiply back to the original flat product,
``R_gamma(a) = R(a) * Π_C gamma(key(C))`` — i.e. the correction is a single
multiplicative *factor* on top of the existing product, so singleton /
independent components (gamma = 1) leave the estimate exactly equal to baseline.

Design goals (see the spec this implements):
  * The estimate is identical to the flat product when no dependency is detected.
  * Correction is *local* to detected components, never a full joint over Pre(a).
  * gamma defaults are static research values, overridable via config.
  * Optional lazy rollout calibration refines gamma from reachable rollout states;
    one rollout trace updates many candidate pairs (samples are not wasted), with
    absence data, confidence gating, and shrinkage toward the static default.
  * The goal is better heuristic *ranking*, not a calibrated joint probability.

Nothing here touches MCTS, the deadline/STN logic, the noisy-OR achiever formula,
or the survival/delete update — only the AND-layer precondition estimate.
"""

from __future__ import annotations

import heapq
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

Fact = Hashable

# Component / pair dependency types.
SINGLETON = "singleton"
MUTEX = "mutex"
NEGATIVE = "negative"
POSITIVE = "positive"
UNKNOWN = "unknown"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AndGammaConfig:
    """Tunable constants for the AND-layer gamma correction.

    The gamma defaults are initial *research* values, not final constants; tune
    them per domain. Budgets bound the optional rollout calibration so it cannot
    dominate an MCTS decision.
    """

    # --- Default static gamma table (per component type). ------------------
    gamma_singleton: float = 1.0
    gamma_unknown: float = 1.0
    gamma_positive: float = 1.2  # shared-achiever / causal-support => under-counted
    gamma_negative: float = 0.7  # delete / resource-conflict => over-counted
    gamma_mutex: float = 0.3  # contradiction => strongly over-counted

    # --- Stability cap on the gamma *factor* (NOT clipping of R(a)). -------
    gamma_min: float = 0.1
    gamma_max: float = 3.0
    epsilon: float = 1e-9

    # --- Lazy rollout calibration (off by default => pure static gamma). ---
    enable_rollout_calibration: bool = False
    initial_batch: int = 16  # rollouts on first low-confidence encounter
    incremental_batch: int = 8  # rollouts on a repeated low-confidence key
    max_samples_per_key: int = 128  # stop refining a key past this many samples
    max_total_rollouts: int = 2000  # global budget proxy for "calibration time"
    rollout_horizon: int = 30
    max_calibration_seconds: float = 0.0  # 0 => unbounded (rely on rollout caps)

    # --- Confidence / shrinkage. -------------------------------------------
    n0: int = 50  # shrinkage strength: lambda = n / (n + n0)
    min_n: int = 30  # below this a key is "low confidence"
    min_count_fg: int = 5  # min co-occurrence count for an exact-pair estimate
    min_count_downstream: int = 20  # min P(downstream) count for conditional gamma

    # --- Candidate-pair update budget. -------------------------------------
    max_pair_updates_per_state: int = 100
    max_candidate_pairs: int = 5000

    # --- Exact-pair cache gating. ------------------------------------------
    exact_pair_min_n: int = 30

    # --- Misc. -------------------------------------------------------------
    use_object_edges: bool = False  # enable best-effort shared-object edges
    seed: Optional[int] = 0
    fallback_probabilistic_add_prob: float = 1.0

    def default_gamma(self, comp_type: str) -> float:
        return {
            SINGLETON: self.gamma_singleton,
            UNKNOWN: self.gamma_unknown,
            POSITIVE: self.gamma_positive,
            NEGATIVE: self.gamma_negative,
            MUTEX: self.gamma_mutex,
        }.get(comp_type, 1.0)

    def clamp_gamma(self, gamma: float) -> float:
        return max(self.gamma_min, min(self.gamma_max, float(gamma)))


# ---------------------------------------------------------------------------
# Static dependency structure
# ---------------------------------------------------------------------------


@dataclass
class ComponentInfo:
    """A separable dependency component inside an action's preconditions."""

    facts: frozenset  # the facts in the component
    comp_type: str  # singleton | positive | negative | mutex | unknown
    size: int
    key: Tuple[str, int]  # (comp_type, size) -- the broad gamma cache key
    edge_kinds: frozenset = frozenset()


@dataclass
class StructuralContext:
    """Cheap static-dependency relations between facts, extracted from actions.

    Everything here is computed once from the (delete-relaxed) action models plus
    the delete effects, and is independent of state/layer. Edge detection answers
    "is there cheap static evidence that facts f and g are dependent, and of what
    sign?" — used both to form precondition components and to classify candidate
    calibration pairs.
    """

    pre_by_name: Dict[str, frozenset]
    add_by_name: Dict[str, frozenset]
    del_by_name: Dict[str, frozenset]
    achievers_by_fact: Dict[Fact, Set[str]]  # fact -> action names that add it
    deleters_by_fact: Dict[Fact, Set[str]]  # fact -> action names that delete it
    achiever_pre_by_fact: Dict[Fact, Set[Fact]]  # fact -> ∪ pre(achievers(fact))
    object_token_fn: Optional[Callable[[Fact], Set[Hashable]]] = None
    use_object_edges: bool = False

    def _toggles(self, deleted: Fact, produced: Fact) -> bool:
        """∃ action that deletes ``deleted`` and adds ``produced``."""
        for name in self.deleters_by_fact.get(deleted, ()):  # type: ignore[arg-type]
            if produced in self.add_by_name.get(name, frozenset()):
                return True
        return False

    def _deletes_or_negates(self, deleted: Fact, other: Fact) -> bool:
        """∃ action that deletes ``deleted`` and requires or produces ``other``."""
        for name in self.deleters_by_fact.get(deleted, ()):  # type: ignore[arg-type]
            if other in self.pre_by_name.get(name, frozenset()):
                return True
            if other in self.add_by_name.get(name, frozenset()):
                return True
        return False

    def _shared_achiever(self, f: Fact, g: Fact) -> bool:
        return bool(self.achievers_by_fact.get(f, set()) & self.achievers_by_fact.get(g, set()))

    def _causal(self, f: Fact, g: Fact) -> bool:
        """One fact is a precondition of an achiever of the other."""
        if f in self.achiever_pre_by_fact.get(g, set()):
            return True
        if g in self.achiever_pre_by_fact.get(f, set()):
            return True
        return False

    def _shared_object(self, f: Fact, g: Fact) -> bool:
        if not self.use_object_edges or self.object_token_fn is None:
            return False
        try:
            tf = self.object_token_fn(f)
            tg = self.object_token_fn(g)
        except Exception:
            return False
        return bool(tf and tg and (tf & tg))

    def pair_edges(self, f: Fact, g: Fact) -> Set[str]:
        """Static dependency edge-kinds between two facts (possibly empty)."""
        kinds: Set[str] = set()
        if self._shared_achiever(f, g) or self._causal(f, g):
            kinds.add(POSITIVE)
        if self._deletes_or_negates(f, g) or self._deletes_or_negates(g, f):
            kinds.add(NEGATIVE)
        if self._toggles(f, g) and self._toggles(g, f):
            kinds.add(MUTEX)
        if not kinds and self._shared_object(f, g):
            kinds.add("object")
        return kinds

    def causal_direction(self, f: Fact, g: Fact) -> Optional[Tuple[Fact, Fact]]:
        """Return (upstream, downstream) when one fact causally supports the other."""
        f_supports_g = f in self.achiever_pre_by_fact.get(g, set())
        g_supports_f = g in self.achiever_pre_by_fact.get(f, set())
        if f_supports_g and not g_supports_f:
            return (f, g)
        if g_supports_f and not f_supports_g:
            return (g, f)
        return None


def classify_component(edge_kinds: Iterable[str], size: int) -> str:
    """Classify a component from the union of its internal edge kinds.

    Priority: mutex > negative > positive > unknown. A component with no internal
    edges (a lone fact) is a singleton.
    """
    if size <= 1:
        return SINGLETON
    kinds = set(edge_kinds)
    if MUTEX in kinds:
        return MUTEX
    if NEGATIVE in kinds:
        return NEGATIVE
    if POSITIVE in kinds:
        return POSITIVE
    return UNKNOWN


def build_components(
    preconditions: Sequence[Fact],
    ctx: StructuralContext,
) -> List[ComponentInfo]:
    """Connected dependency components of an action's preconditions.

    Two preconditions are connected when ``ctx.pair_edges`` reports any static
    dependency between them. Facts with no detected dependency stay separate
    singletons, so the corrected estimate collapses to the flat product.
    """
    facts = list(preconditions)
    n = len(facts)
    if n == 0:
        return []
    if n == 1:
        return [
            ComponentInfo(
                facts=frozenset(facts),
                comp_type=SINGLETON,
                size=1,
                key=(SINGLETON, 1),
            )
        ]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pair_kinds: Dict[Tuple[int, int], Set[str]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            kinds = ctx.pair_edges(facts[i], facts[j])
            if kinds:
                union(i, j)
                pair_kinds[(i, j)] = kinds

    members: Dict[int, List[int]] = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)

    components: List[ComponentInfo] = []
    for idxs in members.values():
        idxset = set(idxs)
        kinds: Set[str] = set()
        for (i, j), k in pair_kinds.items():
            if i in idxset and j in idxset:
                kinds |= k
        size = len(idxs)
        comp_type = classify_component(kinds, size)
        components.append(
            ComponentInfo(
                facts=frozenset(facts[i] for i in idxs),
                comp_type=comp_type,
                size=size,
                key=(comp_type, size),
                edge_kinds=frozenset(kinds),
            )
        )
    return components


# ---------------------------------------------------------------------------
# Rollout statistics
# ---------------------------------------------------------------------------


@dataclass
class PairStat:
    """Co-occurrence counts for a candidate dependent pair.

    ``n`` counts *every* sampled state the pair was tracked in — including states
    where both facts are false — so the marginals are unconditional. Dropping
    absence data would condition the estimate and bias it.
    """

    n: int = 0
    both: int = 0
    singles: Dict[Fact, int] = field(default_factory=dict)

    def p(self, fact: Fact) -> float:
        return (self.singles.get(fact, 0) / self.n) if self.n > 0 else 0.0

    def p_both(self) -> float:
        return (self.both / self.n) if self.n > 0 else 0.0


# ---------------------------------------------------------------------------
# Rollout simulator (valid reachable states; calibration only)
# ---------------------------------------------------------------------------


@dataclass
class _SimAction:
    name: str
    pos: frozenset
    neg: frozenset
    add: frozenset
    dels: frozenset
    prob_effects: tuple


class RolloutSimulator:
    """Light forward simulator producing *valid reachable* rollout traces.

    Durations / concurrency / the STN are intentionally ignored — calibration
    only needs co-occurrence statistics over reachable truth assignments, so
    actions are applied instantaneously, one per step, respecting their real
    positive/negative preconditions and add/delete/probabilistic effects. This
    never feeds invalid random truth assignments to the statistics.
    """

    def __init__(
        self,
        actions: Sequence[object],
        rng: random.Random,
        horizon: int,
        fallback_add_prob: float = 1.0,
    ):
        self._rng = rng
        self._horizon = max(1, int(horizon))
        self._fallback_add_prob = float(fallback_add_prob)
        self._actions = self._compile(actions)

    @staticmethod
    def _compile(actions: Sequence[object]) -> List[_SimAction]:
        compiled: List[_SimAction] = []
        for action in actions:
            if hasattr(action, "actions") and not hasattr(action, "add_effects"):
                continue
            compiled.append(
                _SimAction(
                    name=getattr(action, "name", repr(action)),
                    pos=frozenset(getattr(action, "pos_preconditions", set())),
                    neg=frozenset(getattr(action, "neg_preconditions", set())),
                    add=frozenset(getattr(action, "add_effects", set())),
                    dels=frozenset(getattr(action, "del_effects", set())),
                    prob_effects=tuple(getattr(action, "probabilistic_effects", []) or []),
                )
            )
        return compiled

    def _sample_prob_effect(self, prob_effect, state: Set[Fact]) -> None:
        try:
            outcomes = prob_effect.probability_function(
                SimpleNamespace(predicates=set(state)), None
            )
        except Exception:
            outcomes = None
        if not outcomes:
            # Opaque outcome: optimistically add the structurally-declared fluents.
            for fact in getattr(prob_effect, "fluents", []):
                if self._rng.random() <= self._fallback_add_prob:
                    state.add(fact)
            return
        items = list(outcomes.items())
        roll = self._rng.random()
        cumulative = 0.0
        chosen = None
        for probability, assignments in items:
            cumulative += _clamp01(probability)
            if roll <= cumulative:
                chosen = assignments
                break
        if chosen is None:
            chosen = items[-1][1]
        for fact, value in chosen.items():
            is_true = bool(value.bool_constant_value()) if hasattr(
                value, "bool_constant_value"
            ) else bool(value)
            if is_true:
                state.add(fact)
            else:
                state.discard(fact)

    def rollout(self, start_facts: Iterable[Fact]) -> List[frozenset]:
        state: Set[Fact] = set(start_facts)
        trace: List[frozenset] = [frozenset(state)]
        for _ in range(self._horizon):
            applicable = [
                sim
                for sim in self._actions
                if sim.pos <= state and not (sim.neg & state)
            ]
            if not applicable:
                break
            sim = self._rng.choice(applicable)
            state = (state - sim.dels) | sim.add
            for prob_effect in sim.prob_effects:
                self._sample_prob_effect(prob_effect, state)
            trace.append(frozenset(state))
        return trace


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class AndGammaCalibrator:
    """Owns the gamma caches, candidate-pair statistics and (optional) rollouts.

    Default gamma comes from the static table; when rollout calibration is on,
    ``gamma_for_component`` blends a learned estimate toward the default with
    shrinkage ``lambda = n / (n + n0)`` so sparse keys stay close to default.
    """

    def __init__(
        self,
        config: AndGammaConfig,
        ctx: StructuralContext,
        candidate_pairs: Iterable[frozenset],
        simulator: Optional[RolloutSimulator] = None,
    ):
        self.config = config
        self.ctx = ctx
        self._simulator = simulator

        # Candidate dependency universe + per-pair classification.
        self.candidate_pairs: List[frozenset] = list(candidate_pairs)
        self.pair_type: Dict[frozenset, str] = {}
        self.pairs_by_type: Dict[str, List[frozenset]] = {}
        for pair in self.candidate_pairs:
            members = tuple(pair)
            if len(members) != 2:
                continue
            kinds = ctx.pair_edges(members[0], members[1])
            ptype = classify_component(kinds, 2)
            self.pair_type[pair] = ptype
            self.pairs_by_type.setdefault(ptype, []).append(pair)

        self.pair_stats: Dict[frozenset, PairStat] = {}

        # gamma memo (keyed by the component fact-set) + invalidation flag.
        self._gamma_memo: Dict[frozenset, float] = {}

        # Diagnostics counters.
        self.and_calls: int = 0
        self.components_found: int = 0
        self.type_counts: Counter = Counter()
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.calibration_calls: int = 0
        self.rollouts_used: int = 0
        self.traces_reused: int = 0
        self.pair_updates: int = 0
        self.exact_used: int = 0
        self.broad_used: int = 0
        self.calibration_seconds: float = 0.0

    # -- gamma lookup -------------------------------------------------------

    def gamma_for_component(self, comp: ComponentInfo) -> float:
        """Return the multiplicative gamma factor for one component."""
        if comp.comp_type == SINGLETON:
            return 1.0
        memo_key = comp.facts
        cached = self._gamma_memo.get(memo_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        gamma = self._compute_gamma(comp)
        self._gamma_memo[memo_key] = gamma
        return gamma

    def _compute_gamma(self, comp: ComponentInfo) -> float:
        config = self.config
        default = config.default_gamma(comp.comp_type)
        learned: Optional[float] = None
        n = 0
        if comp.size == 2:
            pair = comp.facts
            stat = self.pair_stats.get(pair)
            if (
                stat is not None
                and stat.n >= config.exact_pair_min_n
                and stat.both >= config.min_count_fg
            ):
                learned = self._pair_gamma(pair, stat)
                n = stat.n
                self.exact_used += 1
            else:
                broad, broad_n = self._broad_gamma(comp.key)
                if broad is not None:
                    learned, n = broad, broad_n
                    self.broad_used += 1
        # size != 2 (or no samples): no pairwise learned estimate -> static default.
        if learned is None or n <= 0:
            return config.clamp_gamma(default)
        lam = n / (n + config.n0)
        blended = lam * learned + (1.0 - lam) * default
        return config.clamp_gamma(blended)

    def _pair_gamma(self, pair: frozenset, stat: PairStat) -> float:
        config = self.config
        members = tuple(pair)
        f, g = members[0], members[1]
        # Conditional gamma for a known causal direction with a well-sampled
        # downstream fact: P(upstream | downstream) / P(upstream).
        direction = self.ctx.causal_direction(f, g)
        if direction is not None:
            upstream, downstream = direction
            c_down = stat.singles.get(downstream, 0)
            if c_down >= config.min_count_downstream:
                p_up_given_down = stat.both / (c_down + config.epsilon)
                p_up = stat.p(upstream)
                return p_up_given_down / (p_up + config.epsilon)
        p_fg = stat.p_both()
        p_f = stat.p(f)
        p_g = stat.p(g)
        return p_fg / (p_f * p_g + config.epsilon)

    def _broad_gamma(self, key: Tuple[str, int]) -> Tuple[Optional[float], int]:
        comp_type, size = key
        if size != 2:
            return None, 0
        numerator = 0.0
        denominator = 0.0
        total_n = 0
        for pair in self.pairs_by_type.get(comp_type, ()):  # type: ignore[arg-type]
            stat = self.pair_stats.get(pair)
            if stat is None or stat.n <= 0:
                continue
            members = tuple(pair)
            numerator += stat.p_both()
            denominator += stat.p(members[0]) * stat.p(members[1])
            total_n += stat.n
        if total_n <= 0 or denominator <= 0.0:
            return None, total_n
        return numerator / (denominator + self.config.epsilon), total_n

    # -- confidence ---------------------------------------------------------

    def _key_low_confidence(self, key: Tuple[str, int]) -> bool:
        comp_type, size = key
        if size != 2:
            return False  # we only ever learn pairwise gamma
        _, total_n = self._broad_gamma(key)
        return total_n < self.config.min_n

    # -- calibration --------------------------------------------------------

    def calibrate(self, start_facts: Iterable[Fact], needed_keys: Iterable[Tuple[str, int]]) -> None:
        """Lazily refine gamma for the requested keys from rollout traces.

        Runs at most one rollout batch from ``start_facts`` when calibration is
        enabled, some requested key is low-confidence, and budget remains. A
        single batch updates *all* candidate pairs (one trace -> many pair
        observations), so the requested keys and every other tracked pair benefit
        together.
        """
        config = self.config
        if not config.enable_rollout_calibration or self._simulator is None:
            return
        if self.rollouts_used >= config.max_total_rollouts:
            return
        keys = list(needed_keys)
        if not any(self._key_low_confidence(k) for k in keys):
            return

        start = frozenset(start_facts)
        batch = config.initial_batch
        t0 = time.perf_counter()
        for _ in range(batch):
            if self.rollouts_used >= config.max_total_rollouts:
                break
            if (
                config.max_calibration_seconds > 0.0
                and (time.perf_counter() - t0) > config.max_calibration_seconds
            ):
                break
            trace = self._simulator.rollout(start)
            self.ingest_trace(trace)
            self.rollouts_used += 1
        self.calibration_calls += 1
        self.calibration_seconds += time.perf_counter() - t0
        # New evidence invalidates the gamma memo.
        self._gamma_memo.clear()

    def ingest_trace(self, trace: Sequence[Iterable[Fact]]) -> None:
        """Update candidate-pair statistics from one rollout trace.

        Reuses the whole trace: every sampled state contributes to many candidate
        pairs, not only the pair that triggered calibration. Updates are capped
        and prioritized (lowest-``n`` pairs first) so cost stays bounded and we
        never blindly fan out over all O(n^2) fact pairs.
        """
        self.traces_reused += 1
        for state in trace:
            self._update_state(set(state))
        self._gamma_memo.clear()

    def _pairs_to_update(self) -> List[frozenset]:
        cap = self.config.max_pair_updates_per_state
        if len(self.candidate_pairs) <= cap:
            return self.candidate_pairs
        # Prioritize the least-sampled (lowest-confidence) candidate pairs.
        return heapq.nsmallest(
            cap,
            self.candidate_pairs,
            key=lambda pair: self.pair_stats.get(pair, _EMPTY_STAT).n,
        )

    def _update_state(self, state: Set[Fact]) -> None:
        for pair in self._pairs_to_update():
            members = tuple(pair)
            if len(members) != 2:
                continue
            f, g = members[0], members[1]
            stat = self.pair_stats.get(pair)
            if stat is None:
                stat = PairStat()
                self.pair_stats[pair] = stat
            stat.n += 1  # absence data: increments even when both facts are false
            f_true = f in state
            g_true = g in state
            if f_true:
                stat.singles[f] = stat.singles.get(f, 0) + 1
            if g_true:
                stat.singles[g] = stat.singles.get(g, 0) + 1
            if f_true and g_true:
                stat.both += 1
            self.pair_updates += 1

    # -- diagnostics --------------------------------------------------------

    def avg_gamma_by_type(self) -> Dict[str, float]:
        sums: Dict[str, float] = {}
        counts: Dict[str, float] = {}
        for facts, gamma in self._gamma_memo.items():
            kinds = self.pair_type.get(facts)
            ctype = kinds if kinds is not None else UNKNOWN
            sums[ctype] = sums.get(ctype, 0.0) + gamma
            counts[ctype] = counts.get(ctype, 0.0) + 1.0
        return {t: sums[t] / counts[t] for t in sums if counts[t] > 0}

    def diagnostics(self) -> Dict[str, object]:
        return {
            "and_calls": self.and_calls,
            "components_found": self.components_found,
            "component_type_counts": dict(self.type_counts),
            "candidate_pairs": len(self.candidate_pairs),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "calibration_calls": self.calibration_calls,
            "rollouts_used": self.rollouts_used,
            "traces_reused": self.traces_reused,
            "pair_updates": self.pair_updates,
            "exact_pair_gamma_used": self.exact_used,
            "broad_gamma_used": self.broad_used,
            "calibration_seconds": self.calibration_seconds,
            "avg_gamma_by_type": self.avg_gamma_by_type(),
        }


_EMPTY_STAT = PairStat()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_structural_context(
    *,
    pre_by_name: Mapping[str, Iterable[Fact]],
    add_by_name: Mapping[str, Iterable[Fact]],
    del_by_name: Mapping[str, Iterable[Fact]],
    config: AndGammaConfig,
    object_token_fn: Optional[Callable[[Fact], Set[Hashable]]] = None,
) -> StructuralContext:
    """Assemble a ``StructuralContext`` from per-action precondition/effect maps."""
    pre_fz = {name: frozenset(v) for name, v in pre_by_name.items()}
    add_fz = {name: frozenset(v) for name, v in add_by_name.items()}
    del_fz = {name: frozenset(v) for name, v in del_by_name.items()}

    achievers_by_fact: Dict[Fact, Set[str]] = {}
    for name, adds in add_fz.items():
        for fact in adds:
            achievers_by_fact.setdefault(fact, set()).add(name)

    deleters_by_fact: Dict[Fact, Set[str]] = {}
    for name, dels in del_fz.items():
        for fact in dels:
            deleters_by_fact.setdefault(fact, set()).add(name)

    achiever_pre_by_fact: Dict[Fact, Set[Fact]] = {}
    for fact, names in achievers_by_fact.items():
        union: Set[Fact] = set()
        for name in names:
            union |= pre_fz.get(name, frozenset())
        achiever_pre_by_fact[fact] = union

    return StructuralContext(
        pre_by_name=pre_fz,
        add_by_name=add_fz,
        del_by_name=del_fz,
        achievers_by_fact=achievers_by_fact,
        deleters_by_fact=deleters_by_fact,
        achiever_pre_by_fact=achiever_pre_by_fact,
        object_token_fn=object_token_fn,
        use_object_edges=config.use_object_edges,
    )


def build_candidate_pairs(
    pre_by_name: Mapping[str, Iterable[Fact]],
    ctx: StructuralContext,
    config: AndGammaConfig,
) -> List[frozenset]:
    """Precomputed candidate dependency universe for calibration.

    Primary source: facts that co-occur in some action's preconditions (exactly
    the pairs that can ever form a runtime component). Enriched with statically
    dependent pairs among precondition facts. Bounded by ``max_candidate_pairs``;
    unrelated fact pairs are never mined.
    """
    candidates: "dict[frozenset, None]" = {}

    # 1) Co-occurrence pairs within a single action's preconditions.
    prec_facts: Set[Fact] = set()
    for pres in pre_by_name.values():
        pres_list = list(pres)
        prec_facts.update(pres_list)
        for i in range(len(pres_list)):
            for j in range(i + 1, len(pres_list)):
                if pres_list[i] == pres_list[j]:
                    continue
                candidates[frozenset((pres_list[i], pres_list[j]))] = None
                if len(candidates) >= config.max_candidate_pairs:
                    return list(candidates.keys())

    # 2) Statically dependent pairs among precondition facts (cross-action).
    prec_list = list(prec_facts)
    for i in range(len(prec_list)):
        for j in range(i + 1, len(prec_list)):
            pair = frozenset((prec_list[i], prec_list[j]))
            if pair in candidates:
                continue
            if ctx.pair_edges(prec_list[i], prec_list[j]):
                candidates[pair] = None
                if len(candidates) >= config.max_candidate_pairs:
                    break
        if len(candidates) >= config.max_candidate_pairs:
            break

    return list(candidates.keys())
