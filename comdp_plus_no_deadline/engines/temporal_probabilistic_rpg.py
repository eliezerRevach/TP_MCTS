from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from types import SimpleNamespace
import math
import time
from typing import Dict, Hashable, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from comdp_plus_no_deadline.engines.probabilistic_rpg import (
    compute_precondition_support,
)
from comdp_plus_no_deadline.engines.correlation_heuristic import (
    build_action_specs,
    compute_correlation_preplanning,
    run_correlation_dp,
)


Fact = Hashable


@dataclass(frozen=True)
class TemporalRelaxedActionModel:
    """Action abstraction for duration-aware relaxed propagation."""

    name: str
    preconditions: frozenset[Fact]
    add_probabilities: Mapping[Fact, float]
    effect_delay_steps: int


@dataclass
class TemporalLayerTrace:
    """Debug snapshot for one temporal layer."""

    layer: int
    fact_probabilities: Dict[Fact, float]
    action_support: Dict[str, float] = field(default_factory=dict)
    arrivals: Dict[Fact, float] = field(default_factory=dict)


@dataclass
class TemporalPropagationResult:
    """Output bundle for the duration-aware heuristic."""

    probabilities_by_layer: Dict[int, Dict[Fact, float]]
    depth_used: int
    traces: List[TemporalLayerTrace]
    cache_hit: bool
    fact_cache_hits: int
    action_cache_hits: int
    cached_table: Optional["CachedPTRPGTable"] = None
    # Carries the (temp_dict, new_state_facts) pair from baseline_cached.
    temp_result: Optional[Tuple[Dict[int, Dict[Fact, float]], frozenset]] = None


@dataclass
class CachedPTRPGTable:
    """Mutable cache container for incremental baseline propagation."""

    probabilities_by_layer: Dict[int, Dict[Fact, float]]
    state_facts: frozenset[Fact]
    depth_used: int
    start_layer: int

    def clone(self) -> "CachedPTRPGTable":
        return CachedPTRPGTable(
            probabilities_by_layer={
                layer: dict(probabilities)
                for layer, probabilities in self.probabilities_by_layer.items()
            },
            state_facts=frozenset(self.state_facts),
            depth_used=self.depth_used,
            start_layer=self.start_layer,
        )



def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_resolution_delta_schedule(
    remaining: int,
    *,
    alpha: float = 2.0,
    k_target: int = 8,
    t_ref: Optional[int] = None,
    delta_min: int = 1,
    forced_minimum: bool = False,
) -> List[int]:
    """
    Piece widths Δ_k that partition ``remaining`` (sum = ``remaining``).

    Raw width per layer ``k``:
    - ``forced_minimum=False``: Δ_k_raw = α^floor(k/2) (no remaining/T factor).
    - ``forced_minimum=True``: Δ_k_raw = α^floor(k/2) · remaining / T with
      T = ``t_ref`` if set, else ``remaining`` (ratio 1 until you pass a reference T).

    Final width: max(Δ_min, round(Δ_k_raw)) so steps do not collapse to 0 when
    rounding is harsh; sum correction still matches ``remaining`` exactly.
    """
    remaining = max(0, int(remaining))
    if remaining == 0:
        return []
    alpha = float(alpha)
    if alpha < 1.0:
        raise ValueError(f"resolution alpha must be >= 1, got {alpha}")
    k_target = max(1, int(k_target))
    delta_min = max(1, int(delta_min))
    T = int(t_ref) if t_ref is not None else remaining
    T = max(1, T)

    if forced_minimum:
        k_from_ref = min(k_target, T // delta_min)
        k_cap = max(1, remaining // delta_min)
        K = min(k_from_ref, k_cap)
    else:
        K = min(k_target, remaining // delta_min)
    if K <= 0:
        return []

    deltas: List[int] = []
    for k in range(K):
        exp_w = alpha ** (k // 2)
        if forced_minimum:
            raw = exp_w * (float(remaining) / float(T))
        else:
            raw = float(exp_w)
        deltas.append(max(delta_min, int(round(raw))))

    diff = remaining - sum(deltas)
    if diff > 0:
        deltas[-1] += diff
    elif diff < 0:
        excess = -diff
        i = K - 1
        while excess > 0 and i >= 0:
            take = min(deltas[i] - delta_min, excess)
            deltas[i] -= take
            excess -= take
            i -= 1
        if excess > 0:
            # Should not happen if K was capped by remaining//delta_min; keep safe.
            raise ValueError(
                f"resolution schedule could not absorb excess={excess} "
                f"(remaining={remaining}, K={K}, delta_min={delta_min})"
            )

    assert sum(deltas) == remaining
    return deltas


def _raw_delta_steps_for_resolution_depth(
    depth: int,
    *,
    alpha: float = 2.0,
    k_target: int = 8,
    t_ref: Optional[int] = None,
    delta_min: int = 1,
    forced_minimum: bool = False,
) -> List[int]:
    """Partition ``depth`` into resolution layer widths (see ``build_resolution_delta_schedule``)."""
    return build_resolution_delta_schedule(
        depth,
        alpha=alpha,
        k_target=k_target,
        t_ref=t_ref,
        delta_min=delta_min,
        forced_minimum=forced_minimum,
    )


def _resolution_anchors_ascending(
    depth: int,
    *,
    alpha: float = 2.0,
    k_target: int = 8,
    t_ref: Optional[int] = None,
    delta_min: int = 1,
    forced_minimum: bool = False,
) -> List[int]:
    """Cumulative time anchors [0, …, depth] after largest-to-smallest delta reorganization."""
    depth = max(0, int(depth))
    if depth == 0:
        return [0]
    deltas = _raw_delta_steps_for_resolution_depth(
        depth,
        alpha=alpha,
        k_target=k_target,
        t_ref=t_ref,
        delta_min=delta_min,
        forced_minimum=forced_minimum,
    )
    sorted_desc = sorted(deltas, reverse=True)
    t = depth
    backward: List[int] = [depth]
    for w in sorted_desc:
        t -= w
        backward.append(t)
    return list(reversed(backward))


def _resolution_completion_times(
    anchors_asc: Sequence[int],
    first_completion: int,
    horizon: int,
) -> List[int]:
    """Anchor completion times in [first_completion, horizon], ascending."""
    times = sorted(
        {int(a) for a in anchors_asc if first_completion <= int(a) <= int(horizon)}
    )
    if not times and horizon >= first_completion:
        return [int(horizon)]
    return times


def _extract_state_facts(state) -> Set[Fact]:
    if hasattr(state, "predicates"):
        return set(state.predicates)
    return set(state)


class TemporalProbabilisticRPGHeuristic:
    """
    Duration-aware optimistic relaxed heuristic with fixed temporal depth.

    Compared to the non-temporal PRPG, this version delays action effects by
    `effect_delay_steps` and propagates fact support over fixed temporal layers.
    Actions are ignored at a layer when their precondition support is zero.
    """

    def __init__(
        self,
        actions: Sequence[object],
        facts: Optional[Iterable[Fact]] = None,
        initial_facts: Optional[Iterable[Fact]] = None,
        goal_facts: Optional[Iterable[Fact]] = None,
    ):
        self._actions = list(actions)
        self._facts: Set[Fact] = set(facts or [])
        # Facts that are TRUE in the initial state (used to seed the ET tables).
        # Distinct from self._facts which includes goals and other known facts.
        self._initial_facts: frozenset = frozenset(initial_facts or [])
        self._goal_facts: frozenset[Fact] = frozenset(goal_facts or [])
        # Wall time for compute_correlation_preplanning only (excludes ET tables, graphs, build_action_specs).
        self.correlation_preplanning_time_sec: Optional[float] = None
        self._action_models = self._build_action_models()
        self._fact_dependency_graph = self._build_fact_dependency_graph()
        self._actions_by_effect_fact = self._build_actions_by_effect_fact()
        # Cross-query memoization to reuse the same fixed-depth computation.
        # Keys may include a goal-key suffix for strategies that slice target_facts.
        self._query_cache: Dict[Tuple, TemporalPropagationResult] = {}
        # Dedicated memoization for the atom strategy recurrence.
        self._atom_split_cache: Dict[Tuple[frozenset[Fact], Fact, int, str], float] = {}
        # Structural cache for atom_backtrack_exact (goal_facts, fixed_depth) -> ordered list
        self._schedule_cache: Dict[Tuple[frozenset[Fact], int], List[Tuple[Fact, int]]] = {}
        # Cross-call lazy memos for fast_atom_cache (keyed by root state_facts signature).
        self._fast_atom_fact_memo: Dict[Tuple[frozenset[Fact], Fact, int], float] = {}
        self._fast_atom_action_memo: Dict[Tuple[frozenset[Fact], str, Fact, int], float] = {}
        # Precomputed availability tables for heuristic_expected_time.
        # Built once at construction; each query just reads from these tables.
        self._et_table_depth: int = 200
        self._et_tables: Dict[Fact, List[Tuple["TemporalRelaxedActionModel", float, List[List[float]]]]] = self._build_et_tables()
        # Correlation-aware pre-planning (fact–fact tags + joint miss-corner pairs).
        self._corr_specs = build_action_specs(self._actions)
        self._corr_name_to_spec = {s.name: s for s in self._corr_specs}
        all_f = set(self._facts)
        for spec in self._corr_specs:
            all_f |= set(spec.preconditions)
            for o in spec.joint_adds:
                all_f |= set(o)
        if self._goal_facts:
            t_corr = time.perf_counter()
            ct, jp, ach, _ = compute_correlation_preplanning(
                self._corr_specs,
                set(self._goal_facts),
                all_f,
            )
            self.correlation_preplanning_time_sec = time.perf_counter() - t_corr
            self._correlation_table: Dict[frozenset, str] = ct
            self._joint_pairs: Set[frozenset] = jp
            self._achievers_by_fact_corr: Dict[Fact, Set[str]] = ach
        else:
            self._correlation_table = {}
            self._joint_pairs = set()
            self._achievers_by_fact_corr = {}

    @classmethod
    def from_problem(cls, problem) -> "TemporalProbabilisticRPGHeuristic":
        initial = set(getattr(problem, "initial_values", {}).keys())
        facts = initial | set(getattr(problem, "goals", set()))
        goals = set(getattr(problem, "goals", set()))
        return cls(
            getattr(problem, "actions", []),
            facts=facts,
            initial_facts=initial,
            goal_facts=goals,
        )

    def _query_cache_goal_key(
        self,
        goal_facts: Optional[Iterable[Fact]],
        state_facts: Set[Fact],
    ) -> Tuple[str, frozenset[Fact]]:
        """Disambiguate cache entries when target_facts depend on goal_facts."""
        if goal_facts is not None:
            return ("explicit_goals", frozenset(goal_facts))
        return ("default_targets", frozenset(self._facts.union(state_facts)))

    def heuristic_propagate(
        self,
        state,
        goal_facts: Optional[Iterable[Fact]] = None,
        fixed_depth: int = 25,
        start_time: float = 0.0,
        strategy: str = "baseline",
        cached_table=None,
        debug: bool = False,
        *,
        resolution_alpha: float = 2.0,
        resolution_forced_minimum: bool = False,
        resolution_k_target: int = 8,
        resolution_reference_t: Optional[int] = None,
    ) -> TemporalPropagationResult:
        state_facts = _extract_state_facts(state)
        state_sig = frozenset(state_facts)
        start_layer = max(0, int(math.floor(start_time)))
        chosen_strategy = self._normalize_strategy(strategy)
        query_key: Tuple = (state_sig, int(fixed_depth), start_layer, chosen_strategy)
        if chosen_strategy in (
            "atom_backtrack_exact",
            "atom_backtrack_exact_resolution",
            "atom_backtrack_cached",
            "fast_atom_cache",
        ):
            query_key = query_key + (self._query_cache_goal_key(goal_facts, state_facts),)
        if chosen_strategy == "atom_backtrack_exact_resolution":
            query_key = query_key + (
                float(resolution_alpha),
                bool(resolution_forced_minimum),
                int(resolution_k_target),
                resolution_reference_t,
            )
        if query_key in self._query_cache:
            cached = self._query_cache[query_key]
            return TemporalPropagationResult(
                probabilities_by_layer=cached.probabilities_by_layer,
                depth_used=cached.depth_used,
                traces=cached.traces if debug else [],
                cache_hit=True,
                fact_cache_hits=cached.fact_cache_hits,
                action_cache_hits=cached.action_cache_hits,
            )

        if chosen_strategy == "atom_half_split":
            result = self._heuristic_propagate_atom_half_split(
                state=state,
                fixed_depth=fixed_depth,
                start_time=start_time,
                debug=debug,
            )
            self._query_cache[query_key] = result
            return result
        if chosen_strategy == "atom_backtrack_exact":
            result = self._heuristic_propagate_atom_backtrack_exact(
                state=state,
                goal_facts=goal_facts,
                fixed_depth=fixed_depth,
                start_time=start_time,
                debug=debug,
            )
            self._query_cache[query_key] = result
            return result
        if chosen_strategy == "atom_backtrack_exact_resolution":
            result = self._heuristic_propagate_atom_backtrack_exact_resolution(
                state=state,
                goal_facts=goal_facts,
                fixed_depth=fixed_depth,
                start_time=start_time,
                debug=debug,
                resolution_alpha=resolution_alpha,
                resolution_forced_minimum=resolution_forced_minimum,
                resolution_k_target=resolution_k_target,
                resolution_reference_t=resolution_reference_t,
            )
            self._query_cache[query_key] = result
            return result
        if chosen_strategy == "atom_backtrack_cached":
            result = self._heuristic_propagate_atom_backtrack_cached(
                state=state,
                goal_facts=goal_facts,
                fixed_depth=fixed_depth,
                start_time=start_time,
                debug=debug,
            )
            self._query_cache[query_key] = result
            return result
        if chosen_strategy == "fast_atom_cache":
            result = self._heuristic_propagate_fast_atom_cache(
                state=state,
                goal_facts=goal_facts,
                fixed_depth=fixed_depth,
                start_time=start_time,
                debug=debug,
            )
            self._query_cache[query_key] = result
            return result
        if chosen_strategy == "baseline_cached":
            result = self._heuristic_propagate_baseline_cached(
                state=state,
                fixed_depth=fixed_depth,
                start_time=start_time,
                cached_table=cached_table,
                debug=debug,
            )
            self._query_cache[query_key] = result
            return result

        depth = max(0, int(fixed_depth))
        facts = self._facts.union(state_facts)
        probabilities_by_layer: Dict[int, Dict[Fact, float]] = {
            t: {fact: 0.0 for fact in facts} for t in range(depth + 1)
        }
        for fact in facts:
            probabilities_by_layer[0][fact] = 1.0 if fact in state_facts else 0.0

        pending_successes: Dict[Tuple[int, Fact], List[float]] = {}
        fact_support_cache: Dict[Tuple[Fact, int], float] = {}
        action_support_cache: Dict[Tuple[str, int], float] = {}
        fact_cache_hits = 0
        action_cache_hits = 0
        traces: List[TemporalLayerTrace] = []

        def fact_support(fact: Fact, layer: int) -> float:
            nonlocal fact_cache_hits
            key = (fact, layer)
            if key in fact_support_cache:
                fact_cache_hits += 1
                return fact_support_cache[key]
            value = _clamp_probability(probabilities_by_layer[layer].get(fact, 0.0))
            fact_support_cache[key] = value
            return value

        for layer in range(depth + 1):
            # Persistence from previous layer.
            if layer > 0:
                for fact in facts:
                    probabilities_by_layer[layer][fact] = max(
                        probabilities_by_layer[layer][fact],
                        probabilities_by_layer[layer - 1][fact],
                    )

            arrivals: Dict[Fact, float] = {}
            # Apply effects scheduled to arrive at this layer.
            for fact in facts:
                successes = pending_successes.get((layer, fact), [])
                if not successes:
                    continue
                arrival_hazard = _clamp_probability(1.0 - prod(1.0 - s for s in successes))
                current = probabilities_by_layer[layer][fact]
                updated = _clamp_probability(current + (1.0 - current) * arrival_hazard)
                probabilities_by_layer[layer][fact] = max(current, updated)
                arrivals[fact] = arrival_hazard
                fact_support_cache[(fact, layer)] = probabilities_by_layer[layer][fact]

            # No outgoing actions from the last layer of fixed depth.
            if layer == depth:
                if debug:
                    traces.append(
                        TemporalLayerTrace(
                            layer=layer,
                            fact_probabilities=dict(probabilities_by_layer[layer]),
                            arrivals=arrivals,
                        )
                    )
                break

            action_support: Dict[str, float] = {}
            for action_model in self._action_models:
                support_key = (action_model.name, layer)
                if support_key in action_support_cache:
                    action_support_value = action_support_cache[support_key]
                    action_cache_hits += 1
                else:
                    probs_for_preconditions = {
                        fact: fact_support(fact, layer) for fact in action_model.preconditions
                    }
                    action_support_value = compute_precondition_support(
                        action_model.preconditions,
                        probs_for_preconditions,
                        strict=True,
                    )
                    action_support_cache[support_key] = action_support_value

                action_support[action_model.name] = action_support_value
                if action_support_value <= 0.0:
                    continue

                arrival_layer = layer + action_model.effect_delay_steps
                if arrival_layer > depth:
                    continue
                for fact, add_prob in action_model.add_probabilities.items():
                    success = _clamp_probability(action_support_value * add_prob)
                    pending_successes.setdefault((arrival_layer, fact), []).append(success)

            if debug:
                traces.append(
                    TemporalLayerTrace(
                        layer=layer,
                        fact_probabilities=dict(probabilities_by_layer[layer]),
                        action_support=action_support,
                        arrivals=arrivals,
                    )
                )

        result = TemporalPropagationResult(
            probabilities_by_layer=probabilities_by_layer,
            depth_used=depth,
            traces=traces,
            cache_hit=False,
            fact_cache_hits=fact_cache_hits,
            action_cache_hits=action_cache_hits,
            cached_table=CachedPTRPGTable(
                probabilities_by_layer={
                    layer: dict(values) for layer, values in probabilities_by_layer.items()
                },
                state_facts=frozenset(state_facts),
                depth_used=depth,
                start_layer=start_layer,
            ),
        )
        self._query_cache[query_key] = result
        return result

    def heuristic_score(
        self,
        state,
        goal_facts: Iterable[Fact],
        aggregation: str = "product",
        fixed_depth: int = 25,
        start_time: float = 0.0,
        strategy: str = "baseline",
        cached_table=None,
        return_cache_table: bool = False,
        debug: bool = False,
        *,
        resolution_alpha: float = 2.0,
        resolution_forced_minimum: bool = False,
        resolution_k_target: int = 8,
        resolution_reference_t: Optional[int] = None,
    ):
        result = self.heuristic_propagate(
            state=state,
            goal_facts=goal_facts,
            fixed_depth=fixed_depth,
            start_time=start_time,
            strategy=strategy,
            cached_table=cached_table,
            debug=debug,
            resolution_alpha=resolution_alpha,
            resolution_forced_minimum=resolution_forced_minimum,
            resolution_k_target=resolution_k_target,
            resolution_reference_t=resolution_reference_t,
        )
        final_probabilities = result.probabilities_by_layer[result.depth_used]
        goal_probabilities = [
            _clamp_probability(final_probabilities.get(goal, 0.0)) for goal in goal_facts
        ]

        if aggregation == "product":
            score = 1.0
            for probability in goal_probabilities:
                score *= probability
        elif aggregation == "min":
            score = min(goal_probabilities, default=1.0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        score = _clamp_probability(score)
        if debug:
            return score, result
        if return_cache_table:
            # baseline_cached: return (temp_dict, new_state_facts) pair.
            if result.temp_result is not None:
                return score, result.temp_result
            # All other strategies: return a CachedPTRPGTable clone.
            if result.cached_table is not None:
                ct = result.cached_table
                return score, CachedPTRPGTable(
                    probabilities_by_layer={
                        layer: dict(vals)
                        for layer, vals in ct.probabilities_by_layer.items()
                    },
                    state_facts=frozenset(ct.state_facts),
                    depth_used=ct.depth_used,
                    start_layer=ct.start_layer,
                )
            return score, None
        return score

    def pessimistic_heuristic(
        self,
        state,
        goal_facts: Optional[Iterable[Fact]] = None,
        fixed_depth: int = 25,
        start_time: float = 0.0,
        problem_deadline: Optional[float] = None,
    ) -> float:
        """Lower-bound estimate on P(all goals by deadline) using correlation-aware DP."""
        gf = list(goal_facts) if goal_facts is not None else list(self._goal_facts)
        state_facts = _extract_state_facts(state)
        eff = max(0, int(fixed_depth))
        if problem_deadline is not None:
            rem = max(0, int(math.floor(float(problem_deadline) - start_time)))
            eff = min(eff, rem)
        return run_correlation_dp(
            state_facts=set(state_facts),
            goal_facts=gf,
            action_specs=self._corr_specs,
            achievers_by_fact=self._achievers_by_fact_corr,
            name_to_spec=self._corr_name_to_spec,
            correlation_table=self._correlation_table,
            joint_pairs=self._joint_pairs,
            deadline=eff,
            pessimistic=True,
        )

    def optimistic_heuristic(
        self,
        state,
        goal_facts: Optional[Iterable[Fact]] = None,
        fixed_depth: int = 25,
        start_time: float = 0.0,
        problem_deadline: Optional[float] = None,
    ) -> float:
        """Upper-bound estimate on P(all goals by deadline) using correlation-aware DP."""
        gf = list(goal_facts) if goal_facts is not None else list(self._goal_facts)
        state_facts = _extract_state_facts(state)
        eff = max(0, int(fixed_depth))
        if problem_deadline is not None:
            rem = max(0, int(math.floor(float(problem_deadline) - start_time)))
            eff = min(eff, rem)
        return run_correlation_dp(
            state_facts=set(state_facts),
            goal_facts=gf,
            action_specs=self._corr_specs,
            achievers_by_fact=self._achievers_by_fact_corr,
            name_to_spec=self._corr_name_to_spec,
            correlation_table=self._correlation_table,
            joint_pairs=self._joint_pairs,
            deadline=eff,
            pessimistic=False,
        )

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        value = (strategy or "baseline").strip().lower()
        valid = {
            "baseline",
            "baseline_cached",
            "atom_half_split",
            "atom_backtrack_exact",
            "atom_backtrack_exact_resolution",
            "atom_backtrack_cached",
            "fast_atom_cache",
        }
        if value not in valid:
            raise ValueError(
                f"Unknown temporal heuristic strategy: {strategy!r}. "
                "Supported strategies: baseline, baseline_cached, atom_half_split, "
                "atom_backtrack_exact, atom_backtrack_exact_resolution, "
                "atom_backtrack_cached, fast_atom_cache."
            )
        return value

    def _build_fact_dependency_graph(self) -> Dict[Fact, Set[Fact]]:
        graph: Dict[Fact, Set[Fact]] = {}
        for action_model in self._action_models:
            for precondition in action_model.preconditions:
                graph.setdefault(precondition, set())
                for effect_fact in action_model.add_probabilities.keys():
                    graph[precondition].add(effect_fact)
        for fact in self._facts:
            graph.setdefault(fact, set())
        return graph

    def _build_actions_by_effect_fact(self) -> Dict[Fact, List[TemporalRelaxedActionModel]]:
        actions_by_effect: Dict[Fact, List[TemporalRelaxedActionModel]] = {}
        for action_model in self._action_models:
            for effect_fact in action_model.add_probabilities.keys():
                actions_by_effect.setdefault(effect_fact, []).append(action_model)
        return actions_by_effect

    def _expand_dirty_facts(self, changed_facts: Set[Fact]) -> Set[Fact]:
        if not changed_facts:
            return set()
        dirty: Set[Fact] = set(changed_facts)
        queue = list(changed_facts)
        while queue:
            source = queue.pop()
            for dependent in self._fact_dependency_graph.get(source, set()):
                if dependent in dirty:
                    continue
                dirty.add(dependent)
                queue.append(dependent)
        return dirty

    def _heuristic_propagate_baseline_cached(
        self,
        state,
        fixed_depth: int,
        start_time: float,
        cached_table: Optional[CachedPTRPGTable],
        debug: bool,
    ) -> TemporalPropagationResult:
        state_facts = _extract_state_facts(state)
        depth = max(0, int(fixed_depth))
        start_layer = max(0, int(math.floor(start_time)))

        # Fallback to full baseline when there is no compatible cache yet.
        if (
            cached_table is None
            or cached_table.depth_used != depth
            or cached_table.start_layer != start_layer
        ):
            return self.heuristic_propagate(
                state=state,
                fixed_depth=depth,
                start_time=start_time,
                strategy="baseline",
                cached_table=None,
                debug=debug,
            )

        facts = self._facts.union(state_facts).union(cached_table.state_facts)
        reused = cached_table.clone()
        probabilities_by_layer = reused.probabilities_by_layer
        for layer in range(depth + 1):
            probabilities_by_layer.setdefault(layer, {})
            for fact in facts:
                probabilities_by_layer[layer].setdefault(fact, 0.0)

        changed_facts = set(cached_table.state_facts.symmetric_difference(state_facts))
        if not changed_facts:
            reused.state_facts = frozenset(state_facts)
            return TemporalPropagationResult(
                probabilities_by_layer=probabilities_by_layer,
                depth_used=depth,
                traces=[],
                cache_hit=True,
                fact_cache_hits=0,
                action_cache_hits=0,
                cached_table=reused,
            )

        dirty_facts = self._expand_dirty_facts(changed_facts)
        fact_cache_hits = 0
        action_cache_hits = 0
        traces: List[TemporalLayerTrace] = []

        # Reset dirty columns to recompute them exactly.
        for fact in dirty_facts:
            for layer in range(depth + 1):
                probabilities_by_layer[layer][fact] = 0.0
            probabilities_by_layer[0][fact] = 1.0 if fact in state_facts else 0.0
        for fact in facts:
            if fact not in dirty_facts:
                probabilities_by_layer[0][fact] = 1.0 if fact in state_facts else 0.0

        # Recompute only dirty effects using the current table values.
        pending_successes: Dict[Tuple[int, Fact], List[float]] = {}
        action_support_cache: Dict[Tuple[str, int], float] = {}
        fact_support_cache: Dict[Tuple[Fact, int], float] = {}

        # Models hold a dict field (add_probabilities), so they cannot live in a set().
        relevant_by_name: Dict[str, TemporalRelaxedActionModel] = {}
        for fact in dirty_facts:
            for action_model in self._actions_by_effect_fact.get(fact, []):
                relevant_by_name.setdefault(action_model.name, action_model)
        relevant_actions = list(relevant_by_name.values())

        def fact_support(fact: Fact, layer: int) -> float:
            nonlocal fact_cache_hits
            key = (fact, layer)
            if key in fact_support_cache:
                fact_cache_hits += 1
                return fact_support_cache[key]
            value = _clamp_probability(probabilities_by_layer[layer].get(fact, 0.0))
            fact_support_cache[key] = value
            return value

        for layer in range(depth + 1):
            arrivals: Dict[Fact, float] = {}
            if layer > 0:
                for fact in dirty_facts:
                    probabilities_by_layer[layer][fact] = _clamp_probability(
                        max(
                            probabilities_by_layer[layer][fact],
                            probabilities_by_layer[layer - 1][fact],
                        )
                    )

            for fact in dirty_facts:
                successes = pending_successes.get((layer, fact), [])
                if not successes:
                    continue
                arrival_hazard = _clamp_probability(1.0 - prod(1.0 - s for s in successes))
                current = probabilities_by_layer[layer][fact]
                updated = _clamp_probability(current + (1.0 - current) * arrival_hazard)
                probabilities_by_layer[layer][fact] = max(current, updated)
                arrivals[fact] = arrival_hazard
                fact_support_cache[(fact, layer)] = probabilities_by_layer[layer][fact]

            if layer == depth:
                if debug:
                    traces.append(
                        TemporalLayerTrace(
                            layer=layer,
                            fact_probabilities=dict(probabilities_by_layer[layer]),
                            arrivals=arrivals,
                        )
                    )
                break

            action_support: Dict[str, float] = {}
            for action_model in relevant_actions:
                support_key = (action_model.name, layer)
                if support_key in action_support_cache:
                    action_support_value = action_support_cache[support_key]
                    action_cache_hits += 1
                else:
                    probs_for_preconditions = {
                        fact: fact_support(fact, layer)
                        for fact in action_model.preconditions
                    }
                    action_support_value = compute_precondition_support(
                        action_model.preconditions,
                        probs_for_preconditions,
                        strict=True,
                    )
                    action_support_cache[support_key] = action_support_value

                action_support[action_model.name] = action_support_value
                if action_support_value <= 0.0:
                    continue
                arrival_layer = layer + action_model.effect_delay_steps
                if arrival_layer > depth:
                    continue
                for fact, add_prob in action_model.add_probabilities.items():
                    if fact not in dirty_facts:
                        continue
                    success = _clamp_probability(action_support_value * add_prob)
                    pending_successes.setdefault((arrival_layer, fact), []).append(success)

            if debug:
                traces.append(
                    TemporalLayerTrace(
                        layer=layer,
                        fact_probabilities=dict(probabilities_by_layer[layer]),
                        action_support=action_support,
                        arrivals=arrivals,
                    )
                )

        reused.state_facts = frozenset(state_facts)
        reused.probabilities_by_layer = probabilities_by_layer
        return TemporalPropagationResult(
            probabilities_by_layer=probabilities_by_layer,
            depth_used=depth,
            traces=traces,
            cache_hit=False,
            fact_cache_hits=fact_cache_hits,
            action_cache_hits=action_cache_hits,
            cached_table=reused,
        )

    @staticmethod
    def _is_single_fact_precondition(action_model: TemporalRelaxedActionModel) -> bool:
        if len(action_model.preconditions) != 1:
            return False
        precondition = next(iter(action_model.preconditions))
        return not isinstance(precondition, (list, tuple, set, frozenset))

    def _build_atom_eligibility(self) -> Dict[Fact, List[TemporalRelaxedActionModel]]:
        achievers_by_fact: Dict[Fact, List[TemporalRelaxedActionModel]] = {}
        for action_model in self._action_models:
            for fact in action_model.add_probabilities:
                achievers_by_fact.setdefault(fact, []).append(action_model)

        eligible: Dict[Fact, List[TemporalRelaxedActionModel]] = {}
        for fact, achievers in achievers_by_fact.items():
            if not achievers:
                continue
            if all(
                action_model.effect_delay_steps == 1
                and self._is_single_fact_precondition(action_model)
                for action_model in achievers
            ):
                eligible[fact] = achievers
        return eligible

    def _heuristic_propagate_atom_half_split(
        self,
        state,
        fixed_depth: int,
        start_time: float,
        debug: bool,
    ) -> TemporalPropagationResult:
        state_facts = _extract_state_facts(state)
        depth = max(0, int(fixed_depth))
        start_layer = max(0, int(math.floor(start_time)))
        facts = self._facts.union(state_facts)
        probabilities_by_layer: Dict[int, Dict[Fact, float]] = {
            t: {fact: 0.0 for fact in facts} for t in range(depth + 1)
        }
        for fact in facts:
            probabilities_by_layer[0][fact] = 1.0 if fact in state_facts else 0.0

        state_signature = frozenset(state_facts)
        atom_eligible_achievers = self._build_atom_eligibility()
        fact_cache_hits = 0
        action_cache_hits = 0
        traces: List[TemporalLayerTrace] = []

        if depth == 0:
            return TemporalPropagationResult(
                probabilities_by_layer=probabilities_by_layer,
                depth_used=depth,
                traces=traces,
                cache_hit=False,
                fact_cache_hits=0,
                action_cache_hits=0,
            )

        pending_successes: Dict[Tuple[int, Fact], List[float]] = {}
        fact_support_cache: Dict[Tuple[Fact, int], float] = {}
        action_support_cache: Dict[Tuple[str, int], float] = {}

        def fact_support(fact: Fact, layer: int) -> float:
            nonlocal fact_cache_hits
            key = (fact, layer)
            if key in fact_support_cache:
                fact_cache_hits += 1
                return fact_support_cache[key]
            value = _clamp_probability(probabilities_by_layer[layer].get(fact, 0.0))
            fact_support_cache[key] = value
            return value

        for layer in range(depth + 1):
            # Persistence from previous layer.
            if layer > 0:
                for fact in facts:
                    probabilities_by_layer[layer][fact] = max(
                        probabilities_by_layer[layer][fact],
                        probabilities_by_layer[layer - 1][fact],
                    )

            arrivals: Dict[Fact, float] = {}
            # Apply delayed effects for non-atom and rejected updates.
            for fact in facts:
                successes = pending_successes.get((layer, fact), [])
                if not successes:
                    continue
                arrival_hazard = _clamp_probability(1.0 - prod(1.0 - s for s in successes))
                current = probabilities_by_layer[layer][fact]
                updated = _clamp_probability(current + (1.0 - current) * arrival_hazard)
                probabilities_by_layer[layer][fact] = max(current, updated)
                arrivals[fact] = arrival_hazard
                fact_support_cache[(fact, layer)] = probabilities_by_layer[layer][fact]

            # No outgoing actions from the last layer of fixed depth.
            if layer == depth:
                if debug:
                    traces.append(
                        TemporalLayerTrace(
                            layer=layer,
                            fact_probabilities=dict(probabilities_by_layer[layer]),
                            arrivals=arrivals,
                        )
                    )
                break

            action_support: Dict[str, float] = {}
            atom_successes: Dict[Fact, List[float]] = {}
            for action_model in self._action_models:
                support_key = (action_model.name, layer)
                if support_key in action_support_cache:
                    action_support_value = action_support_cache[support_key]
                    action_cache_hits += 1
                else:
                    probs_for_preconditions = {
                        fact: fact_support(fact, layer) for fact in action_model.preconditions
                    }
                    action_support_value = compute_precondition_support(
                        action_model.preconditions,
                        probs_for_preconditions,
                        strict=True,
                    )
                    action_support_cache[support_key] = action_support_value

                action_support[action_model.name] = action_support_value
                if action_support_value <= 0.0:
                    continue

                arrival_layer = layer + action_model.effect_delay_steps
                if arrival_layer > depth:
                    continue
                for fact, add_prob in action_model.add_probabilities.items():
                    success = _clamp_probability(action_support_value * add_prob)
                    eligible_achievers = atom_eligible_achievers.get(fact)
                    if (
                        eligible_achievers is not None
                        and action_model in eligible_achievers
                        and arrival_layer == layer + 1
                    ):
                        atom_successes.setdefault(fact, []).append(success)
                    else:
                        pending_successes.setdefault((arrival_layer, fact), []).append(success)

            # Atom-eligible updates use local one-step recurrence on (t-1).
            for fact, successes in atom_successes.items():
                cache_key = (state_signature, fact, layer + 1, "atom_half_split")
                if cache_key in self._atom_split_cache:
                    fact_cache_hits += 1
                    updated = self._atom_split_cache[cache_key]
                else:
                    current = probabilities_by_layer[layer].get(fact, 0.0)
                    hazard = _clamp_probability(1.0 - prod(1.0 - s for s in successes))
                    updated = _clamp_probability(current + (1.0 - current) * hazard)
                    self._atom_split_cache[cache_key] = updated
                probabilities_by_layer[layer + 1][fact] = max(
                    probabilities_by_layer[layer + 1].get(fact, 0.0),
                    updated,
                )
                fact_support_cache[(fact, layer + 1)] = probabilities_by_layer[layer + 1][fact]

            if debug:
                traces.append(
                    TemporalLayerTrace(
                        layer=layer,
                        fact_probabilities=dict(probabilities_by_layer[layer]),
                        action_support=action_support,
                        arrivals=arrivals,
                    )
                )

        return TemporalPropagationResult(
            probabilities_by_layer=probabilities_by_layer,
            depth_used=depth,
            traces=traces,
            cache_hit=False,
            fact_cache_hits=fact_cache_hits,
            action_cache_hits=action_cache_hits,
        )

    def _build_evaluation_schedule(
        self,
        goal_facts: Iterable[Fact],
        fixed_depth: int,
    ) -> List[Tuple[Fact, int]]:
        schedule: List[Tuple[Fact, int]] = []
        visited: Set[Tuple[Fact, int]] = set()
        visiting: Set[Tuple[Fact, int]] = set()
        
        def visit(fact: Fact, horizon: int):
            if horizon < 0:
                return
            key = (fact, horizon)
            if key in visited:
                return
            if key in visiting:
                return
                
            visiting.add(key)
            
            if horizon > 0:
                for action_model in self._actions_by_effect_fact.get(fact, []):
                    delay = max(0, int(action_model.effect_delay_steps))
                    first_completion = max(1, delay)
                    for completion_time in range(first_completion, horizon + 1):
                        available_horizon = completion_time - delay
                        for precondition in action_model.preconditions:
                            visit(precondition, available_horizon)
                            
            visiting.remove(key)
            visited.add(key)
            schedule.append(key)

        for goal in goal_facts:
            visit(goal, fixed_depth)
            
        return schedule

    def _heuristic_propagate_atom_backtrack_exact(
        self,
        state,
        goal_facts: Optional[Iterable[Fact]],
        fixed_depth: int,
        start_time: float,
        debug: bool,
    ) -> TemporalPropagationResult:
        del start_time
        state_facts = _extract_state_facts(state)
        depth = max(0, int(fixed_depth))
        target_facts: Set[Fact] = set(goal_facts) if goal_facts is not None else self._facts.union(state_facts)

        fact_cache_hits = 0
        action_cache_hits = 0
        fact_memo: Dict[Tuple[Fact, int], float] = {}
        action_term_memo: Dict[Tuple[str, Fact, int], float] = {}
        recursion_stack: Set[Tuple[Fact, int]] = set()

        def is_atom_action(action_model: TemporalRelaxedActionModel) -> bool:
            return (
                len(action_model.preconditions) == 0
                or action_model.preconditions.issubset(state_facts)
            )

        def fact_probability(fact: Fact, horizon: int) -> float:
            nonlocal fact_cache_hits
            horizon = int(horizon)
            if horizon < 0:
                return 0.0
            key = (fact, horizon)
            if key in fact_memo:
                fact_cache_hits += 1
                return fact_memo[key]
            if key in recursion_stack:
                return 0.0

            if fact in state_facts:
                value = 1.0
            elif horizon == 0:
                value = 0.0
            else:
                recursion_stack.add(key)
                failure = 1.0
                for action_model in self._actions_by_effect_fact.get(fact, []):
                    action_key = (action_model.name, fact, horizon)
                    if action_key in action_term_memo:
                        action_cache_hits += 1
                        achiever_failure = action_term_memo[action_key]
                    else:
                        delay = max(0, int(action_model.effect_delay_steps))
                        add_probability = _clamp_probability(
                            action_model.add_probabilities.get(fact, 0.0)
                        )
                        if add_probability <= 0.0 or horizon < delay:
                            achiever_failure = 1.0
                        else:
                            first_completion = max(1, delay)
                            attempts = horizon - first_completion + 1
                            if attempts <= 0:
                                achiever_failure = 1.0
                            elif is_atom_action(action_model):
                                achiever_failure = _clamp_probability(
                                    (1.0 - add_probability) ** attempts
                                )
                            else:
                                achiever_failure = 1.0
                                for completion_time in range(first_completion, horizon + 1):
                                    available_horizon = completion_time - delay
                                    precondition_support = 1.0
                                    for precondition in action_model.preconditions:
                                        precondition_support *= fact_probability(
                                            precondition,
                                            available_horizon,
                                        )
                                    step_success = _clamp_probability(
                                        add_probability * precondition_support
                                    )
                                    achiever_failure *= 1.0 - step_success
                                achiever_failure = _clamp_probability(achiever_failure)
                        action_term_memo[action_key] = achiever_failure

                    failure *= achiever_failure
                recursion_stack.remove(key)
                value = _clamp_probability(1.0 - failure)

            fact_memo[key] = value
            return value

        probabilities_by_layer: Dict[int, Dict[Fact, float]] = {
            0: {fact: (1.0 if fact in state_facts else 0.0) for fact in target_facts},
            depth: {fact: fact_probability(fact, depth) for fact in target_facts},
        }

        traces: List[TemporalLayerTrace] = []
        if debug:
            traces.append(
                TemporalLayerTrace(
                    layer=0,
                    fact_probabilities=dict(probabilities_by_layer[0]),
                )
            )
            traces.append(
                TemporalLayerTrace(
                    layer=depth,
                    fact_probabilities=dict(probabilities_by_layer[depth]),
                )
            )

        return TemporalPropagationResult(
            probabilities_by_layer=probabilities_by_layer,
            depth_used=depth,
            traces=traces,
            cache_hit=False,
            fact_cache_hits=fact_cache_hits,
            action_cache_hits=action_cache_hits,
        )

    def _heuristic_propagate_atom_backtrack_exact_resolution(
        self,
        state,
        goal_facts: Optional[Iterable[Fact]],
        fixed_depth: int,
        start_time: float,
        debug: bool,
        *,
        resolution_alpha: float = 2.0,
        resolution_forced_minimum: bool = False,
        resolution_k_target: int = 8,
        resolution_reference_t: Optional[int] = None,
    ) -> TemporalPropagationResult:
        del start_time
        state_facts = _extract_state_facts(state)
        depth = max(0, int(fixed_depth))
        target_facts: Set[Fact] = set(goal_facts) if goal_facts is not None else self._facts.union(state_facts)
        anchors_asc = _resolution_anchors_ascending(
            depth,
            alpha=resolution_alpha,
            k_target=resolution_k_target,
            t_ref=resolution_reference_t,
            delta_min=1,
            forced_minimum=resolution_forced_minimum,
        )

        fact_cache_hits = 0
        action_cache_hits = 0
        fact_memo: Dict[Tuple[Fact, int], float] = {}
        action_term_memo: Dict[Tuple[str, Fact, int], float] = {}
        recursion_stack: Set[Tuple[Fact, int]] = set()

        def is_atom_action(action_model: TemporalRelaxedActionModel) -> bool:
            return (
                len(action_model.preconditions) == 0
                or action_model.preconditions.issubset(state_facts)
            )

        def fact_probability(fact: Fact, horizon: int) -> float:
            nonlocal fact_cache_hits
            horizon = int(horizon)
            if horizon < 0:
                return 0.0
            key = (fact, horizon)
            if key in fact_memo:
                fact_cache_hits += 1
                return fact_memo[key]
            if key in recursion_stack:
                return 0.0

            if fact in state_facts:
                value = 1.0
            elif horizon == 0:
                value = 0.0
            else:
                recursion_stack.add(key)
                failure = 1.0
                for action_model in self._actions_by_effect_fact.get(fact, []):
                    action_key = (action_model.name, fact, horizon)
                    if action_key in action_term_memo:
                        action_cache_hits += 1
                        achiever_failure = action_term_memo[action_key]
                    else:
                        delay = max(0, int(action_model.effect_delay_steps))
                        add_probability = _clamp_probability(
                            action_model.add_probabilities.get(fact, 0.0)
                        )
                        if add_probability <= 0.0 or horizon < delay:
                            achiever_failure = 1.0
                        else:
                            first_completion = max(1, delay)
                            attempts = horizon - first_completion + 1
                            if attempts <= 0:
                                achiever_failure = 1.0
                            else:
                                completion_times = _resolution_completion_times(
                                    anchors_asc, first_completion, horizon
                                )
                                achiever_failure = 1.0
                                for completion_time in reversed(completion_times):
                                    available_horizon = completion_time - delay
                                    precondition_support = 1.0
                                    if not is_atom_action(action_model):
                                        for precondition in action_model.preconditions:
                                            precondition_support *= fact_probability(
                                                precondition,
                                                available_horizon,
                                            )
                                    step_success = _clamp_probability(
                                        add_probability * precondition_support
                                    )
                                    achiever_failure *= 1.0 - step_success
                                achiever_failure = _clamp_probability(achiever_failure)
                        action_term_memo[action_key] = achiever_failure

                    failure *= achiever_failure
                recursion_stack.remove(key)
                value = _clamp_probability(1.0 - failure)

            fact_memo[key] = value
            return value

        probabilities_by_layer: Dict[int, Dict[Fact, float]] = {
            0: {fact: (1.0 if fact in state_facts else 0.0) for fact in target_facts},
            depth: {fact: fact_probability(fact, depth) for fact in target_facts},
        }

        traces: List[TemporalLayerTrace] = []
        if debug:
            traces.append(
                TemporalLayerTrace(
                    layer=0,
                    fact_probabilities=dict(probabilities_by_layer[0]),
                )
            )
            traces.append(
                TemporalLayerTrace(
                    layer=depth,
                    fact_probabilities=dict(probabilities_by_layer[depth]),
                )
            )

        return TemporalPropagationResult(
            probabilities_by_layer=probabilities_by_layer,
            depth_used=depth,
            traces=traces,
            cache_hit=False,
            fact_cache_hits=fact_cache_hits,
            action_cache_hits=action_cache_hits,
        )

    def _heuristic_propagate_atom_backtrack_cached(
        self,
        state,
        goal_facts: Optional[Iterable[Fact]],
        fixed_depth: int,
        start_time: float,
        debug: bool,
    ) -> TemporalPropagationResult:
        del start_time
        state_facts = _extract_state_facts(state)
        depth = max(0, int(fixed_depth))
        target_facts: Set[Fact] = set(goal_facts) if goal_facts is not None else self._facts.union(state_facts)

        schedule_key = (frozenset(target_facts), depth)
        schedule = self._schedule_cache.get(schedule_key)
        if schedule is None:
            schedule = self._build_evaluation_schedule(target_facts, depth)
            self._schedule_cache[schedule_key] = schedule

        fact_cache_hits = 0
        action_cache_hits = 0
        fact_memo: Dict[Tuple[Fact, int], float] = {}
        action_term_memo: Dict[Tuple[str, Fact, int], float] = {}

        # Precompute atom actions for fast Case 1 vs Case 2 lookup
        # An action is atom if all its preconditions are true in the root state.
        atom_actions = {
            action_model.name: (
                len(action_model.preconditions) == 0
                or action_model.preconditions.issubset(state_facts)
            )
            for action_model in self._action_models
        }

        for fact, horizon in schedule:
            if fact in state_facts:
                fact_memo[(fact, horizon)] = 1.0
                continue
            if horizon == 0:
                fact_memo[(fact, horizon)] = 0.0
                continue

            failure = 1.0
            for action_model in self._actions_by_effect_fact.get(fact, []):
                action_key = (action_model.name, fact, horizon)
                if action_key in action_term_memo:
                    action_cache_hits += 1
                    achiever_failure = action_term_memo[action_key]
                else:
                    delay = max(0, int(action_model.effect_delay_steps))
                    add_probability = _clamp_probability(
                        action_model.add_probabilities.get(fact, 0.0)
                    )
                    if add_probability <= 0.0 or horizon < delay:
                        achiever_failure = 1.0
                    else:
                        first_completion = max(1, delay)
                        attempts = horizon - first_completion + 1
                        if attempts <= 0:
                            achiever_failure = 1.0
                        elif atom_actions[action_model.name]:
                            achiever_failure = _clamp_probability(
                                (1.0 - add_probability) ** attempts
                            )
                        else:
                            achiever_failure = 1.0
                            for completion_time in range(first_completion, horizon + 1):
                                available_horizon = completion_time - delay
                                precondition_support = 1.0
                                for precondition in action_model.preconditions:
                                    # Default to 0.0 handles cyclic dependencies (not in memo)
                                    precondition_support *= fact_memo.get((precondition, available_horizon), 0.0)
                                step_success = _clamp_probability(
                                    add_probability * precondition_support
                                )
                                achiever_failure *= 1.0 - step_success
                            achiever_failure = _clamp_probability(achiever_failure)
                    action_term_memo[action_key] = achiever_failure

                failure *= achiever_failure
            fact_memo[(fact, horizon)] = _clamp_probability(1.0 - failure)

        probabilities_by_layer: Dict[int, Dict[Fact, float]] = {
            0: {fact: (1.0 if fact in state_facts else 0.0) for fact in target_facts},
            depth: {fact: fact_memo.get((fact, depth), 0.0) for fact in target_facts},
        }

        traces: List[TemporalLayerTrace] = []
        if debug:
            traces.append(
                TemporalLayerTrace(
                    layer=0,
                    fact_probabilities=dict(probabilities_by_layer[0]),
                )
            )
            traces.append(
                TemporalLayerTrace(
                    layer=depth,
                    fact_probabilities=dict(probabilities_by_layer[depth]),
                )
            )

        return TemporalPropagationResult(
            probabilities_by_layer=probabilities_by_layer,
            depth_used=depth,
            traces=traces,
            cache_hit=False,
            fact_cache_hits=fact_cache_hits,
            action_cache_hits=action_cache_hits,
        )

    def _heuristic_propagate_fast_atom_cache(
        self,
        state,
        goal_facts: Optional[Iterable[Fact]],
        fixed_depth: int,
        start_time: float,
        debug: bool,
    ) -> TemporalPropagationResult:
        """
        Same schedule-ordered semantics as ``atom_backtrack_cached``, but:
        - always uses the per-timestep completion loop for achievers (no closed-form
          ``(1-p)^attempts`` shortcut), and
        - persists fact/action term values on the heuristic instance for reuse across calls.
        """
        del start_time
        state_facts = _extract_state_facts(state)
        state_sig = frozenset(state_facts)
        depth = max(0, int(fixed_depth))
        target_facts: Set[Fact] = set(goal_facts) if goal_facts is not None else self._facts.union(state_facts)

        schedule_key = (frozenset(target_facts), depth)
        schedule = self._schedule_cache.get(schedule_key)
        if schedule is None:
            schedule = self._build_evaluation_schedule(target_facts, depth)
            self._schedule_cache[schedule_key] = schedule

        fact_cache_hits = 0
        action_cache_hits = 0
        fact_memo: Dict[Tuple[Fact, int], float] = {}
        action_term_memo: Dict[Tuple[str, Fact, int], float] = {}

        for fact, horizon in schedule:
            fact_sig = (state_sig, fact, horizon)
            if fact in state_facts:
                v = 1.0
                fact_memo[(fact, horizon)] = v
                self._fast_atom_fact_memo[fact_sig] = v
                continue
            if horizon == 0:
                v = 0.0
                fact_memo[(fact, horizon)] = v
                self._fast_atom_fact_memo[fact_sig] = v
                continue

            if fact_sig in self._fast_atom_fact_memo:
                fact_cache_hits += 1
                v = self._fast_atom_fact_memo[fact_sig]
                fact_memo[(fact, horizon)] = v
                continue

            failure = 1.0
            for action_model in self._actions_by_effect_fact.get(fact, []):
                action_sig = (state_sig, action_model.name, fact, horizon)
                action_key = (action_model.name, fact, horizon)
                if action_sig in self._fast_atom_action_memo:
                    action_cache_hits += 1
                    achiever_failure = self._fast_atom_action_memo[action_sig]
                elif action_key in action_term_memo:
                    action_cache_hits += 1
                    achiever_failure = action_term_memo[action_key]
                else:
                    delay = max(0, int(action_model.effect_delay_steps))
                    add_probability = _clamp_probability(
                        action_model.add_probabilities.get(fact, 0.0)
                    )
                    if add_probability <= 0.0 or horizon < delay:
                        achiever_failure = 1.0
                    else:
                        first_completion = max(1, delay)
                        achiever_failure = 1.0
                        for completion_time in range(first_completion, horizon + 1):
                            available_horizon = completion_time - delay
                            precondition_support = 1.0
                            for precondition in action_model.preconditions:
                                precondition_support *= fact_memo.get((precondition, available_horizon), 0.0)
                            step_success = _clamp_probability(
                                add_probability * precondition_support
                            )
                            achiever_failure *= 1.0 - step_success
                        achiever_failure = _clamp_probability(achiever_failure)
                    action_term_memo[action_key] = achiever_failure
                    self._fast_atom_action_memo[action_sig] = achiever_failure

                failure *= achiever_failure
            v = _clamp_probability(1.0 - failure)
            fact_memo[(fact, horizon)] = v
            self._fast_atom_fact_memo[fact_sig] = v

        probabilities_by_layer: Dict[int, Dict[Fact, float]] = {
            0: {fact: (1.0 if fact in state_facts else 0.0) for fact in target_facts},
            depth: {fact: fact_memo.get((fact, depth), 0.0) for fact in target_facts},
        }

        traces: List[TemporalLayerTrace] = []
        if debug:
            traces.append(
                TemporalLayerTrace(
                    layer=0,
                    fact_probabilities=dict(probabilities_by_layer[0]),
                )
            )
            traces.append(
                TemporalLayerTrace(
                    layer=depth,
                    fact_probabilities=dict(probabilities_by_layer[depth]),
                )
            )

        return TemporalPropagationResult(
            probabilities_by_layer=probabilities_by_layer,
            depth_used=depth,
            traces=traces,
            cache_hit=False,
            fact_cache_hits=fact_cache_hits,
            action_cache_hits=action_cache_hits,
        )

    def heuristic_expected_time(
        self,
        state,
        goal_facts: Iterable[Fact],
    ) -> float:
        """
        Estimate E[T_goal] — expected steps to achieve all goal facts — without
        any deadline or fixed depth.

        Uses the tail-sum identity:
            E[T] = sum_{t=0}^{inf} P(T > t)

        Precondition availability tables are precomputed once at construction
        in the all-false world.  At query time, preconditions that are true in
        ``state`` are substituted with availability = 1.0 — no recursion, no
        stepper allocation, just list index reads.

        For conjunctive goals {g1, ..., gk} the joint failure at step t is:
            1 - prod_i (1 - failure_i(t))
        summed to give E[T_goal] = E[max(T_g1, ..., T_gk)].
        """
        state_facts = frozenset(_extract_state_facts(state))
        goals = list(goal_facts)

        if not goals:
            return 0.0
        if all(g in state_facts for g in goals):
            return 0.0

        if len(goals) == 1:
            return self._et_from_tables(goals[0], state_facts)

        # Conjunctive: E[T_goal] = 1 + sum_{s=1}^{S} joint_failure(s) + tail
        S = self._et_table_depth
        E_T = 1.0
        failure_per_goal = [1.0] * len(goals)
        for s in range(1, S + 1):
            joint_success = 1.0
            for i, g in enumerate(goals):
                failure_per_goal[i] = self._failure_at_step(g, s, state_facts, failure_per_goal[i])
                joint_success *= 1.0 - failure_per_goal[i]
            joint_failure = _clamp_probability(1.0 - joint_success)
            E_T += joint_failure
            if joint_failure < 1e-12:
                break
        else:
            # Geometric tail: rate = geometric mean of per-goal tail rates
            tail_survival = 1.0
            for i, g in enumerate(goals):
                tail_survival *= self._et_tail_survival(g, state_facts)
            if tail_survival >= 1.0:
                return float("inf")
            joint_failure_at_S = _clamp_probability(1.0 - sum(
                (1.0 - failure_per_goal[i]) for i in range(len(goals))
            ) + (len(goals) - 1))
            last_jf = _clamp_probability(1.0 - prod(
                1.0 - failure_per_goal[i] for i in range(len(goals))
            ))
            if last_jf > 0.0:
                E_T += last_jf * tail_survival / (1.0 - tail_survival)

        return E_T

    # ------------------------------------------------------------------
    # Internal helpers for heuristic_expected_time (table-based)
    # ------------------------------------------------------------------

    def _build_et_tables(self) -> Dict[Fact, List[Tuple["TemporalRelaxedActionModel", float, List[List[float]]]]]:
        """
        Precompute, for each fact, a list of (action_model, p_a, prec_avail_tables).

        prec_avail_tables[i][s] = P(precondition_i achieved by step s+1) seeded
        from the initial state (self._facts).

        Uses an iterative forward-propagation approach (no recursion, no cycle
        guards needed):
          - Facts in self._facts start available (delete-free: they stay true forever).
          - Dynamic facts start unavailable and accumulate availability as actions fire.
          - Each step s computes availability using the previous step's values.

        This is essentially the planning graph computed forward from the initial state.
        """
        S = self._et_table_depth
        all_achievable = set(self._actions_by_effect_fact.keys())

        # fact_avail[fact][s] = P(fact achieved by step s+1), 0-indexed.
        # Seed from self._initial_facts: facts that are TRUE in the initial
        # state start fully available (delete-free relaxation: they stay true).
        # Goals and other non-initial facts start at 0 and grow as steps unfold.
        fact_avail: Dict[Fact, List[float]] = {}
        for fact in all_achievable:
            if fact in self._initial_facts:
                fact_avail[fact] = [1.0] * S  # always available (initially true)
            else:
                fact_avail[fact] = [0.0] * S

        # Pre-cache static tables to avoid re-creating them in the loop.
        static_avail_true: List[float] = [1.0] * S
        static_avail_false: List[float] = [0.0] * S

        def _get_prec_avail_fast(prec: Fact) -> List[float]:
            if prec in fact_avail:
                return fact_avail[prec]
            # Static fact (no achievers): initially true → always available;
            # otherwise never achievable.
            return static_avail_true if prec in self._initial_facts else static_avail_false

        # Precompute achiever lists with numeric parameters for speed.
        # achiever_specs[fact] = [(p_a, delay, [prec_avail_ref, ...]), ...]
        achiever_specs: Dict[Fact, List[Tuple[float, int, List[List[float]]]]] = {}
        for fact in all_achievable:
            specs = []
            for action_model in self._actions_by_effect_fact.get(fact, []):
                p_a = _clamp_probability(action_model.add_probabilities.get(fact, 0.0))
                if p_a <= 0.0:
                    continue
                delay = max(0, int(action_model.effect_delay_steps) - 1)
                prec_refs = [_get_prec_avail_fast(prec) for prec in action_model.preconditions]
                specs.append((p_a, delay, prec_refs))
            achiever_specs[fact] = specs

        # Iterative forward pass: compute step s for all facts before moving to s+1.
        # s is 0-indexed; avail[s] = P(fact by step s+1).
        for s in range(S):
            for fact in all_achievable:
                if fact_avail[fact][s] >= 1.0:
                    # Already fully available — no need to update.
                    if s + 1 < S:
                        fact_avail[fact][s + 1] = 1.0
                    continue
                prev_failure = 1.0 - (fact_avail[fact][s - 1] if s > 0 else 0.0)
                step_survival = 1.0
                for p_a, delay, prec_refs in achiever_specs[fact]:
                    prec_support = 1.0
                    idx = s - delay - 1  # avail at step s-delay (0-indexed)
                    for pa_ref in prec_refs:
                        prec_support *= pa_ref[idx] if idx >= 0 else 0.0
                    step_success = _clamp_probability(p_a * prec_support)
                    step_survival *= 1.0 - step_success
                failure = _clamp_probability(prev_failure * step_survival)
                fact_avail[fact][s] = _clamp_probability(1.0 - failure)

        # Build et_tables from the precomputed avail tables.
        et_tables: Dict[Fact, List[Tuple["TemporalRelaxedActionModel", float, List[List[float]]]]] = {}
        for fact in all_achievable:
            entries = []
            for action_model in self._actions_by_effect_fact.get(fact, []):
                p_a = _clamp_probability(action_model.add_probabilities.get(fact, 0.0))
                if p_a <= 0.0:
                    continue
                prec_avail_tables = [
                    _get_prec_avail_fast(prec) for prec in action_model.preconditions
                ]
                entries.append((action_model, p_a, prec_avail_tables))
            if entries:
                et_tables[fact] = entries
        return et_tables

    def _et_tail_survival(self, fact: Fact, state_facts: frozenset) -> float:
        """
        Return the per-step survival factor in the geometric tail for this fact
        given the current state.  This is the step_survival value once all
        preconditions have stabilized (i.e. at large s where prec_avail ~ 1 for
        true precs, or the precomputed asymptote for false ones).
        """
        achievers_data = self._et_tables.get(fact)
        if not achievers_data:
            return 1.0
        S = self._et_table_depth
        step_survival = 1.0
        for action_model, p_a, prec_avail_tables in achievers_data:
            prec_support = 1.0
            for prec, pa_table in zip(action_model.preconditions, prec_avail_tables):
                if prec in state_facts:
                    avail = 1.0
                else:
                    avail = pa_table[S - 1]  # asymptotic value
                prec_support *= avail
            step_success = _clamp_probability(p_a * prec_support)
            step_survival *= 1.0 - step_success
        return step_survival

    def _failure_at_step(
        self,
        fact: Fact,
        s: int,
        state_facts: frozenset,
        prev_failure: float,
    ) -> float:
        """
        Return failure_fact(s) given failure_fact(s-1) = prev_failure.

        step_survival at s is computed from precomputed tables:
        - preconditions in state_facts contribute availability 1.0
        - preconditions not in state_facts contribute their precomputed
          all-false availability at step s (0-indexed: s-1 with lag: s-2)
        """
        achievers_data = self._et_tables.get(fact)
        if not achievers_data:
            return prev_failure  # no achievers, failure stays at prev value

        step_survival = 1.0
        for action_model, p_a, prec_avail_tables in achievers_data:
            prec_support = 1.0
            for prec, pa_table in zip(action_model.preconditions, prec_avail_tables):
                if prec in state_facts:
                    avail = 1.0
                else:
                    # Precondition availability at step s-delay-1 (lagged).
                    delay = max(0, int(action_model.effect_delay_steps) - 1)
                    idx = s - delay - 2  # s - delay - 1, then 0-indexed
                    avail = pa_table[idx] if idx >= 0 else 0.0
                prec_support *= avail
            step_success = _clamp_probability(p_a * prec_support)
            step_survival *= 1.0 - step_success
        return _clamp_probability(prev_failure * step_survival)

    def _et_from_tables(self, fact: Fact, state_facts: frozenset) -> float:
        """
        Compute E[T_fact] using the precomputed availability tables.

        No recursion, no stepper allocation.  Each step is O(achievers * precs)
        list index reads.  A geometric tail closes the sum beyond table depth.
        """
        if fact in state_facts:
            return 0.0
        achievers_data = self._et_tables.get(fact)
        if not achievers_data:
            return float("inf")

        # Fast path: all achievers fully unlocked (all precs in state_facts).
        # Combined success per step is constant -> E[T] = 1 / p_combined.
        all_unlocked = all(
            action_model.preconditions.issubset(state_facts)
            and action_model.effect_delay_steps == 1
            for action_model, _p_a, _prec_tables in achievers_data
        )
        if all_unlocked:
            combined_failure = 1.0
            for action_model, p_a, _ in achievers_data:
                combined_failure *= 1.0 - p_a
            p_combined = _clamp_probability(1.0 - combined_failure)
            if p_combined <= 0.0:
                return float("inf")
            return 1.0 / p_combined

        # General path: iterate table, accumulate tail-sum.
        S = self._et_table_depth
        failure = 1.0
        E_T = 1.0  # t=0 term: P(T > 0) = 1 since fact not in state
        for s in range(1, S + 1):
            step_survival = 1.0
            for action_model, p_a, prec_avail_tables in achievers_data:
                prec_support = 1.0
                delay = max(0, int(action_model.effect_delay_steps) - 1)
                for prec, pa_table in zip(action_model.preconditions, prec_avail_tables):
                    if prec in state_facts:
                        avail = 1.0
                    else:
                        idx = s - delay - 2  # lag: use availability at step s-delay-1
                        avail = pa_table[idx] if idx >= 0 else 0.0
                    prec_support *= avail
                step_success = _clamp_probability(p_a * prec_support)
                step_survival *= 1.0 - step_success
            failure = _clamp_probability(failure * step_survival)
            E_T += failure
            if failure < 1e-12:
                return E_T

        # Geometric tail: failure(s) ~ failure(S) * tail_survival^(s-S) for s > S.
        # sum_{s=S+1}^{inf} failure(S) * tail_survival^(s-S) = failure(S) * tail_survival / (1 - tail_survival)
        tail_survival = self._et_tail_survival(fact, state_facts)
        if tail_survival >= 1.0:
            return float("inf")
        E_T += failure * tail_survival / (1.0 - tail_survival)
        return E_T

    def _build_action_models(self) -> List[TemporalRelaxedActionModel]:
        models: List[TemporalRelaxedActionModel] = []
        for action in self._actions:
            if hasattr(action, "actions") and not hasattr(action, "add_effects"):
                continue
            preconditions = frozenset(getattr(action, "pos_preconditions", set()))
            add_probabilities = self._extract_add_probabilities(action)
            if not preconditions and not add_probabilities:
                continue
            effect_delay_steps = self._extract_effect_delay_steps(action)
            self._facts.update(preconditions)
            self._facts.update(add_probabilities.keys())
            models.append(
                TemporalRelaxedActionModel(
                    name=getattr(action, "name", repr(action)),
                    preconditions=preconditions,
                    add_probabilities=add_probabilities,
                    effect_delay_steps=effect_delay_steps,
                )
            )
        return models

    @staticmethod
    def _extract_effect_delay_steps(action) -> int:
        """
        Approximate when an action's add effects should become available.

        For split durative actions:
        - `start_*` actions expose start/inExecution effects in the next layer.
        - `end_*` actions expose the original durative end effects only after the
          remaining duration beyond that start layer.
        This prevents end effects from collapsing to one layer in the converted model.
        """
        if hasattr(action, "start_action") and getattr(action, "start_action", None) is not None:
            try:
                total_duration = int(action.start_action.duration_int())
                return max(1, total_duration - 1)
            except Exception:
                return 1
        if hasattr(action, "end_action") and getattr(action, "end_action", None) is not None:
            return 1
        if hasattr(action, "duration_int"):
            try:
                return max(1, int(action.duration_int()))
            except Exception:
                pass
        if hasattr(action, "duration"):
            try:
                return max(1, int(action.duration.lower.int_constant_value()))
            except Exception:
                pass
        return 1

    @staticmethod
    def _extract_add_probabilities(action) -> Dict[Fact, float]:
        add_probabilities: Dict[Fact, float] = {}
        for fact in getattr(action, "add_effects", set()):
            add_probabilities[fact] = 1.0

        for probabilistic_effect in getattr(action, "probabilistic_effects", []):
            # Structural extraction only: sum probabilities of outcomes that add fact=True.
            per_effect_probability: MutableMapping[Fact, float] = {}
            try:
                outcomes = probabilistic_effect.probability_function(
                    SimpleNamespace(predicates=set()),
                    None,
                )
            except Exception:
                outcomes = {}
            for outcome_probability, assignments in outcomes.items():
                p = _clamp_probability(outcome_probability)
                for fact, value in assignments.items():
                    is_positive = bool(value.bool_constant_value()) if hasattr(
                        value,
                        "bool_constant_value",
                    ) else bool(value)
                    if is_positive:
                        per_effect_probability[fact] = _clamp_probability(
                            per_effect_probability.get(fact, 0.0) + p
                        )

            for fact, probability in per_effect_probability.items():
                existing = add_probabilities.get(fact, 0.0)
                add_probabilities[fact] = _clamp_probability(
                    1.0 - (1.0 - existing) * (1.0 - probability)
                )

        return add_probabilities
