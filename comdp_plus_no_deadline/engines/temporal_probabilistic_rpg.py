from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from types import SimpleNamespace
import math
import random
import time
from typing import Callable, Dict, Hashable, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from comdp_plus_no_deadline.engines.probabilistic_rpg import (
    compute_precondition_support,
)
from comdp_plus_no_deadline.engines.correlation_heuristic import (
    build_action_specs,
    compute_correlation_preplanning,
    run_correlation_dp,
)
from comdp_plus_no_deadline.engines.and_gamma import (
    AndGammaCalibrator,
    AndGammaConfig,
    RolloutSimulator,
    build_candidate_pairs,
    build_components,
    build_structural_context,
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
    alpha: Optional[float] = None,
    t_ref: Optional[int] = None,
    delta_min: int = 1,
    forced_minimum: bool = False,
) -> List[int]:
    """
    Piece widths Δ_k that partition ``remaining`` (sum = ``remaining``).

    Layer count ``K`` is derived locally: append deltas until the cumulative sum
    reaches ``remaining`` (last delta truncates). No external ``k_target`` cap.

    - ``forced_minimum=False``: raw ``α^floor(k/2)`` (``alpha=None`` -> ``2.0``).
    - ``forced_minimum=True``: raw ``α^floor(k/2) · remaining / T`` with
      ``T = t_ref`` if set, else ``remaining`` (set ``t_ref`` to deadline for
      ``remaining/deadline`` scaling).

    Each step uses ``max(Δ_min, round(raw))`` then truncation so widths stay at
    least ``Δ_min`` and never overshoot ``remaining``.
    """
    remaining = max(0, int(remaining))
    if remaining == 0:
        return []
    if alpha is None:
        alpha = 2.0
    alpha = float(alpha)
    if alpha < 1.0:
        raise ValueError(f"resolution alpha must be >= 1, got {alpha}")
    delta_min = max(1, int(delta_min))
    T = int(t_ref) if t_ref is not None else remaining
    T = max(1, T)

    deltas: List[int] = []
    k = 0
    s = 0
    while s < remaining:
        exp_w = alpha ** (k // 2)
        if forced_minimum:
            raw = exp_w * (float(remaining) / float(T))
        else:
            raw = float(exp_w)
        w = max(delta_min, int(round(raw)))
        if s + w > remaining:
            w = remaining - s
        deltas.append(w)
        s += w
        k += 1

    assert sum(deltas) == remaining
    return deltas


def _raw_delta_steps_for_resolution_depth(
    depth: int,
    *,
    alpha: Optional[float] = None,
    t_ref: Optional[int] = None,
    delta_min: int = 1,
    forced_minimum: bool = False,
) -> List[int]:
    """Partition ``depth`` into resolution layer widths (see ``build_resolution_delta_schedule``)."""
    return build_resolution_delta_schedule(
        depth,
        alpha=alpha,
        t_ref=t_ref,
        delta_min=delta_min,
        forced_minimum=forced_minimum,
    )


def _resolution_anchors_ascending(
    depth: int,
    *,
    alpha: Optional[float] = None,
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

    # Fallback add-probability for a probabilistic effect whose outcome
    # distribution cannot be evaluated at model-build time (its probability
    # function is an opaque, state-dependent callable that may raise when
    # probed). Each structurally-declared affected fluent is then registered as
    # an achiever with this probability so the fact stays reachable instead of
    # being silently dropped to P=0. Optimistic (1.0) by default, matching the
    # delete-relaxation spirit; lower it for more conservative stochastic
    # achievement (override per-instance before constructing if desired).
    _FALLBACK_PROBABILISTIC_ADD_PROB: float = 1.0

    # Coefficient alpha for the "meanvar" goal aggregation: score = mean - c*std
    # over per-goal areas, with c = alpha * sqrt(k-1). alpha=1 is the Samuelson
    # "snap to min when lopsided" extreme, but as a SEARCH GRADIENT that is too
    # aggressive: completing one goal increases the spread, so alpha>=~0.75 makes
    # the penalty reverse the progress gradient (finishing a goal lowers the
    # score). Empirically the score stays monotone in progress for alpha<=~0.5;
    # 0.35 keeps a safe margin while still steering toward the laggard goal.
    _MEANVAR_ALPHA: float = 0.35

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
        # Precomputed structural data for atom_backtrack_exact_unbiased (lazy).
        self._unbiased_structural_built: bool = False
        self._unbiased_action_del_effects: Dict[str, frozenset] = {}
        self._unbiased_deleters_by_fact: Dict[Fact, List[str]] = {}
        self._unbiased_name_to_model: Dict[str, TemporalRelaxedActionModel] = {}
        # Per (state_sig, target_sig, depth) cache of {lambda_total, lambda_breakdown, B_table}.
        self._unbiased_correction_cache: Dict[Tuple[frozenset[Fact], frozenset[Fact], int], Dict] = {}
        # Lazy delete-probability table for the survival-aware baseline strategy.
        # name -> {fact: Pr_del(a, f)}; plus the set of facts that any action deletes.
        self._survival_table_built: bool = False
        self._survival_del_prob_by_name: Dict[str, Dict[Fact, float]] = {}
        self._survival_facts_with_deleters: Set[Fact] = set()
        # AND-layer gamma correction (baseline_survival_and_gamma). Config is
        # overridable per-instance before the first query; everything else is
        # built lazily once on first use.
        self._and_gamma_config: AndGammaConfig = AndGammaConfig()
        self._and_gamma_built: bool = False
        self._and_gamma_calibrator: Optional[AndGammaCalibrator] = None
        self._and_gamma_components_by_name: Dict[str, List] = {}
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
        resolution_alpha: Optional[float] = None,
        resolution_forced_minimum: bool = False,
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
            "atom_backtrack_exact_resolution_and_gamma",
            "atom_backtrack_exact_unbiased",
            "atom_backtrack_cached",
            "fast_atom_cache",
        ):
            query_key = query_key + (self._query_cache_goal_key(goal_facts, state_facts),)
        if chosen_strategy in (
            "atom_backtrack_exact_resolution",
            "atom_backtrack_exact_resolution_and_gamma",
            "atom_backtrack_exact_unbiased",
            "baseline_survival_resolution",
        ):
            query_key = query_key + (
                float(2.0 if resolution_alpha is None else resolution_alpha),
                bool(resolution_forced_minimum),
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
                resolution_reference_t=resolution_reference_t,
            )
            self._query_cache[query_key] = result
            return result
        if chosen_strategy == "atom_backtrack_exact_resolution_and_gamma":
            result = self._heuristic_propagate_atom_backtrack_exact_resolution_and_gamma(
                state=state,
                goal_facts=goal_facts,
                fixed_depth=fixed_depth,
                start_time=start_time,
                debug=debug,
                resolution_alpha=resolution_alpha,
                resolution_forced_minimum=resolution_forced_minimum,
                resolution_reference_t=resolution_reference_t,
            )
            self._query_cache[query_key] = result
            return result
        if chosen_strategy == "atom_backtrack_exact_unbiased":
            result = self._heuristic_propagate_atom_backtrack_exact_unbiased(
                state=state,
                goal_facts=goal_facts,
                fixed_depth=fixed_depth,
                start_time=start_time,
                debug=debug,
                resolution_alpha=resolution_alpha,
                resolution_forced_minimum=resolution_forced_minimum,
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
        if chosen_strategy in ("baseline_survival", "baseline_survival_meanvar"):
            # Same survival propagation for both; they differ only in the
            # goal aggregation chosen at scoring time (product vs meanvar),
            # so they can be compared head-to-head.
            result = self._heuristic_propagate_baseline_survival(
                state=state,
                fixed_depth=fixed_depth,
                start_time=start_time,
                debug=debug,
            )
            self._query_cache[query_key] = result
            return result
        if chosen_strategy == "baseline_survival_resolution":
            # Survival/delete recursion over log-spaced (exponential-width)
            # resolution layers: P_{t-k} with k = exponential gap.
            result = self._heuristic_propagate_baseline_survival_resolution(
                state=state,
                fixed_depth=fixed_depth,
                start_time=start_time,
                debug=debug,
                resolution_alpha=resolution_alpha,
                resolution_forced_minimum=resolution_forced_minimum,
                resolution_reference_t=resolution_reference_t,
            )
            self._query_cache[query_key] = result
            return result
        if chosen_strategy == "baseline_survival_and_gamma":
            # Identical survival propagation, but the AND-layer precondition
            # support R(a) is replaced by the component-wise gamma-corrected
            # estimate R_gamma(a). Reduces to baseline_survival when no
            # dependency is detected.
            result = self._heuristic_propagate_baseline_survival_and_gamma(
                state=state,
                fixed_depth=fixed_depth,
                start_time=start_time,
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
        resolution_alpha: Optional[float] = None,
        resolution_forced_minimum: bool = False,
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
            resolution_reference_t=resolution_reference_t,
        )
        goals_list = list(goal_facts)
        final_probabilities = result.probabilities_by_layer[result.depth_used]
        goal_probabilities = [
            _clamp_probability(final_probabilities.get(goal, 0.0)) for goal in goals_list
        ]

        if aggregation == "product":
            score = 1.0
            for probability in goal_probabilities:
                score *= probability
        elif aggregation == "min":
            score = min(goal_probabilities, default=1.0)
        elif aggregation == "area":
            # Time-aware graded score: mean over ALL temporal layers of the
            # goal-product at that layer (an integral / area-under-curve of
            # P_t(all goals)). Unlike "product" (which reads only the final
            # layer and saturates to a binary 0->1 step), this:
            #   * rewards achieving the goals EARLIER (more layers contribute),
            #   * is LOWERED by survival/deletion decay (a goal deleted and only
            #     re-achieved later dips the product in the middle layers),
            #   * never collapses to a flat step, so it gives the search a
            #     gradient across tree depths.
            # Requires the full per-layer profile, so it is only meaningful for
            # forward strategies (baseline / baseline_survival / baseline_cached
            # / atom_half_split); backward backtrack strategies only fill layers
            # {0, depth} and should not use this mode.
            layers = sorted(result.probabilities_by_layer.keys())
            if not layers:
                score = 0.0
            else:
                total = 0.0
                for layer in layers:
                    probs = result.probabilities_by_layer[layer]
                    product_at_layer = 1.0
                    for goal in goals_list:
                        product_at_layer *= _clamp_probability(probs.get(goal, 0.0))
                    total += product_at_layer
                score = total / len(layers)
        elif aggregation == "meanvar":
            # Variance-aware AND aggregation (shaped search direction, NOT
            # P(all goals)).
            #
            #   per-goal value a_i = AREA = mean over layers of P_t(g_i)
            #   score = mean_i(a_i) - c * std_i(a_i),   c = alpha * sqrt(k-1)
            #
            # By Samuelson's inequality min >= mean - sqrt(k-1)*std, so with
            # alpha=1 the score equals `mean` when the per-goal areas are
            # balanced (std=0) and collapses to `min` when one goal is the lone
            # laggard, interpolating in between. The std term is what exposes
            # imbalance: a lagging goal pulls the value down toward the
            # bottleneck, giving greedy a gradient that product/min do not.
            #
            # Per-goal AREA (not the final-layer marginal) is used so the values
            # are graded rather than saturated to 0/1; this requires the full
            # per-layer profile (forward strategies only, e.g. baseline_survival).
            layers = sorted(result.probabilities_by_layer.keys())
            if not layers or not goals_list:
                score = 0.0 if goals_list else 1.0
            else:
                per_goal_area: List[float] = []
                for goal in goals_list:
                    s = 0.0
                    for layer in layers:
                        s += _clamp_probability(
                            result.probabilities_by_layer[layer].get(goal, 0.0)
                        )
                    per_goal_area.append(s / len(layers))
                k = len(per_goal_area)
                mean_area = sum(per_goal_area) / k
                if k > 1:
                    var = sum((a - mean_area) ** 2 for a in per_goal_area) / k
                    std = math.sqrt(var)
                    c = self._MEANVAR_ALPHA * math.sqrt(k - 1)
                    score = mean_area - c * std
                else:
                    score = mean_area
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
            "baseline_survival",
            "baseline_survival_meanvar",
            "baseline_survival_and_gamma",
            "baseline_survival_resolution",
            "atom_half_split",
            "atom_backtrack_exact",
            "atom_backtrack_exact_resolution",
            "atom_backtrack_exact_resolution_and_gamma",
            "atom_backtrack_exact_unbiased",
            "atom_backtrack_cached",
            "fast_atom_cache",
        }
        if value not in valid:
            raise ValueError(
                f"Unknown temporal heuristic strategy: {strategy!r}. "
                "Supported strategies: baseline, baseline_cached, baseline_survival, "
                "baseline_survival_meanvar, baseline_survival_and_gamma, "
                "baseline_survival_resolution, "
                "atom_half_split, "
                "atom_backtrack_exact, atom_backtrack_exact_resolution, "
                "atom_backtrack_exact_resolution_and_gamma, "
                "atom_backtrack_exact_unbiased, "
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

    # -------------------------------------------------------------------------
    # baseline_survival: delete/survival-aware forward DP (non-monotone P_t)
    # -------------------------------------------------------------------------
    #
    # The plain baseline recursion is monotone — once a fact is supported its
    # probability can only rise:
    #
    #     P_t(f) = P_{t-1}(f) + (1 - P_{t-1}(f)) * H_t(f)
    #
    # baseline_survival adds a per-step survival factor S_t(f) so that a fact
    # that can be deleted decays back toward 0 when its deleters are reachable:
    #
    #     P_t(f) = P_{t-1}(f)*S_t(f) + (1 - P_{t-1}(f)*S_t(f)) * H_t(f)
    #
    # with the selective/normalized (Option 2) survival model:
    #
    #     h_del(f, t) = sum_{a in Del(f)} R(a, .) * Pr_del(a, f)
    #                   ----------------------------------------
    #                   sum_{a in A}      R(a, .)
    #     S_t(f) = (1 - h_del(f, t)) ** k_t           (k_t = 1 in v1)
    #
    # where R(a, .) is the same achiever reachability (precondition support)
    # already computed for the add side, Pr_del(a, f) is the summed probability
    # of a's outcomes that set f False, and the numerator/denominator are
    # accumulated into per-arrival-layer buckets (sum-and-normalize, NOT
    # 1 - prod(...) — that would be the pessimistic model).
    #
    # v1 timing: deletes are treated as AT-START effects (delete event lands one
    # layer after the deleter fires, R indexed at the firing/start layer). This
    # matches the note "if the delete is a START effect, index R at the action's
    # start time, not finish", keeps the numerator and denominator consistent
    # (both indexed at the firing layer), and is what makes resource facts like
    # free(m) in machine_shop actually decay below 1.
    #
    # Backward compatibility: a fact with no deleters always has S_t(f) = 1, so
    # the recursion collapses to the plain monotone baseline for delete-free
    # facts / delete-free domains.
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_del_probabilities(action) -> Dict[Fact, float]:
        """Pr_del(a, f) per deleted fact: deterministic delete => 1, plus the
        summed probability of probabilistic outcomes that assign f False.

        Mirrors `_extract_add_probabilities` but collects the negative side of
        each effect table."""
        del_probabilities: Dict[Fact, float] = {}
        for fact in getattr(action, "del_effects", set()):
            del_probabilities[fact] = 1.0

        for probabilistic_effect in getattr(action, "probabilistic_effects", []):
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
                    if not is_positive:
                        per_effect_probability[fact] = _clamp_probability(
                            per_effect_probability.get(fact, 0.0) + p
                        )

            for fact, probability in per_effect_probability.items():
                existing = del_probabilities.get(fact, 0.0)
                del_probabilities[fact] = _clamp_probability(
                    1.0 - (1.0 - existing) * (1.0 - probability)
                )

        return del_probabilities

    def _ensure_survival_delete_table(self) -> None:
        """Build name -> {fact: Pr_del} once per heuristic object.

        Keyed by action name to align with `_build_action_models` (which keys
        reachability/support by the same name). Uses the same action filter so
        the deleter set matches the actions whose R we have."""
        if self._survival_table_built:
            return

        del_prob_by_name: Dict[str, Dict[Fact, float]] = {}
        facts_with_deleters: Set[Fact] = set()
        for action in self._actions:
            if hasattr(action, "actions") and not hasattr(action, "add_effects"):
                continue
            name = getattr(action, "name", repr(action))
            dels = {
                f: p
                for f, p in self._extract_del_probabilities(action).items()
                if p > 0.0
            }
            if dels:
                del_prob_by_name[name] = dels
                facts_with_deleters.update(dels.keys())

        self._survival_del_prob_by_name = del_prob_by_name
        self._survival_facts_with_deleters = facts_with_deleters
        self._survival_table_built = True

    def _heuristic_propagate_baseline_survival(
        self,
        state,
        fixed_depth: int,
        start_time: float,
        debug: bool,
        r_estimator: Optional[Callable[["TemporalRelaxedActionModel", Dict[Fact, float]], float]] = None,
    ) -> TemporalPropagationResult:
        # `r_estimator`, when provided, replaces the flat AND-layer precondition
        # support R(a) = compute_precondition_support(...) with a custom estimate
        # (e.g. the component-wise gamma correction). It is used for BOTH the add
        # side and the survival/delete side so the two stay consistent. Default
        # None preserves the exact baseline_survival behaviour.
        self._ensure_survival_delete_table()
        state_facts = _extract_state_facts(state)
        depth = max(0, int(fixed_depth))
        del start_time
        facts = self._facts.union(state_facts)

        # v1 constants.
        DELETE_DELAY = 1  # at-start delete semantics (see method docstring).
        K_T = 1  # expected # actions finishing at t.

        del_prob_by_name = self._survival_del_prob_by_name
        facts_with_deleters = self._survival_facts_with_deleters

        probabilities_by_layer: Dict[int, Dict[Fact, float]] = {
            t: {fact: 0.0 for fact in facts} for t in range(depth + 1)
        }
        for fact in facts:
            probabilities_by_layer[0][fact] = 1.0 if fact in state_facts else 0.0

        # Add arrivals (achiever successes) and delete arrivals, keyed by the
        # layer the effect lands on. Delete buckets implement the selective
        # normalization: numerator over deleters, denominator over all actions.
        pending_successes: Dict[Tuple[int, Fact], List[float]] = {}
        delete_numerator: Dict[Tuple[int, Fact], float] = {}
        delete_denominator: Dict[int, float] = {}

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
            arrivals: Dict[Fact, float] = {}

            # Persistence with decay: P_{t-1}(f) * S_t(f), then arrival hazard.
            if layer > 0:
                den = delete_denominator.get(layer, 0.0)
                for fact in facts:
                    previous = probabilities_by_layer[layer - 1][fact]
                    if (
                        fact in facts_with_deleters
                        and den > 0.0
                        and previous > 0.0
                    ):
                        num = delete_numerator.get((layer, fact), 0.0)
                        h_del = _clamp_probability(num / den)
                        survival = _clamp_probability((1.0 - h_del) ** K_T)
                    else:
                        survival = 1.0
                    decayed = _clamp_probability(previous * survival)
                    successes = pending_successes.get((layer, fact), [])
                    arrival_hazard = (
                        _clamp_probability(1.0 - prod(1.0 - s for s in successes))
                        if successes
                        else 0.0
                    )
                    updated = _clamp_probability(
                        decayed + (1.0 - decayed) * arrival_hazard
                    )
                    probabilities_by_layer[layer][fact] = updated
                    fact_support_cache[(fact, layer)] = updated
                    if arrival_hazard > 0.0:
                        arrivals[fact] = arrival_hazard

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
                        fact: fact_support(fact, layer)
                        for fact in action_model.preconditions
                    }
                    if r_estimator is None:
                        action_support_value = compute_precondition_support(
                            action_model.preconditions,
                            probs_for_preconditions,
                            strict=True,
                        )
                    else:
                        action_support_value = r_estimator(
                            action_model, probs_for_preconditions
                        )
                    action_support_cache[support_key] = action_support_value

                action_support[action_model.name] = action_support_value
                if action_support_value <= 0.0:
                    continue

                # Add side: schedule achiever successes at the add arrival layer.
                arrival_layer = layer + action_model.effect_delay_steps
                if arrival_layer <= depth:
                    for fact, add_prob in action_model.add_probabilities.items():
                        success = _clamp_probability(action_support_value * add_prob)
                        pending_successes.setdefault((arrival_layer, fact), []).append(
                            success
                        )

                # Delete side: every reachable action contributes R to the
                # denominator at the delete-event layer; deleters also add
                # R * Pr_del to the per-fact numerator.
                delete_layer = layer + DELETE_DELAY
                if delete_layer <= depth:
                    delete_denominator[delete_layer] = (
                        delete_denominator.get(delete_layer, 0.0)
                        + action_support_value
                    )
                    dels = del_prob_by_name.get(action_model.name)
                    if dels:
                        for fact, del_prob in dels.items():
                            delete_numerator[(delete_layer, fact)] = (
                                delete_numerator.get((delete_layer, fact), 0.0)
                                + action_support_value * del_prob
                            )

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

    # -------------------------------------------------------------------------
    # baseline_survival_and_gamma: survival DP with AND-layer gamma correction
    # -------------------------------------------------------------------------
    #
    # Identical to baseline_survival except the flat AND-layer precondition
    # support
    #
    #     R(a) = Π_{f in pre(a)} P_{t-1}(f)
    #
    # is replaced by the component-wise corrected estimate
    #
    #     R_gamma(a) = Π_{C in Components(pre(a))} γ(key(C)) · Π_{f in C} P_{t-1}(f)
    #                = R(a) · Π_C γ(key(C))
    #
    # so the correction is a single multiplicative factor on the existing product
    # (singleton / independent components contribute γ = 1, leaving R unchanged).
    # The noisy-OR achiever formula and the survival/delete update are untouched.
    # See comdp_plus_no_deadline/engines/and_gamma.py for the component detection,
    # gamma table, and the optional lazy rollout calibration.
    # -------------------------------------------------------------------------

    def _ensure_and_gamma_built(self) -> None:
        """Build the structural context, per-action components and calibrator once."""
        if self._and_gamma_built:
            return
        self._ensure_unbiased_structural_extracted()  # name_to_model + del effects

        config = self._and_gamma_config
        pre_by_name = {m.name: m.preconditions for m in self._action_models}
        add_by_name = {
            m.name: frozenset(m.add_probabilities.keys()) for m in self._action_models
        }
        del_by_name = {
            name: dels for name, dels in self._unbiased_action_del_effects.items()
        }

        ctx = build_structural_context(
            pre_by_name=pre_by_name,
            add_by_name=add_by_name,
            del_by_name=del_by_name,
            config=config,
        )

        # Per-action precondition components (static; computed once).
        components_by_name: Dict[str, List] = {}
        components_found = 0
        type_counts: Dict[str, int] = {}
        for model in self._action_models:
            comps = build_components(list(model.preconditions), ctx)
            components_by_name[model.name] = comps
            for comp in comps:
                components_found += 1
                type_counts[comp.comp_type] = type_counts.get(comp.comp_type, 0) + 1
        self._and_gamma_components_by_name = components_by_name

        candidate_pairs = build_candidate_pairs(pre_by_name, ctx, config)

        simulator = None
        if config.enable_rollout_calibration:
            rng = random.Random(config.seed)
            simulator = RolloutSimulator(
                self._actions,
                rng=rng,
                horizon=config.rollout_horizon,
                fallback_add_prob=config.fallback_probabilistic_add_prob,
            )

        calibrator = AndGammaCalibrator(
            config=config,
            ctx=ctx,
            candidate_pairs=candidate_pairs,
            simulator=simulator,
        )
        calibrator.components_found = components_found
        for ctype, count in type_counts.items():
            calibrator.type_counts[ctype] = count
        self._and_gamma_calibrator = calibrator
        self._and_gamma_built = True

    def _and_gamma_factor(self, action_name: str) -> float:
        """Product of component gammas for an action's preconditions (≥ 0)."""
        calibrator = self._and_gamma_calibrator
        if calibrator is None:
            return 1.0
        calibrator.and_calls += 1
        factor = 1.0
        for comp in self._and_gamma_components_by_name.get(action_name, []):
            factor *= calibrator.gamma_for_component(comp)
        return factor

    def _heuristic_propagate_baseline_survival_and_gamma(
        self,
        state,
        fixed_depth: int,
        start_time: float,
        debug: bool,
    ) -> TemporalPropagationResult:
        self._ensure_and_gamma_built()
        calibrator = self._and_gamma_calibrator
        state_facts = _extract_state_facts(state)

        # Lazy calibration pre-pass: refine gamma for the keys actually present
        # in this query before the DP reads them (no-op unless rollout
        # calibration is enabled and some key is low-confidence with budget left).
        if calibrator is not None and self._and_gamma_config.enable_rollout_calibration:
            needed_keys = set()
            for model in self._action_models:
                for comp in self._and_gamma_components_by_name.get(model.name, []):
                    if comp.comp_type != "singleton":
                        needed_keys.add(comp.key)
            if needed_keys:
                calibrator.calibrate(start_facts=state_facts, needed_keys=needed_keys)

        def r_estimator(
            action_model: TemporalRelaxedActionModel,
            probs_for_preconditions: Dict[Fact, float],
        ) -> float:
            base = compute_precondition_support(
                action_model.preconditions,
                probs_for_preconditions,
                strict=True,
            )
            factor = self._and_gamma_factor(action_model.name)
            return _clamp_probability(base * factor)

        return self._heuristic_propagate_baseline_survival(
            state=state,
            fixed_depth=fixed_depth,
            start_time=start_time,
            debug=debug,
            r_estimator=r_estimator,
        )

    # -------------------------------------------------------------------------
    # baseline_survival_resolution: survival DP over log-spaced (exp-width) layers
    # -------------------------------------------------------------------------
    #
    # Combines the survival/delete recursion of baseline_survival with the
    # logarithmic-time layer schedule of atom_backtrack_exact_resolution. Instead
    # of stepping one integer layer at a time, it advances over the resolution
    # anchors (gaps grow geometrically: fine near the current time, coarse far in
    # the future). Each super-step of width ``k = cur_anchor - prev_anchor`` uses
    # the same survival formula but references the PREVIOUS ANCHOR ``P_{t-k}`` and
    # raises the per-step factors to the ``k``-th power (k steps of retries /
    # decay):
    #
    #   survival_k(f) = (1 - h_del(f)) ** k          (k delete-events over the gap)
    #   H_k(f)        = 1 - Π_a (1 - s_a(f)) ** att   (att = k - delay(a) + 1 retries)
    #   P_cur(f)      = P_prev(f)*survival_k + (1 - P_prev(f)*survival_k) * H_k(f)
    #
    # with s_a(f) = R(a)·add_prob and R(a), h_del(f) evaluated at the previous
    # anchor (the coarse-far-future relaxation). When a gap is k=1 this collapses
    # to one baseline_survival step, so near-term layers stay exact and only the
    # far-future layers are coarsened. Used standalone and as the v3 suffix
    # evaluator of rollout_aligned_resolution_survival.
    # -------------------------------------------------------------------------

    def _heuristic_propagate_baseline_survival_resolution(
        self,
        state,
        fixed_depth: int,
        start_time: float,
        debug: bool,
        *,
        resolution_alpha: Optional[float] = None,
        resolution_forced_minimum: bool = False,
        resolution_reference_t: Optional[int] = None,
    ) -> TemporalPropagationResult:
        self._ensure_survival_delete_table()
        state_facts = _extract_state_facts(state)
        depth = max(0, int(fixed_depth))
        del start_time
        facts = self._facts.union(state_facts)

        del_prob_by_name = self._survival_del_prob_by_name
        facts_with_deleters = self._survival_facts_with_deleters

        # Log-spaced ascending anchors [0, ..., depth] (fine near 0, coarse far).
        anchors = _resolution_anchors_ascending(
            depth,
            alpha=resolution_alpha,
            t_ref=resolution_reference_t,
            delta_min=1,
            forced_minimum=resolution_forced_minimum,
        )

        # Probability vector, advanced anchor-by-anchor (only {0, depth} are
        # materialized into the result, matching the resolution strategies).
        current_probs: Dict[Fact, float] = {
            fact: (1.0 if fact in state_facts else 0.0) for fact in facts
        }
        probabilities_by_layer: Dict[int, Dict[Fact, float]] = {
            0: dict(current_probs)
        }
        traces: List[TemporalLayerTrace] = []
        if debug:
            traces.append(
                TemporalLayerTrace(layer=0, fact_probabilities=dict(current_probs))
            )

        prev_anchor = 0
        for cur_anchor in anchors[1:]:
            k = int(cur_anchor - prev_anchor)
            if k <= 0:
                continue

            # Action reachability R(a) at the previous anchor (one product per
            # action over its preconditions' current probabilities).
            r_by_name: Dict[str, float] = {}
            denominator = 0.0
            for action_model in self._action_models:
                probs_for_preconditions = {
                    f: _clamp_probability(current_probs.get(f, 0.0))
                    for f in action_model.preconditions
                }
                r = compute_precondition_support(
                    action_model.preconditions,
                    probs_for_preconditions,
                    strict=True,
                )
                r_by_name[action_model.name] = r
                denominator += r  # every reachable action feeds the delete denom

            # Per-fact delete numerator at this anchor.
            delete_num: Dict[Fact, float] = {}
            if denominator > 0.0:
                for name, dels in del_prob_by_name.items():
                    r = r_by_name.get(name, 0.0)
                    if r <= 0.0:
                        continue
                    for f, del_prob in dels.items():
                        delete_num[f] = delete_num.get(f, 0.0) + r * del_prob

            next_probs: Dict[Fact, float] = {}
            arrivals: Dict[Fact, float] = {}
            for fact in facts:
                previous = current_probs.get(fact, 0.0)

                # Arrival hazard over the k-wide window: each achiever can
                # complete (k - delay + 1) times within the gap.
                failure = 1.0
                for achiever in self._actions_by_effect_fact.get(fact, []):
                    delay = max(0, int(achiever.effect_delay_steps))
                    attempts = k - delay + 1
                    if attempts <= 0:
                        continue
                    r = r_by_name.get(achiever.name, 0.0)
                    if r <= 0.0:
                        continue
                    add_prob = _clamp_probability(
                        achiever.add_probabilities.get(fact, 0.0)
                    )
                    step_success = _clamp_probability(r * add_prob)
                    if step_success <= 0.0:
                        continue
                    failure *= (1.0 - step_success) ** attempts
                arrival_hazard = _clamp_probability(1.0 - failure)

                # Survival decay over the k steps.
                if (
                    fact in facts_with_deleters
                    and denominator > 0.0
                    and previous > 0.0
                ):
                    h_del = _clamp_probability(delete_num.get(fact, 0.0) / denominator)
                    survival = _clamp_probability((1.0 - h_del) ** k)
                else:
                    survival = 1.0

                decayed = _clamp_probability(previous * survival)
                updated = _clamp_probability(
                    decayed + (1.0 - decayed) * arrival_hazard
                )
                next_probs[fact] = updated
                if arrival_hazard > 0.0:
                    arrivals[fact] = arrival_hazard

            current_probs = next_probs
            prev_anchor = cur_anchor
            if debug:
                traces.append(
                    TemporalLayerTrace(
                        layer=cur_anchor,
                        fact_probabilities=dict(current_probs),
                        arrivals=arrivals,
                    )
                )

        probabilities_by_layer[depth] = dict(current_probs)
        return TemporalPropagationResult(
            probabilities_by_layer=probabilities_by_layer,
            depth_used=depth,
            traces=traces,
            cache_hit=False,
            fact_cache_hits=0,
            action_cache_hits=0,
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
        resolution_alpha: Optional[float] = None,
        resolution_forced_minimum: bool = False,
        resolution_reference_t: Optional[int] = None,
        gamma_factor_fn: Optional[Callable[[str], float]] = None,
    ) -> TemporalPropagationResult:
        # `gamma_factor_fn`, when provided, returns a per-action multiplicative
        # AND-layer gamma factor applied to the precondition product R(a) (only
        # for real multi-precondition actions, not atoms). Default None keeps the
        # exact atom_backtrack_exact_resolution behaviour.
        del start_time
        state_facts = _extract_state_facts(state)
        depth = max(0, int(fixed_depth))
        target_facts: Set[Fact] = set(goal_facts) if goal_facts is not None else self._facts.union(state_facts)
        anchors_asc = _resolution_anchors_ascending(
            depth,
            alpha=resolution_alpha,
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
                                # AND-layer gamma correction on the precondition
                                # product. Constant per action (structural), so
                                # compute once; only meaningful for real
                                # multi-precondition actions, not atoms.
                                gamma_factor = 1.0
                                if (
                                    gamma_factor_fn is not None
                                    and not is_atom_action(action_model)
                                ):
                                    gamma_factor = gamma_factor_fn(action_model.name)
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
                                        precondition_support *= gamma_factor
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

    def _heuristic_propagate_atom_backtrack_exact_resolution_and_gamma(
        self,
        state,
        goal_facts: Optional[Iterable[Fact]],
        fixed_depth: int,
        start_time: float,
        debug: bool,
        *,
        resolution_alpha: Optional[float] = None,
        resolution_forced_minimum: bool = False,
        resolution_reference_t: Optional[int] = None,
    ) -> TemporalPropagationResult:
        """Resolution backtrack (log-spaced / exponential-width layers) PLUS the
        component-wise AND-layer gamma correction on the precondition product.

        This is ``atom_backtrack_exact_resolution`` with ``R(a)`` replaced by
        ``R_gamma(a) = R(a) * Π_C gamma(key(C))`` — exactly the same AND-layer
        correction used by ``baseline_survival_and_gamma``, but on the
        logarithmic-layer backtrack instead of the linear survival DP. Reduces to
        plain resolution when no precondition dependency is detected (all gamma
        factors = 1). The noisy-OR achiever structure and the resolution anchor
        schedule are untouched.
        """
        self._ensure_and_gamma_built()
        calibrator = self._and_gamma_calibrator
        state_facts = _extract_state_facts(state)

        # Lazy calibration pre-pass: refine gamma for the keys present in this
        # query before the backtrack reads them. No-op unless rollout calibration
        # is enabled and some key is low-confidence with budget left.
        if calibrator is not None and self._and_gamma_config.enable_rollout_calibration:
            needed_keys = set()
            for model in self._action_models:
                for comp in self._and_gamma_components_by_name.get(model.name, []):
                    if comp.comp_type != "singleton":
                        needed_keys.add(comp.key)
            if needed_keys:
                calibrator.calibrate(start_facts=state_facts, needed_keys=needed_keys)

        return self._heuristic_propagate_atom_backtrack_exact_resolution(
            state=state,
            goal_facts=goal_facts,
            fixed_depth=fixed_depth,
            start_time=start_time,
            debug=debug,
            resolution_alpha=resolution_alpha,
            resolution_forced_minimum=resolution_forced_minimum,
            resolution_reference_t=resolution_reference_t,
            gamma_factor_fn=self._and_gamma_factor,
        )

    # -------------------------------------------------------------------------
    # atom_backtrack_exact_unbiased: structural layer-bias correction
    # -------------------------------------------------------------------------
    #
    # Subtracts a per-layer scalar bias B(t) from each goal fact's probability
    # before goal aggregation. B(t) is derived analytically from problem-graph
    # statistics (no rollouts):
    #
    #   lambda = lambda_OR + lambda_AND + lambda_DEL
    #   B(t)   = B(t-1) * (1 - <H_t>) + (1 - <P_{t-1}>) * lambda * <H_t>
    #
    # where <H_t>, <P_t> come from a one-time baseline DP pass and the per-source
    # lambdas come from action mutex / shared-achiever / delete fractions.
    #
    # The pre-planning runs once per (state_sig, target_sig, depth) triple and
    # is cached on the heuristic instance, matching the user-facing semantics
    # of "compute lambda + B(t) once, look up B(t) at every leaf evaluation."
    # -------------------------------------------------------------------------

    def _ensure_unbiased_structural_extracted(self) -> None:
        """Build name->model and fact->deleters indices once per heuristic object."""
        if self._unbiased_structural_built:
            return

        # Map name -> TemporalRelaxedActionModel.
        self._unbiased_name_to_model = {m.name: m for m in self._action_models}

        # Extract delete effects from the original action objects (they are not
        # carried in TemporalRelaxedActionModel because the heuristic itself is
        # delete-relaxed). Match the same filter used in _build_action_models.
        action_del_effects: Dict[str, frozenset] = {}
        for action in self._actions:
            if hasattr(action, "actions") and not hasattr(action, "add_effects"):
                continue
            name = getattr(action, "name", repr(action))
            del_set = set()
            raw_dels = getattr(action, "del_effects", None)
            if raw_dels:
                del_set.update(raw_dels)
            action_del_effects[name] = frozenset(del_set)
        self._unbiased_action_del_effects = action_del_effects

        # Invert: fact -> [action names that delete it].
        deleters: Dict[Fact, List[str]] = {}
        for name, dels in action_del_effects.items():
            for f in dels:
                deleters.setdefault(f, []).append(name)
        self._unbiased_deleters_by_fact = deleters

        self._unbiased_structural_built = True

    def _unbiased_actions_are_mutex(self, name_a: str, name_b: str) -> bool:
        """
        Action-mutex extended beyond pure Graphplan delete-interference.

        Two actions are flagged mutex if any of:
          1. Direct delete-interference: one's delete clobbers the other's
             precondition or add-effect. (Graphplan classic.)
          2. Shared "consumable" precondition: both require a fact that is
             deleted by some action in the domain. The fact is non-monotonic,
             so two competing achievers of a goal that both need it cannot
             both reliably succeed in the same plan window — captures
             resource-mutex without explicit resource modeling.
          3. Shared add-effect: both achieve the same fact. The OR-bias is
             real even without delete-interference: at most one of them can
             "first-achieve" the fact at a given time, so independence
             over-counts.
        """
        if name_a == name_b:
            return False
        model_a = self._unbiased_name_to_model.get(name_a)
        model_b = self._unbiased_name_to_model.get(name_b)
        if model_a is None or model_b is None:
            return False
        del_a = self._unbiased_action_del_effects.get(name_a, frozenset())
        del_b = self._unbiased_action_del_effects.get(name_b, frozenset())
        pre_a = model_a.preconditions
        pre_b = model_b.preconditions
        add_a = frozenset(model_a.add_probabilities.keys())
        add_b = frozenset(model_b.add_probabilities.keys())
        # (1) Direct delete-interference.
        if del_a & (pre_b | add_b):
            return True
        if del_b & (pre_a | add_a):
            return True
        # (2) Shared "consumable" precondition: a fact that can be deleted
        # by some action in the domain, required by both. This catches the
        # resource-style mutex that pure delete-interference misses.
        shared_pre = pre_a & pre_b
        if shared_pre:
            consumables = shared_pre & set(self._unbiased_deleters_by_fact.keys())
            if consumables:
                return True
        # (3) Shared add-effect on the same fact: only one can first-achieve.
        if add_a & add_b:
            return True
        return False

    def _compute_unbiased_correction_table(
        self,
        state,
        target_facts: Set[Fact],
        depth: int,
    ) -> Dict:
        """
        Pre-planning helper: compute lambda + B(t) for the layer-bias correction.

        Caches per (state_sig, target_sig, depth). Returns dict with:
            'lambda_total', 'lambda_breakdown' ({OR, AND, DEL}), 'B_table'
            (list of length depth+1, indexed by t).
        """
        self._ensure_unbiased_structural_extracted()

        state_facts = _extract_state_facts(state)
        target_set = set(target_facts) if target_facts else set()
        cache_key = (frozenset(state_facts), frozenset(target_set), int(depth))
        cached = self._unbiased_correction_cache.get(cache_key)
        if cached is not None:
            return cached

        # ---- Step 1: Run baseline DP once to get per-layer trajectories. -----
        baseline_result = self.heuristic_propagate(
            state=state,
            goal_facts=None,
            fixed_depth=depth,
            start_time=0.0,
            strategy="baseline",
            cached_table=None,
            debug=False,
        )
        probs_by_layer = baseline_result.probabilities_by_layer

        # Reconstruct action support R_t(a) = product of P_t(f) for f in pre(a).
        action_support_by_layer: Dict[int, Dict[str, float]] = {}
        for t in range(depth + 1):
            layer_probs = probs_by_layer.get(t, {})
            layer_support: Dict[str, float] = {}
            for model in self._action_models:
                sup = 1.0
                for pre in model.preconditions:
                    sup *= _clamp_probability(layer_probs.get(pre, 0.0))
                layer_support[model.name] = _clamp_probability(sup)
            action_support_by_layer[t] = layer_support

        # ---- Step 2: Build F* (target ∪ direct preconditions) and A*. --------
        if not target_set:
            target_set = set(self._goal_facts) if self._goal_facts else set(self._facts)
        f_star: Set[Fact] = set(target_set)
        a_star: Set[str] = set()
        for f in list(target_set):
            for model in self._actions_by_effect_fact.get(f, []):
                a_star.add(model.name)
                f_star.update(model.preconditions)
        if not f_star:
            f_star = set(self._facts)
        f_star_list = list(f_star)
        n_facts = max(1, len(f_star_list))

        # ---- Step 3: Marginal scales (averaged over t in [1, depth]). --------
        # <p>_f = mean s_t(a, f) across achievers and t
        p_avg_f: Dict[Fact, float] = {}
        for f in f_star:
            achievers = self._actions_by_effect_fact.get(f, [])
            if not achievers:
                p_avg_f[f] = 0.0
                continue
            sum_val = 0.0
            count = 0
            for t in range(1, depth + 1):
                for model in achievers:
                    d = max(0, int(model.effect_delay_steps))
                    if t < d:
                        continue
                    r = action_support_by_layer.get(t - d, {}).get(model.name, 0.0)
                    add_p = _clamp_probability(model.add_probabilities.get(f, 0.0))
                    sum_val += _clamp_probability(r * add_p)
                    count += 1
            p_avg_f[f] = (sum_val / count) if count > 0 else 0.0

        # <p_shared>_a: mean P_{t-d(a)}(f) over t and shared-achiever preconditions S(a)
        p_shared_a: Dict[str, float] = {}
        s_set_size_by_action: Dict[str, int] = {}
        for a_name in a_star:
            model = self._unbiased_name_to_model.get(a_name)
            if model is None:
                p_shared_a[a_name] = 0.0
                s_set_size_by_action[a_name] = 0
                continue
            pres = list(model.preconditions)
            shared_set: Set[Fact] = set()
            for i in range(len(pres)):
                for j in range(i + 1, len(pres)):
                    ach_i = {m.name for m in self._actions_by_effect_fact.get(pres[i], [])}
                    ach_j = {m.name for m in self._actions_by_effect_fact.get(pres[j], [])}
                    if ach_i & ach_j:
                        shared_set.add(pres[i])
                        shared_set.add(pres[j])
            s_set_size_by_action[a_name] = len(shared_set)
            if not shared_set:
                p_shared_a[a_name] = 0.0
                continue
            d = max(0, int(model.effect_delay_steps))
            sum_val = 0.0
            count = 0
            for t in range(1, depth + 1):
                t_eff = t - d
                if t_eff < 0:
                    continue
                layer_probs = probs_by_layer.get(t_eff, {})
                for f in shared_set:
                    sum_val += _clamp_probability(layer_probs.get(f, 0.0))
                    count += 1
            p_shared_a[a_name] = (sum_val / count) if count > 0 else 0.0

        # <pi_del>_f: mean per-step probability that some delete-action of f fires
        pi_del_f: Dict[Fact, float] = {}
        for f in f_star:
            deleters = self._unbiased_deleters_by_fact.get(f, [])
            if not deleters:
                pi_del_f[f] = 0.0
                continue
            sum_val = 0.0
            count = 0
            for t in range(1, depth + 1):
                step_total = 0.0
                for a_name in deleters:
                    model = self._unbiased_name_to_model.get(a_name)
                    if model is None:
                        continue
                    d = max(0, int(model.effect_delay_steps))
                    if t < d:
                        continue
                    step_total += action_support_by_layer.get(t - d, {}).get(a_name, 0.0)
                sum_val += min(1.0, step_total)
                count += 1
            pi_del_f[f] = (sum_val / count) if count > 0 else 0.0

        # ---- Step 4: Structural constants alpha_f, beta_a, gamma_f. ----------
        alpha_f: Dict[Fact, float] = {}
        n_f_count: Dict[Fact, int] = {}
        for f in f_star:
            achievers = self._actions_by_effect_fact.get(f, [])
            n = len(achievers)
            n_f_count[f] = n
            if n < 2:
                alpha_f[f] = 0.0
                continue
            total_pairs = n * (n - 1) // 2
            mutex_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if self._unbiased_actions_are_mutex(achievers[i].name, achievers[j].name):
                        mutex_count += 1
            alpha_f[f] = (mutex_count / total_pairs) if total_pairs > 0 else 0.0

        beta_a: Dict[str, float] = {}
        m_a_count: Dict[str, int] = {}
        for a_name in a_star:
            model = self._unbiased_name_to_model.get(a_name)
            if model is None:
                beta_a[a_name] = 0.0
                m_a_count[a_name] = 0
                continue
            pres = list(model.preconditions)
            m = len(pres)
            m_a_count[a_name] = m
            if m < 2:
                beta_a[a_name] = 0.0
                continue
            total_pairs = m * (m - 1) // 2
            shared_pairs = 0
            for i in range(m):
                ach_i = {mod.name for mod in self._actions_by_effect_fact.get(pres[i], [])}
                for j in range(i + 1, m):
                    ach_j = {mod.name for mod in self._actions_by_effect_fact.get(pres[j], [])}
                    if ach_i & ach_j:
                        shared_pairs += 1
            beta_a[a_name] = (shared_pairs / total_pairs) if total_pairs > 0 else 0.0

        gamma_f: Dict[Fact, float] = {}
        for f in f_star:
            ach_count = len(self._actions_by_effect_fact.get(f, []))
            del_count = len(self._unbiased_deleters_by_fact.get(f, []))
            denom = ach_count + del_count
            gamma_f[f] = (del_count / denom) if denom > 0 else 0.0

        # ---- Step 5: Per-source lambda contributions (avg of per-fact/action). -
        or_terms: List[float] = []
        for f in f_star:
            n = n_f_count.get(f, 0)
            if n < 2:
                continue
            or_terms.append(
                -alpha_f.get(f, 0.0) * (n - 1) / 2.0 * p_avg_f.get(f, 0.0)
            )
        lambda_or = (sum(or_terms) / len(or_terms)) if or_terms else 0.0

        and_terms: List[float] = []
        for a_name in a_star:
            m = m_a_count.get(a_name, 0)
            if m < 2:
                continue
            ps = p_shared_a.get(a_name, 0.0)
            and_terms.append(
                -beta_a.get(a_name, 0.0) * (m - 1) / 2.0 * ps * (1.0 - ps)
            )
        lambda_and = (sum(and_terms) / len(and_terms)) if and_terms else 0.0

        # λ_DEL averages only over facts that actually have deleters, so the
        # contribution of delete-prone facts is not diluted by the (often
        # large) set of goal facts that have no deleters at all. Without this
        # filter, a domain with one delete-prone fact among 20 goals would see
        # λ_DEL shrunk by 20× for no principled reason.
        del_terms: List[float] = []
        for f in f_star:
            if not self._unbiased_deleters_by_fact.get(f):
                continue  # no deleters → this fact contributes 0 to delete bias
            del_terms.append(+gamma_f.get(f, 0.0) * pi_del_f.get(f, 0.0))
        lambda_del = (sum(del_terms) / len(del_terms)) if del_terms else 0.0

        lambda_total = lambda_or + lambda_and + lambda_del

        # ---- Step 6: B(t) recursion using <P_t>, <H_t> from baseline DP. ------
        if abs(lambda_total) < 1e-4:
            B_table: List[float] = [0.0] * (depth + 1)
        else:
            avg_P: List[float] = []
            for t in range(depth + 1):
                layer_probs = probs_by_layer.get(t, {})
                s = 0.0
                for f in f_star_list:
                    s += _clamp_probability(layer_probs.get(f, 0.0))
                avg_P.append(s / n_facts)

            # Derive <H_t> from <P_t> via the recursion identity:
            #   P_t = P_{t-1} + (1 - P_{t-1}) * H_t  =>  H_t = (P_t - P_{t-1}) / (1 - P_{t-1})
            avg_H: List[float] = [0.0]
            for t in range(1, depth + 1):
                p_prev = avg_P[t - 1]
                p_now = avg_P[t]
                if p_prev >= 1.0:
                    avg_H.append(0.0)
                else:
                    avg_H.append(_clamp_probability((p_now - p_prev) / (1.0 - p_prev)))

            B_table = [0.0]
            for t in range(1, depth + 1):
                B_prev = B_table[t - 1]
                H_t = avg_H[t]
                P_prev = avg_P[t - 1]
                B_t = B_prev * (1.0 - H_t) + (1.0 - P_prev) * lambda_total * H_t
                # Defensive clip: B(t) is in [-1, 1] by construction.
                B_table.append(max(-1.0, min(1.0, B_t)))

        # Mutex / shared-achiever / delete counts for diagnostic visibility.
        # These help quickly answer "is λ small because the structure has no
        # bias to extract, or because the extractor is missing it?"
        mutex_pair_total = 0
        mutex_pair_detected = 0
        for f in f_star:
            achievers = self._actions_by_effect_fact.get(f, [])
            n = len(achievers)
            if n < 2:
                continue
            mutex_pair_total += n * (n - 1) // 2
            mutex_pair_detected += int(round(
                alpha_f.get(f, 0.0) * (n * (n - 1) // 2)
            ))
        deleters_total = sum(
            1 for f in f_star if self._unbiased_deleters_by_fact.get(f)
        )

        result = {
            "lambda_total": lambda_total,
            "lambda_breakdown": {
                "OR": lambda_or,
                "AND": lambda_and,
                "DEL": lambda_del,
            },
            "B_table": B_table,
            "diagnostics": {
                "n_facts_in_f_star": n_facts,
                "n_actions_in_a_star": len(a_star),
                "mutex_pairs_detected": mutex_pair_detected,
                "mutex_pairs_total": mutex_pair_total,
                "facts_with_deleters": deleters_total,
                "B_max": max((abs(b) for b in B_table), default=0.0),
            },
        }
        self._unbiased_correction_cache[cache_key] = result
        return result

    def diagnose_unbiased(
        self,
        state,
        goal_facts: Optional[Iterable[Fact]] = None,
        depth: int = 25,
    ) -> str:
        """
        Pretty-print λ breakdown and B(t) summary for a single state.

        Use this when MCTS appears to ignore the unbiased correction — it
        tells you whether the correction has any magnitude at all. If
        |λ_total| < 0.001 the correction is mathematically zero; the BFS
        pattern in MCTS is then *not* a bias-correction problem and chasing
        further fixes here will not help.
        """
        target_facts = (
            set(goal_facts) if goal_facts is not None
            else (set(self._goal_facts) if self._goal_facts else set(self._facts))
        )
        result = self._compute_unbiased_correction_table(
            state=state, target_facts=target_facts, depth=depth,
        )
        diag = result.get("diagnostics", {})
        breakdown = result.get("lambda_breakdown", {})
        B = result.get("B_table", [])
        sample_t = [0, depth // 4, depth // 2, (3 * depth) // 4, depth]
        sample_B = [(t, B[t]) for t in sample_t if 0 <= t < len(B)]
        lines = [
            f"λ_total = {result.get('lambda_total', 0.0):+.4f}",
            f"  OR  = {breakdown.get('OR', 0.0):+.4f}  (mutex achievers)",
            f"  AND = {breakdown.get('AND', 0.0):+.4f}  (shared-achiever preconditions)",
            f"  DEL = {breakdown.get('DEL', 0.0):+.4f}  (delete-induced bias)",
            f"|B|_max = {diag.get('B_max', 0.0):.4f}",
            f"B(t) samples: " + ", ".join(f"B({t})={b:+.4f}" for t, b in sample_B),
            f"f*: {diag.get('n_facts_in_f_star', 0)} facts, "
            f"a*: {diag.get('n_actions_in_a_star', 0)} actions",
            f"mutex pairs: {diag.get('mutex_pairs_detected', 0)} / "
            f"{diag.get('mutex_pairs_total', 0)} candidate pairs",
            f"facts with deleters: {diag.get('facts_with_deleters', 0)}",
        ]
        return "\n".join(lines)

    def _heuristic_propagate_atom_backtrack_exact_unbiased(
        self,
        state,
        goal_facts: Optional[Iterable[Fact]],
        fixed_depth: int,
        start_time: float,
        debug: bool,
        *,
        resolution_alpha: Optional[float] = None,
        resolution_forced_minimum: bool = False,
        resolution_reference_t: Optional[int] = None,
    ) -> TemporalPropagationResult:
        """
        Layer-bias-corrected variant of atom_backtrack_exact_resolution.

        Bias-corrected empirical heuristic: NOT admissible by default. The
        correction subtracts a structural per-layer bias B(t) from each goal
        fact's probability before goal aggregation. See
        ``_compute_unbiased_correction_table`` for the math.
        """
        state_facts = _extract_state_facts(state)
        depth = max(0, int(fixed_depth))
        target_facts: Set[Fact] = (
            set(goal_facts) if goal_facts is not None else self._facts.union(state_facts)
        )

        # Base scoring path: reuse atom_backtrack_exact_resolution unchanged.
        base_result = self._heuristic_propagate_atom_backtrack_exact_resolution(
            state=state,
            goal_facts=goal_facts,
            fixed_depth=depth,
            start_time=start_time,
            debug=debug,
            resolution_alpha=resolution_alpha,
            resolution_forced_minimum=resolution_forced_minimum,
            resolution_reference_t=resolution_reference_t,
        )

        # Pre-planning: compute lambda + B(t) (cached per state/target/depth).
        correction = self._compute_unbiased_correction_table(
            state=state,
            target_facts=target_facts,
            depth=depth,
        )
        B_table = correction["B_table"]
        b_at_depth = (
            B_table[depth]
            if 0 <= depth < len(B_table)
            else (B_table[-1] if B_table else 0.0)
        )

        # Apply the per-layer correction to every fact at the depth layer
        # before goal aggregation. heuristic_score's aggregation reads from
        # probabilities_by_layer[depth_used] directly, so subtracting here is
        # equivalent to "per goal fact, before aggregation".
        base_layer = base_result.probabilities_by_layer.get(depth, {})
        corrected_layer: Dict[Fact, float] = {
            fact: _clamp_probability(prob - b_at_depth)
            for fact, prob in base_layer.items()
        }

        new_probs_by_layer: Dict[int, Dict[Fact, float]] = {
            0: dict(base_result.probabilities_by_layer.get(0, {})),
            depth: corrected_layer,
        }

        traces = list(base_result.traces) if base_result.traces else []

        return TemporalPropagationResult(
            probabilities_by_layer=new_probs_by_layer,
            depth_used=depth,
            traces=traces,
            cache_hit=False,
            fact_cache_hits=base_result.fact_cache_hits,
            action_cache_hits=base_result.action_cache_hits,
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

    def _extract_add_probabilities(self, action) -> Dict[Fact, float]:
        add_probabilities: Dict[Fact, float] = {}
        for fact in getattr(action, "add_effects", set()):
            add_probabilities[fact] = 1.0

        for probabilistic_effect in getattr(action, "probabilistic_effects", []):
            per_effect_probability: MutableMapping[Fact, float] = {}

            # Preferred path: execute the outcome function to read the exact
            # per-fluent add probabilities. This works whenever the callable is
            # evaluable in the current context.
            outcomes = None
            try:
                outcomes = probabilistic_effect.probability_function(
                    SimpleNamespace(predicates=set()),
                    None,
                )
            except Exception:
                outcomes = None

            if outcomes:
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
            else:
                # Fallback: the outcome distribution could not be evaluated
                # (the probability function is an opaque, state-dependent
                # callable that may raise when probed with a placeholder
                # state). Without this branch the affected fluents would get
                # NO achiever and be silently treated as unreachable (P=0) —
                # which is exactly what made machine_shop's probabilistic
                # goals (shaped/smooth/painted/polished) score 0.
                #
                # The set of affected fluents is available *structurally* via
                # `probabilistic_effect.fluents` without executing anything, so
                # register each as an achiever with a fallback probability.
                # Sign is assumed positive (add); deterministic deletes are
                # captured separately from `del_effects`.
                for fact in getattr(probabilistic_effect, "fluents", []):
                    per_effect_probability[fact] = _clamp_probability(
                        self._FALLBACK_PROBABILISTIC_ADD_PROB
                    )

            for fact, probability in per_effect_probability.items():
                existing = add_probabilities.get(fact, 0.0)
                add_probabilities[fact] = _clamp_probability(
                    1.0 - (1.0 - existing) * (1.0 - probability)
                )

        return add_probabilities
