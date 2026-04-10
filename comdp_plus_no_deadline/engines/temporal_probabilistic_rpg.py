from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from types import SimpleNamespace
import math
from typing import Dict, Hashable, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from comdp_plus_no_deadline.engines.probabilistic_rpg import (
    compute_precondition_support,
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
    # Carries the FormulaCache produced by atom_backtrack_cached so the
    # caller can commit it after mdp.step().
    formula_cache: Optional["FormulaCache"] = None
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


@dataclass
class FormulaCache:
    """Persistent memoization for atom_backtrack_cached across greedy steps.

    Stores the same memos as a single atom_backtrack_exact call, plus a
    reverse-dependency index so that when state facts change only the memo
    entries that transitively read those facts are evicted.

    Design (Option A -- transitive base-fact tracking):
      Each memo entry records the complete set of state facts it read as
      base cases during its computation.  On state change we look up every
      memo key that depended on a changed fact and pop it.  Re-running
      fact_probability then recomputes only those evicted entries.
    """

    fact_memo: Dict[Tuple[Fact, int], float]
    action_term_memo: Dict[Tuple[str, Fact, int], float]
    # reverse_deps_fact[base_fact] = set of (fact, horizon) memo keys that
    # transitively read base_fact as a state base case.
    reverse_deps_fact: Dict[Fact, Set[Tuple[Fact, int]]]
    # reverse_deps_action[base_fact] = set of (action_name, fact, horizon)
    # memo keys that transitively read base_fact.
    reverse_deps_action: Dict[Fact, Set[Tuple[str, Fact, int]]]
    state_facts: frozenset

    def invalidate(self, changed_facts: Set[Fact]) -> None:
        """Evict all memo entries that depend on any of the changed facts."""
        for base_fact in changed_facts:
            for key in self.reverse_deps_fact.pop(base_fact, set()):
                self.fact_memo.pop(key, None)
            for key in self.reverse_deps_action.pop(base_fact, set()):
                self.action_term_memo.pop(key, None)
        # Evict reverse-dep entries for changed facts that are no longer
        # present (they may have migrated from one side of the base case).
        # state_facts is updated by the caller after invalidate().


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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

    def __init__(self, actions: Sequence[object], facts: Optional[Iterable[Fact]] = None):
        self._actions = list(actions)
        self._facts: Set[Fact] = set(facts or [])
        self._action_models = self._build_action_models()
        self._fact_dependency_graph = self._build_fact_dependency_graph()
        self._actions_by_effect_fact = self._build_actions_by_effect_fact()
        # Cross-query memoization to reuse the same fixed-depth computation.
        self._query_cache: Dict[Tuple[frozenset[Fact], int, int, str], TemporalPropagationResult] = {}
        # Dedicated memoization for the atom strategy recurrence.
        self._atom_split_cache: Dict[Tuple[frozenset[Fact], Fact, int, str], float] = {}

    @classmethod
    def from_problem(cls, problem) -> "TemporalProbabilisticRPGHeuristic":
        facts = set(getattr(problem, "initial_values", {}).keys())
        facts.update(getattr(problem, "goals", set()))
        return cls(getattr(problem, "actions", []), facts=facts)

    def heuristic_propagate(
        self,
        state,
        fixed_depth: int = 25,
        start_time: float = 0.0,
        strategy: str = "baseline",
        cached_table=None,
        debug: bool = False,
    ) -> TemporalPropagationResult:
        state_facts = _extract_state_facts(state)
        start_layer = max(0, int(math.floor(start_time)))
        chosen_strategy = self._normalize_strategy(strategy)
        query_key = (frozenset(state_facts), int(fixed_depth), start_layer, chosen_strategy)
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
                fixed_depth=fixed_depth,
                start_time=start_time,
                debug=debug,
            )
            self._query_cache[query_key] = result
            return result
        if chosen_strategy == "atom_backtrack_cached":
            # cached_table carries a FormulaCache (or None) for this strategy.
            formula_cache = cached_table if isinstance(cached_table, FormulaCache) else None
            result = self._heuristic_propagate_atom_backtrack_cached(
                state=state,
                fixed_depth=fixed_depth,
                start_time=start_time,
                formula_cache=formula_cache,
                debug=debug,
            )
            # Do NOT store in _query_cache: the result depends on the live
            # FormulaCache state and must not be replayed from a stale snapshot.
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
    ):
        result = self.heuristic_propagate(
            state=state,
            fixed_depth=fixed_depth,
            start_time=start_time,
            strategy=strategy,
            cached_table=cached_table,
            debug=debug,
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
            # atom_backtrack_cached: return the live FormulaCache object.
            if result.formula_cache is not None:
                return score, result.formula_cache
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

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        value = (strategy or "baseline").strip().lower()
        valid = {
            "baseline",
            "baseline_cached",
            "atom_half_split",
            "atom_backtrack_exact",
            "atom_backtrack_cached",
        }
        if value not in valid:
            raise ValueError(
                f"Unknown temporal heuristic strategy: {strategy!r}. "
                "Supported strategies: baseline, baseline_cached, atom_half_split, "
                "atom_backtrack_exact, atom_backtrack_cached."
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

    def _heuristic_propagate_atom_backtrack_exact(
        self,
        state,
        fixed_depth: int,
        start_time: float,
        debug: bool,
    ) -> TemporalPropagationResult:
        del start_time
        state_facts = _extract_state_facts(state)
        depth = max(0, int(fixed_depth))
        facts = self._facts.union(state_facts)
        probabilities_by_layer: Dict[int, Dict[Fact, float]] = {
            t: {fact: 0.0 for fact in facts} for t in range(depth + 1)
        }
        for fact in facts:
            probabilities_by_layer[0][fact] = 1.0 if fact in state_facts else 0.0

        achievers_by_fact: Dict[Fact, List[TemporalRelaxedActionModel]] = {}
        for action_model in self._action_models:
            for fact in action_model.add_probabilities.keys():
                achievers_by_fact.setdefault(fact, []).append(action_model)

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
                # Conservative fallback for cyclic dependencies.
                return 0.0

            if fact in state_facts:
                value = 1.0
            elif horizon == 0:
                value = 0.0
            else:
                recursion_stack.add(key)
                failure = 1.0
                for action_model in achievers_by_fact.get(fact, []):
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

        for fact in facts:
            probabilities_by_layer[depth][fact] = fact_probability(fact, depth)

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
        fixed_depth: int,
        start_time: float,
        formula_cache: Optional[FormulaCache],
        debug: bool,
    ) -> TemporalPropagationResult:
        """atom_backtrack_exact with persistent cross-call memoization.

        On the first call (formula_cache is None) the computation is identical
        to atom_backtrack_exact.  On subsequent calls only memo entries that
        transitively depended on the facts that changed between states are
        evicted; all other entries survive and serve as cache hits.

        Dependency tracking uses Option A (transitive base-fact sets): each
        memo entry accumulates the full set of state-level base facts it reads
        so that a single invalidation pass is sufficient.
        """
        del start_time
        state_facts = _extract_state_facts(state)
        depth = max(0, int(fixed_depth))
        facts = self._facts.union(state_facts)

        achievers_by_fact: Dict[Fact, List[TemporalRelaxedActionModel]] = {}
        for action_model in self._action_models:
            for fact in action_model.add_probabilities.keys():
                achievers_by_fact.setdefault(fact, []).append(action_model)

        # Set up or reuse FormulaCache.
        if formula_cache is None or formula_cache.state_facts is None:
            fc = FormulaCache(
                fact_memo={},
                action_term_memo={},
                reverse_deps_fact={},
                reverse_deps_action={},
                state_facts=frozenset(state_facts),
            )
        else:
            fc = formula_cache
            changed_facts = set(fc.state_facts.symmetric_difference(state_facts))
            if changed_facts:
                fc.invalidate(changed_facts)
            fc.state_facts = frozenset(state_facts)

        fact_cache_hits = 0
        action_cache_hits = 0
        recursion_stack: Set[Tuple[Fact, int]] = set()

        def is_atom_action(action_model: TemporalRelaxedActionModel) -> bool:
            return (
                len(action_model.preconditions) == 0
                or action_model.preconditions.issubset(state_facts)
            )

        def fact_probability(fact: Fact, horizon: int) -> Tuple[float, Set[Fact]]:
            """Return (probability, base_fact_deps) for (fact, horizon).

            base_fact_deps is the transitive set of state facts read as base
            cases during this computation.  Propagated upward so every ancestor
            memo entry records the complete dependency set.
            """
            nonlocal fact_cache_hits, action_cache_hits
            horizon = int(horizon)
            if horizon < 0:
                return 0.0, {fact}
            key = (fact, horizon)
            if key in fc.fact_memo:
                fact_cache_hits += 1
                deps: Set[Fact] = set()
                for base_fact, keys_set in fc.reverse_deps_fact.items():
                    if key in keys_set:
                        deps.add(base_fact)
                return fc.fact_memo[key], deps
            if key in recursion_stack:
                return 0.0, set()

            all_base_deps: Set[Fact] = set()

            if fact in state_facts:
                value = 1.0
                all_base_deps.add(fact)
            elif horizon == 0:
                value = 0.0
                all_base_deps.add(fact)
            else:
                recursion_stack.add(key)
                failure = 1.0
                for action_model in achievers_by_fact.get(fact, []):
                    action_key = (action_model.name, fact, horizon)
                    if action_key in fc.action_term_memo:
                        action_cache_hits += 1
                        achiever_failure = fc.action_term_memo[action_key]
                        for base_fact, keys_set in fc.reverse_deps_action.items():
                            if action_key in keys_set:
                                all_base_deps.add(base_fact)
                    else:
                        delay = max(0, int(action_model.effect_delay_steps))
                        add_probability = _clamp_probability(
                            action_model.add_probabilities.get(fact, 0.0)
                        )
                        action_base_deps: Set[Fact] = set()
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
                                action_base_deps.update(action_model.preconditions)
                            else:
                                achiever_failure = 1.0
                                for completion_time in range(first_completion, horizon + 1):
                                    available_horizon = completion_time - delay
                                    precondition_support = 1.0
                                    for precondition in action_model.preconditions:
                                        pre_val, pre_deps = fact_probability(
                                            precondition,
                                            available_horizon,
                                        )
                                        precondition_support *= pre_val
                                        action_base_deps.update(pre_deps)
                                    step_success = _clamp_probability(
                                        add_probability * precondition_support
                                    )
                                    achiever_failure *= 1.0 - step_success
                                achiever_failure = _clamp_probability(achiever_failure)

                        fc.action_term_memo[action_key] = achiever_failure
                        for base_fact in action_base_deps:
                            fc.reverse_deps_action.setdefault(base_fact, set()).add(action_key)
                        all_base_deps.update(action_base_deps)

                    failure *= achiever_failure
                recursion_stack.remove(key)
                value = _clamp_probability(1.0 - failure)

            fc.fact_memo[key] = value
            for base_fact in all_base_deps:
                fc.reverse_deps_fact.setdefault(base_fact, set()).add(key)
            return value, all_base_deps

        probabilities_by_layer: Dict[int, Dict[Fact, float]] = {
            0: {fact: (1.0 if fact in state_facts else 0.0) for fact in facts},
            depth: {},
        }
        for fact in facts:
            prob, _ = fact_probability(fact, depth)
            probabilities_by_layer[depth][fact] = prob

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
            formula_cache=fc,
        )

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
