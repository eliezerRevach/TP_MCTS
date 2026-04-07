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
        # Cross-query memoization to reuse the same fixed-depth computation.
        self._query_cache: Dict[Tuple[frozenset[Fact], int, int], TemporalPropagationResult] = {}

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
        debug: bool = False,
    ) -> TemporalPropagationResult:
        state_facts = _extract_state_facts(state)
        start_layer = max(0, int(math.floor(start_time)))
        query_key = (frozenset(state_facts), int(fixed_depth), start_layer)
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
        debug: bool = False,
    ):
        result = self.heuristic_propagate(
            state=state,
            fixed_depth=fixed_depth,
            start_time=start_time,
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
        return score

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
