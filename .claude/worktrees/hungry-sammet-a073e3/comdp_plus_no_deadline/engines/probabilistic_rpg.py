from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from types import SimpleNamespace
from typing import Any, Dict, Hashable, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


Fact = Hashable


@dataclass(frozen=True)
class RelaxedActionModel:
    """Fact-level optimistic abstraction of an action."""

    name: str
    preconditions: frozenset[Fact]
    precondition_structure: object
    add_probabilities: Mapping[Fact, float]


@dataclass
class LayerTrace:
    """Debug snapshot for one propagation layer."""

    layer: int
    fact_probabilities: Dict[Fact, float]
    action_support: Dict[str, float] = field(default_factory=dict)
    action_support_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fact_hazards: Dict[Fact, float] = field(default_factory=dict)
    max_delta: float = 0.0
    warning: Optional[str] = None


@dataclass
class PropagationResult:
    """Output bundle returned by ``heuristic_propagate``."""

    probabilities: Dict[Fact, float]
    layers_run: int
    reached_goal_threshold: bool
    converged: bool
    cyclic_dependency: bool
    warnings: List[str]
    traces: List[LayerTrace]


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_state_facts(state) -> Set[Fact]:
    if hasattr(state, "predicates"):
        return set(state.predicates)
    return set(state)


def _reference_state_from_problem(problem) -> Optional[object]:
    """Build an engine ``State`` from problem initial values for PE evaluation."""
    initial = getattr(problem, "initial_values", None)
    if not initial:
        return None
    from unified_planning.engines.state import State

    pos = {k for k, v in initial.items() if v.bool_constant_value()}
    return State(pos)


@dataclass(frozen=True)
class PreconditionSupportResult:
    """Computed support for a relaxed precondition formula."""

    probability: float
    structure: str
    mode: str
    warning: Optional[str] = None


def _is_clause_container(value: object) -> bool:
    return isinstance(value, (set, frozenset, list, tuple))


def _is_atomic_fact(value: object) -> bool:
    return not _is_clause_container(value)


def _is_dnf_structure(precondition: object) -> bool:
    return isinstance(precondition, (list, tuple)) and bool(precondition) and all(
        _is_clause_container(clause) for clause in precondition
    )


def _normalize_clause(clause: object) -> Tuple[Fact, ...]:
    if _is_atomic_fact(clause):
        return (clause,)
    if not _is_clause_container(clause):
        raise TypeError(f"Unsupported clause type: {type(clause)!r}")

    normalized: List[Fact] = []
    for fact in clause:
        if _is_clause_container(fact):
            raise TypeError("Nested clause containers are only supported in explicit DNF mode.")
        normalized.append(fact)
    return tuple(normalized)


def _format_clause(clause: Sequence[Fact]) -> str:
    if not clause:
        return "TRUE"
    return " AND ".join(str(fact) for fact in clause)


def _format_precondition_structure(precondition: object) -> str:
    if _is_atomic_fact(precondition):
        return str(precondition)
    if _is_dnf_structure(precondition):
        clauses = [_normalize_clause(clause) for clause in precondition]
        return " OR ".join(f"({_format_clause(clause)})" for clause in clauses)
    clause = _normalize_clause(precondition)
    if len(clause) == 1:
        return str(clause[0])
    return f"({_format_clause(clause)})"


def _compute_clause_support(clause: Sequence[Fact], fact_probs: Mapping[Fact, float]) -> float:
    """Return the relaxed support of one conjunctive clause."""

    # Summing fact probabilities is wrong because an action needs all conjuncts,
    # not "any one of them". Under the heuristic relaxation we approximate joint
    # support by multiplying fact supports as if they were independent.
    return _clamp_probability(prod(_clamp_probability(fact_probs.get(fact, 0.0)) for fact in clause))


def _compute_precondition_support_result(
    precondition: object,
    fact_probs: Mapping[Fact, float],
    strict: bool = True,
) -> PreconditionSupportResult:
    """
    Compute ``R_t(a)``, the relaxed support of an action precondition formula.

    Supported forms:
    - atomic fact: ``x``
    - conjunction: ``(x1, x2, ..., xk)`` or ``{x1, ..., xk}``
    - explicit DNF: ``((...), (...), ...)`` where each inner clause is conjunctive

    For conjunctions we use a product because every conjunct must hold. For DNF
    with pairwise-disjoint clauses, ``1 - prod(1 - S_j)`` is valid under the same
    independence relaxation. Overlapping clauses share literals, so clause events
    are correlated; treating them as independent would be an approximation.
    """

    structure = _format_precondition_structure(precondition)

    if _is_atomic_fact(precondition):
        probability = _clamp_probability(fact_probs.get(precondition, 0.0))
        return PreconditionSupportResult(
            probability=probability,
            structure=structure,
            mode="exact_conjunctive",
        )

    if _is_dnf_structure(precondition):
        clauses = [_normalize_clause(clause) for clause in precondition]
        clause_supports = [_compute_clause_support(clause, fact_probs) for clause in clauses]
        clause_sets = [set(clause) for clause in clauses]
        overlap = False
        for index, clause_set in enumerate(clause_sets):
            for other_clause_set in clause_sets[index + 1 :]:
                if clause_set.intersection(other_clause_set):
                    overlap = True
                    break
            if overlap:
                break

        if overlap:
            warning = (
                "Overlapping DNF clauses share literals, so clause supports are "
                "correlated under the heuristic relaxation."
            )
            if strict:
                raise NotImplementedError(warning)
            probability = _clamp_probability(1.0 - prod(1.0 - support for support in clause_supports))
            return PreconditionSupportResult(
                probability=probability,
                structure=structure,
                mode="approximate_overlap",
                warning=warning,
            )

        probability = _clamp_probability(1.0 - prod(1.0 - support for support in clause_supports))
        return PreconditionSupportResult(
            probability=probability,
            structure=structure,
            mode="exact_disjoint_dnf",
        )

    clause = _normalize_clause(precondition)
    return PreconditionSupportResult(
        probability=_compute_clause_support(clause, fact_probs),
        structure=structure,
        mode="exact_conjunctive",
    )


def compute_precondition_support(
    precondition: object,
    fact_probs: Mapping[Fact, float],
    strict: bool = True,
) -> float:
    """Return only the relaxed support probability ``R_t(a)`` for a precondition."""

    return _compute_precondition_support_result(
        precondition=precondition,
        fact_probs=fact_probs,
        strict=strict,
    ).probability


class ProbabilisticOptimisticRPGHeuristic:
    """
    Optimistic probabilistic RPG-style heuristic with monotone retry updates.

    For each fact ``x`` we maintain a relaxed probability ``P_t(x)`` that the fact
    has been achieved by the end of layer ``t``. Preconditions are supported by
    the current relaxed fact probabilities:

    ``R_t(a) = support_t(pre(a))``

    In the current runtime environment, ``pre(a)`` is a flat conjunction of
    positive facts, so ``support_t(pre(a))`` is the product of those fact
    supports. The helper also accepts explicit DNF structures for tests and
    future extensions.

    Each fact receives a per-layer success hazard from all of its achievers:

    ``H_t(x) = 1 - prod_{a in Ach(x)} (1 - add_prob(a, x) * R_t(a))``

    Retry-aware accumulation is then:

    ``P_{t+1}(x) = P_t(x) + (1 - P_t(x)) * H_t(x)``

    This is an optimistic relaxed heuristic:
    - delete effects are ignored
    - failures have no negative side effects
    - action outcomes are treated as independent
    - retries are compressed into layered hazard updates
    """

    def __init__(
        self,
        actions: Sequence[object],
        facts: Optional[Iterable[Fact]] = None,
        reference_state_for_structure: Optional[object] = None,
    ):
        self._actions = list(actions)
        self._facts: Set[Fact] = set(facts or [])
        # Probabilistic-effect lambdas need a real state with ``predicates`` when
        # building the dependency graph; use initial state from the problem when available.
        self._reference_state_for_structure = reference_state_for_structure
        self._graph_action_models = self._build_structural_models()
        self._achievers = self._build_achievers(self._graph_action_models)
        self._is_acyclic = self._detect_acyclic_dependency_graph(self._graph_action_models)

    @classmethod
    def from_problem(cls, problem) -> "ProbabilisticOptimisticRPGHeuristic":
        facts = set(getattr(problem, "initial_values", {}).keys())
        facts.update(getattr(problem, "goals", set()))
        ref = _reference_state_from_problem(problem)
        return cls(
            getattr(problem, "actions", []),
            facts=facts,
            reference_state_for_structure=ref,
        )

    @property
    def facts(self) -> Set[Fact]:
        return set(self._facts)

    @property
    def is_acyclic(self) -> bool:
        return self._is_acyclic

    def heuristic_propagate(
        self,
        state,
        goal_facts: Optional[Iterable[Fact]] = None,
        max_layers: int = 25,
        epsilon: float = 1e-6,
        goal_threshold: float = 0.99,
        debug: bool = False,
    ) -> PropagationResult:
        """
        Propagate relaxed fact probabilities from ``state``.

        If the fact-action dependency graph is acyclic, we run ordinary layered
        forward propagation. If a cycle is detected, we keep the same monotone
        recurrence but treat it as fixed-point iteration and stop when the maximum
        update falls below ``epsilon`` or ``max_layers`` is reached.
        """

        state_facts = _extract_state_facts(state)
        goals = set(goal_facts or [])
        probabilities = {
            fact: 1.0 if fact in state_facts else 0.0
            for fact in self._facts.union(state_facts).union(goals)
        }
        action_models = self._instantiate_action_models(state)
        achievers = self._build_achievers(action_models)
        warnings: List[str] = []
        traces: List[LayerTrace] = []
        converged = False
        reached_goal_threshold = not goals or all(probabilities.get(goal, 0.0) >= goal_threshold for goal in goals)
        cycle_warning = None

        if not self._is_acyclic:
            cycle_warning = (
                "Cycle detected in the fact-action dependency graph; "
                "using monotone fixed-point iteration."
            )
            warnings.append(cycle_warning)

        if debug:
            traces.append(
                LayerTrace(
                    layer=0,
                    fact_probabilities=dict(probabilities),
                    warning=cycle_warning,
                )
            )

        if reached_goal_threshold:
            return PropagationResult(
                probabilities=probabilities,
                layers_run=0,
                reached_goal_threshold=True,
                converged=True,
                cyclic_dependency=not self._is_acyclic,
                warnings=warnings,
                traces=traces,
            )

        layers_run = 0
        for layer in range(1, max_layers + 1):
            action_support, action_support_details = self._compute_action_support(
                probabilities,
                action_models,
            )
            hazards, next_probabilities = self._advance_probabilities(
                probabilities,
                action_support,
                achievers,
            )
            max_delta = max(
                (next_probabilities[fact] - probabilities[fact]) for fact in next_probabilities
            ) if next_probabilities else 0.0

            layers_run = layer
            probabilities = next_probabilities

            if debug:
                traces.append(
                    LayerTrace(
                        layer=layer,
                        fact_probabilities=dict(probabilities),
                        action_support=action_support,
                        action_support_details=action_support_details,
                        fact_hazards=hazards,
                        max_delta=max_delta,
                    )
                )

            if goals and all(probabilities.get(goal, 0.0) >= goal_threshold for goal in goals):
                reached_goal_threshold = True
                break

            if max_delta < epsilon:
                converged = True
                if not self._is_acyclic:
                    break
                if all(probabilities.get(goal, 0.0) == 1.0 for goal in goals):
                    break

        if not converged and layers_run < max_layers:
            converged = True
        if layers_run == max_layers and not reached_goal_threshold:
            converged = converged or False

        return PropagationResult(
            probabilities=probabilities,
            layers_run=layers_run,
            reached_goal_threshold=reached_goal_threshold,
            converged=converged,
            cyclic_dependency=not self._is_acyclic,
            warnings=warnings,
            traces=traces,
        )

    def heuristic_score(
        self,
        state,
        goal_facts: Iterable[Fact],
        aggregation: str = "product",
        max_layers: int = 25,
        epsilon: float = 1e-6,
        goal_threshold: float = 0.99,
        debug: bool = False,
    ):
        """
        Score a state with either product or minimum goal aggregation.

        ``product`` is more aggressive and assumes goal-fact independence.
        ``min`` is a more conservative bottleneck score.
        """

        goals = tuple(goal_facts)
        result = self.heuristic_propagate(
            state,
            goal_facts=goals,
            max_layers=max_layers,
            epsilon=epsilon,
            goal_threshold=goal_threshold,
            debug=debug,
        )

        goal_probabilities = [_clamp_probability(result.probabilities.get(goal, 0.0)) for goal in goals]

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

    def _build_structural_models(self) -> List[RelaxedActionModel]:
        ref_state = self._reference_state_for_structure
        if ref_state is None:
            # PE lambdas only need ``state.predicates``; avoid importing ``up.engines``
            # before submodules load (breaks lightweight tests).
            ref_state = SimpleNamespace(predicates=set())

        models: List[RelaxedActionModel] = []
        for action in self._actions:
            if hasattr(action, "actions") and not hasattr(action, "add_effects"):
                continue
            preconditions = frozenset(getattr(action, "pos_preconditions", set()))
            add_probabilities = self._extract_add_probabilities(action, state=ref_state)
            if not add_probabilities and not preconditions:
                continue
            self._facts.update(preconditions)
            self._facts.update(add_probabilities.keys())
            models.append(
                RelaxedActionModel(
                    name=getattr(action, "name", repr(action)),
                    preconditions=preconditions,
                    precondition_structure=preconditions,
                    add_probabilities=add_probabilities,
                )
            )
        return models

    def _instantiate_action_models(self, state) -> List[RelaxedActionModel]:
        models: List[RelaxedActionModel] = []
        state_facts = _extract_state_facts(state)
        for action in self._actions:
            if hasattr(action, "actions") and not hasattr(action, "add_effects"):
                continue
            preconditions = frozenset(getattr(action, "pos_preconditions", set()))
            add_probabilities = self._extract_add_probabilities(action, state=state)
            if not add_probabilities and not preconditions:
                continue
            self._facts.update(preconditions)
            self._facts.update(add_probabilities.keys())
            self._facts.update(state_facts)
            models.append(
                RelaxedActionModel(
                    name=getattr(action, "name", repr(action)),
                    preconditions=preconditions,
                    precondition_structure=preconditions,
                    add_probabilities=add_probabilities,
                )
            )
        return models

    def _extract_add_probabilities(self, action, state) -> Dict[Fact, float]:
        add_probabilities: Dict[Fact, float] = {}

        for fact in getattr(action, "add_effects", set()):
            add_probabilities[fact] = 1.0

        synthetic_adds = getattr(action, "probabilistic_add_effects", None)
        if synthetic_adds:
            for fact, probability in synthetic_adds.items():
                add_probabilities[fact] = self._merge_independent_success(
                    add_probabilities.get(fact, 0.0),
                    _clamp_probability(probability),
                )

        for probabilistic_effect in getattr(action, "probabilistic_effects", []):
            outcomes = probabilistic_effect.probability_function(state, None)
            if not outcomes:
                continue

            per_effect_probability: MutableMapping[Fact, float] = {}
            for outcome_probability, assignments in outcomes.items():
                outcome_probability = _clamp_probability(outcome_probability)
                for fact, value in assignments.items():
                    is_positive = False
                    if hasattr(value, "bool_constant_value"):
                        is_positive = bool(value.bool_constant_value())
                    elif hasattr(value, "constant_value"):
                        is_positive = bool(value.constant_value())
                    else:
                        is_positive = bool(value)
                    if is_positive:
                        per_effect_probability[fact] = _clamp_probability(
                            per_effect_probability.get(fact, 0.0) + outcome_probability
                        )

            for fact, probability in per_effect_probability.items():
                add_probabilities[fact] = self._merge_independent_success(
                    add_probabilities.get(fact, 0.0),
                    probability,
                )

        return {fact: _clamp_probability(probability) for fact, probability in add_probabilities.items()}

    @staticmethod
    def _merge_independent_success(existing: float, additional: float) -> float:
        return _clamp_probability(1.0 - (1.0 - existing) * (1.0 - additional))

    @staticmethod
    def _build_achievers(
        action_models: Sequence[RelaxedActionModel],
    ) -> Dict[Fact, List[RelaxedActionModel]]:
        achievers: Dict[Fact, List[RelaxedActionModel]] = {}
        for action_model in action_models:
            for fact in action_model.add_probabilities:
                achievers.setdefault(fact, []).append(action_model)
        return achievers

    @staticmethod
    def _compute_action_support(
        probabilities: Mapping[Fact, float],
        action_models: Sequence[RelaxedActionModel],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        support: Dict[str, float] = {}
        details: Dict[str, Dict[str, Any]] = {}
        for action_model in action_models:
            result = _compute_precondition_support_result(
                action_model.precondition_structure,
                probabilities,
                strict=True,
            )
            support[action_model.name] = result.probability
            details[action_model.name] = {
                "precondition_structure": result.structure,
                "support": result.probability,
                "mode": result.mode,
                "warning": result.warning,
            }
        return support, details

    @staticmethod
    def _advance_probabilities(
        probabilities: Mapping[Fact, float],
        action_support: Mapping[str, float],
        achievers: Mapping[Fact, Sequence[RelaxedActionModel]],
    ) -> Tuple[Dict[Fact, float], Dict[Fact, float]]:
        hazards: Dict[Fact, float] = {}
        next_probabilities: Dict[Fact, float] = {}

        for fact, current_probability in probabilities.items():
            failure_probability = 1.0
            for achiever in achievers.get(fact, []):
                success_probability = _clamp_probability(
                    achiever.add_probabilities[fact] * action_support.get(achiever.name, 0.0)
                )
                failure_probability *= 1.0 - success_probability

            hazard = _clamp_probability(1.0 - failure_probability)
            next_probability = _clamp_probability(
                current_probability + (1.0 - current_probability) * hazard
            )
            if next_probability < current_probability:
                next_probability = current_probability

            hazards[fact] = hazard
            next_probabilities[fact] = next_probability

        return hazards, next_probabilities

    @staticmethod
    def _detect_acyclic_dependency_graph(action_models: Sequence[RelaxedActionModel]) -> bool:
        adjacency: Dict[Tuple[str, object], Set[Tuple[str, object]]] = {}
        indegree: Dict[Tuple[str, object], int] = {}

        def ensure(node: Tuple[str, object]) -> None:
            adjacency.setdefault(node, set())
            indegree.setdefault(node, 0)

        for action_model in action_models:
            action_node = ("action", action_model.name)
            ensure(action_node)
            for fact in action_model.preconditions:
                fact_node = ("fact", fact)
                ensure(fact_node)
                if action_node not in adjacency[fact_node]:
                    adjacency[fact_node].add(action_node)
                    indegree[action_node] += 1
            for fact in action_model.add_probabilities:
                fact_node = ("fact", fact)
                ensure(fact_node)
                if fact_node not in adjacency[action_node]:
                    adjacency[action_node].add(fact_node)
                    indegree[fact_node] += 1

        queue = [node for node, degree in indegree.items() if degree == 0]
        visited = 0

        while queue:
            node = queue.pop()
            visited += 1
            for neighbor in adjacency[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        return visited == len(indegree)


# Known limitations of this heuristic:
# - ignores delete effects
# - assumes independence between action outcomes and goal facts
# - is not an exact Bellman backup and not an exact MDP probability
# - is optimistic under retries
# - shared causes can create double counting


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class DemoAction:
        name: str
        pos_preconditions: frozenset[str]
        add_effects: frozenset[str] = frozenset()
        probabilistic_add_effects: Mapping[str, float] = field(default_factory=dict)

    actions = [
        DemoAction("a_to_ab", frozenset({"A"}), probabilistic_add_effects={"AB": 0.3}),
        DemoAction("b_to_ab", frozenset({"B"}), probabilistic_add_effects={"AB": 0.6}),
        DemoAction("c_to_c1", frozenset({"C"}), probabilistic_add_effects={"C1": 0.9}),
        DemoAction("ab_to_d", frozenset({"AB"}), probabilistic_add_effects={"D": 0.1}),
        DemoAction("c1_to_d", frozenset({"C1"}), probabilistic_add_effects={"D": 0.3}),
    ]

    heuristic = ProbabilisticOptimisticRPGHeuristic(actions, facts={"A", "B", "C", "AB", "C1", "D"})
    score, result = heuristic.heuristic_score(
        {"A", "B", "C"},
        {"D"},
        aggregation="product",
        max_layers=3,
        debug=True,
    )

    print("Toy worked example:")
    for trace in result.traces:
        print(f"layer={trace.layer}")
        print(f"  P={trace.fact_probabilities}")
        if trace.action_support:
            print(f"  R={trace.action_support}")
            print(f"  R-details={trace.action_support_details}")
            print(f"  H={trace.fact_hazards}")
            print(f"  max_delta={trace.max_delta:.6f}")
    print(f"Goal score for D: {score:.6f}")
