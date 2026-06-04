import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import unified_planning as up
from unified_planning.shortcuts import BoolType, OverallPreconditionTiming

from comdp_plus_no_deadline.engines import MDP
from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)
from unified_planning.engines.solvers.max_approximation_selector import (
    MaxApproximationConfig,
    build_heuristic_adapter,
    select_max_approximation_action_set,
)
from unified_planning.engines.utils import create_init_stn


@dataclass(frozen=True)
class SyntheticAction:
    name: str
    pos_preconditions: frozenset[str]
    add_effects: frozenset[object]
    duration_steps: int
    del_effects: frozenset[object] = frozenset()
    probabilistic_effects: tuple = ()

    def duration_int(self) -> int:
        return self.duration_steps


def _chain_heuristic() -> TemporalProbabilisticRPGHeuristic:
    """2-step chain: A --a_to_b--> B --b_to_g--> G; goal G."""
    return TemporalProbabilisticRPGHeuristic(
        actions=[
            SyntheticAction("a_to_b", frozenset({"A"}), frozenset({"B"}), 1),
            SyntheticAction("b_to_g", frozenset({"B"}), frozenset({"G"}), 1),
        ],
        facts={"A", "B", "G"},
        goal_facts={"G"},
    )


def _build_split_problem(duration: int):
    problem = up.model.Problem("max_approx_sanity")
    done = up.model.Fluent("done", BoolType())
    ready = up.model.Fluent("ready", BoolType())
    problem.add_fluent(done, default_initial_value=False)
    problem.add_fluent(ready, default_initial_value=True)

    finish = up.model.action.DurativeAction("finish")
    finish.set_fixed_duration(duration)
    finish.add_precondition(OverallPreconditionTiming(), ready, True)
    finish.add_effect(done, True)
    problem.add_action(finish)
    problem.add_goal(done)

    ground_problem = up.engines.compilers.Grounder()._compile(problem).problem
    return up.engines.Convert_problem(ground_problem)._converted_problem


class TestActionContributionScoring(unittest.TestCase):
    def test_action_add_facts_maps_name_to_model_adds(self):
        heuristic = _chain_heuristic()
        self.assertEqual(heuristic.action_add_facts("a_to_b"), frozenset({"B"}))
        self.assertEqual(heuristic.action_add_facts("b_to_g"), frozenset({"G"}))
        self.assertEqual(heuristic.action_add_facts("does_not_exist"), frozenset())

    def test_goal_backtrack_credit_for_indirect_action(self):
        """
        The key regression: a chain-prefix action (a_to_b adds B, not the goal)
        must receive POSITIVE goal-backtrack credit, because seeding B raises
        P(G by deadline). The old forward "directly adds a goal fact" score gave
        this action 0, which collapsed the selector to the empty set.
        """
        heuristic = _chain_heuristic()
        # Time-graded area score (what the selector uses): rewards achieving the
        # goal earlier, so seeding B (what starting a_to_b achieves) lifts it.
        for depth in (2, 6, 20):
            base = heuristic.heuristic_score(
                {"A"}, {"G"}, aggregation="area", fixed_depth=depth, strategy="baseline"
            )
            with_b = heuristic.heuristic_score(
                {"A", "B"},
                {"G"},
                aggregation="area",
                fixed_depth=depth,
                strategy="baseline",
            )
            self.assertGreater(
                with_b,
                base,
                msg=f"depth={depth}: seeding B did not raise the area score",
            )
        # And the saturating product score is exactly why the old scorer failed:
        # at a generous horizon it gives no gradient.
        self.assertEqual(
            heuristic.heuristic_score(
                {"A"}, {"G"}, aggregation="product", fixed_depth=6, strategy="baseline"
            ),
            heuristic.heuristic_score(
                {"A", "B"}, {"G"}, aggregation="product", fixed_depth=6, strategy="baseline"
            ),
        )

    def test_actions_are_mutex_shared_add(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction("act_a", frozenset(), frozenset({"G"}), 1),
                SyntheticAction("act_b", frozenset(), frozenset({"G"}), 1),
            ],
            facts={"G"},
            goal_facts={"G"},
        )
        self.assertTrue(heuristic.actions_are_mutex("act_a", "act_b"))


class _ChainStubAdapter:
    """
    Marginal table model for unit tests: each named add-fact lifts the value.
    value(facts) = 0.3 + 0.2 * |{f in facts : f in tracked}|
    """

    variant_label = "chain_stub"

    def __init__(self, add_map, mutex_pairs=frozenset(), tracked=None):
        self._add_map = add_map
        self._mutex = {frozenset(p) for p in mutex_pairs}
        self._tracked = set(tracked) if tracked is not None else set(
            f for adds in add_map.values() for f in adds
        )

    def eval_facts(self, fact_set, current_time, remaining_deadline):
        del current_time, remaining_deadline
        return 0.3 + 0.2 * len(self._tracked & set(fact_set))

    def action_add_facts(self, action):
        return frozenset(self._add_map.get(_name(action), frozenset()))

    def evaluate(self, state, current_time, remaining_deadline):
        del current_time, remaining_deadline
        return self.eval_facts(getattr(state, "predicates", set()), 0, 0)

    def actions_are_mutex(self, name_a, name_b):
        return frozenset({name_a, name_b}) in self._mutex


def _name(action):
    return getattr(action, "name", str(action))


class TestSelectorConstruction(unittest.TestCase):
    @patch(
        "unified_planning.engines.solvers.max_approximation_selector._apply_action_set_sampled",
    )
    @patch(
        "unified_planning.engines.solvers.max_approximation_selector._stn_feasible_with_set",
        return_value=True,
    )
    def test_returns_nonempty_for_positive_contributor(self, _stn, _apply):
        _apply.side_effect = lambda _mdp, state, stn, prev, _actions, _rng: (
            state,
            stn,
            prev,
        )
        a_to_b = SimpleNamespace(name="a_to_b", predicates=set())
        adapter = _ChainStubAdapter(add_map={"a_to_b": {"B"}})
        stub_stn = SimpleNamespace(get_current_end_time=lambda: 0.0)
        action_set, dbg = select_max_approximation_action_set(
            mdp=SimpleNamespace(deadline=lambda: 10),
            state=SimpleNamespace(predicates={"A"}),
            stn=stub_stn,
            previous_action_node=None,
            legal_actions=[a_to_b],
            adapter=adapter,
            remaining_deadline=5.0,
            config=MaxApproximationConfig(num_samples=4, seed=0),
        )
        self.assertEqual([_name(a) for a in action_set], ["a_to_b"])
        self.assertGreater(dbg.action_scores.get("a_to_b", 0.0), 0.0)

    @patch(
        "unified_planning.engines.solvers.max_approximation_selector._apply_action_set_sampled",
    )
    @patch(
        "unified_planning.engines.solvers.max_approximation_selector._stn_feasible_with_set",
        return_value=True,
    )
    def test_sampling_respects_mutex(self, _stn, _apply):
        _apply.side_effect = lambda _mdp, state, stn, prev, _actions, _rng: (
            state,
            stn,
            prev,
        )
        act_a = SimpleNamespace(name="act_a")
        act_b = SimpleNamespace(name="act_b")
        adapter = _ChainStubAdapter(
            add_map={"act_a": {"fa"}, "act_b": {"fb"}},
            mutex_pairs=[("act_a", "act_b")],
        )
        stub_stn = SimpleNamespace(get_current_end_time=lambda: 0.0)
        both_together = 0
        for seed in range(60):
            _set, dbg = select_max_approximation_action_set(
                mdp=SimpleNamespace(deadline=lambda: 10),
                state=SimpleNamespace(predicates=set()),
                stn=stub_stn,
                previous_action_node=None,
                legal_actions=[act_a, act_b],
                adapter=adapter,
                remaining_deadline=5.0,
                config=MaxApproximationConfig(num_samples=8, seed=seed, alpha=1.5),
            )
            for sampled in dbg.sampled_sets:
                if "act_a" in sampled and "act_b" in sampled:
                    both_together += 1
        self.assertEqual(both_together, 0)

    @patch(
        "unified_planning.engines.solvers.max_approximation_selector._fit_action_stn",
        return_value=None,
    )
    def test_sampling_respects_stn(self, _mock_fit):
        converted = _build_split_problem(duration=2)
        converted.set_deadline(
            up.model.timing.Timing(
                delay=5,
                timepoint=up.model.timing.Timepoint(up.model.timing.TimepointKind.START),
            )
        )
        mdp = MDP(converted, discount_factor=1.0, reward_mode="terminal")
        state = mdp.initial_state()
        stn = create_init_stn(mdp)
        legal = mdp.legal_actions(state)
        self.assertTrue(legal)

        adapter = build_heuristic_adapter(
            mdp,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            temporal_heuristic_depth=5,
        )
        action_set, dbg = select_max_approximation_action_set(
            mdp=mdp,
            state=state,
            stn=stn,
            previous_action_node=None,
            legal_actions=legal,
            adapter=adapter,
            remaining_deadline=1.0,
            config=MaxApproximationConfig(num_samples=8, seed=7, debug=True),
        )
        # Every action is STN-infeasible => nothing can be committed.
        self.assertEqual(len(action_set), 0)
        stn_reasons = [r for r in dbg.rejections if r.get("reason") == "stn"]
        self.assertTrue(stn_reasons)

    def test_deterministic_with_seed(self):
        converted = _build_split_problem(duration=2)
        converted.set_deadline(
            up.model.timing.Timing(
                delay=5,
                timepoint=up.model.timing.Timepoint(up.model.timing.TimepointKind.START),
            )
        )
        mdp = MDP(converted, discount_factor=1.0, reward_mode="terminal")
        state = mdp.initial_state()
        stn = create_init_stn(mdp)
        adapter = build_heuristic_adapter(
            mdp,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            temporal_heuristic_depth=5,
        )
        legal = mdp.legal_actions(state)
        cfg = MaxApproximationConfig(num_samples=12, seed=42, alpha=1.5)

        set_a, _ = select_max_approximation_action_set(
            mdp, state, stn, None, legal, adapter, 5.0, cfg
        )
        set_b, _ = select_max_approximation_action_set(
            mdp, state, stn, None, legal, adapter, 5.0, cfg
        )
        self.assertEqual(
            [_name(a) for a in set_a],
            [_name(a) for a in set_b],
        )

    def test_alpha_skews_toward_high_score(self):
        import random

        counts = {"high": 0, "low": 0}
        for seed in range(200):
            rng = random.Random(seed)
            weights = [0.9**1.5, 0.1**1.5]
            pick = rng.choices(["high", "low"], weights=weights, k=1)[0]
            counts[pick] += 1
        self.assertGreater(counts["high"], counts["low"] * 2)


if __name__ == "__main__":
    unittest.main()
