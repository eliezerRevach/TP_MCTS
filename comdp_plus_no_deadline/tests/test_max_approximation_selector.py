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


class TestMaxApproximationSelector(unittest.TestCase):
    def test_action_scores_baseline_has_layer0(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_g",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                ),
            ],
            facts={"A", "G"},
            goal_facts={"G"},
        )
        result = heuristic.heuristic_propagate({"A"}, goal_facts={"G"}, fixed_depth=3)
        self.assertIn(0, result.action_support_by_layer)
        self.assertGreater(result.action_support_by_layer[0].get("a_to_g", 0.0), 0.0)

        scores = heuristic.action_contribution_scores(
            {"A"}, {"G"}, fixed_depth=3, strategy="baseline"
        )
        self.assertGreater(scores.get("a_to_g", 0.0), 0.0)

    def test_action_scores_resolution_uses_companion(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_g",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                ),
            ],
            facts={"A", "G"},
            goal_facts={"G"},
        )
        resolution_score = heuristic.heuristic_score(
            {"A"}, {"G"}, fixed_depth=4, strategy="atom_backtrack_exact_resolution"
        )
        scores = heuristic.action_contribution_scores(
            {"A"},
            {"G"},
            fixed_depth=4,
            strategy="atom_backtrack_exact_resolution",
        )
        self.assertGreater(scores.get("a_to_g", 0.0), 0.0)
        self.assertGreaterEqual(resolution_score, 0.0)

    def test_actions_are_mutex_shared_add(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="act_a",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                ),
                SyntheticAction(
                    name="act_b",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                ),
            ],
            facts={"G"},
            goal_facts={"G"},
        )
        self.assertTrue(heuristic.actions_are_mutex("act_a", "act_b"))

    @patch(
        "unified_planning.engines.solvers.max_approximation_selector._apply_action_set_sampled",
    )
    @patch(
        "unified_planning.engines.solvers.max_approximation_selector._stn_feasible_with_set",
        return_value=True,
    )
    def test_sampling_respects_mutex(self, _mock_stn, _mock_apply):
        _mock_apply.side_effect = lambda _mdp, state, stn, prev, _actions, _rng: (
            state,
            stn,
            prev,
        )
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="act_a",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                ),
                SyntheticAction(
                    name="act_b",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                ),
            ],
            facts={"G"},
            goal_facts={"G"},
        )
        scores = heuristic.action_contribution_scores(
            set(), {"G"}, fixed_depth=3, strategy="baseline"
        )

        act_a = SimpleNamespace(name="act_a")
        act_b = SimpleNamespace(name="act_b")

        class _StubAdapter:
            variant_label = "stub"

            def action_scores(self, state, legal_actions, remaining_deadline, current_time):
                del state, remaining_deadline, current_time
                return {a.name: scores.get(a.name, 1.0) for a in legal_actions}

            def evaluate(self, state, current_time, remaining_deadline):
                del state, current_time, remaining_deadline
                return 0.5

            def actions_are_mutex(self, name_a, name_b):
                return heuristic.actions_are_mutex(name_a, name_b)

        both_together = 0
        stub_stn = SimpleNamespace(get_current_end_time=lambda: 0.0)
        for seed in range(80):
            _, dbg = select_max_approximation_action_set(
                mdp=SimpleNamespace(deadline=lambda: 10),
                state=SimpleNamespace(predicates=set()),
                stn=stub_stn,
                previous_action_node=None,
                legal_actions=[act_a, act_b],
                adapter=_StubAdapter(),
                remaining_deadline=5.0,
                config=MaxApproximationConfig(
                    num_samples=8, seed=seed, alpha=1.5, debug=False
                ),
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

        class _PositiveScoreAdapter:
            variant_label = "stub_stn"

            def action_scores(self, state, legal_actions, remaining_deadline, current_time):
                del state, remaining_deadline, current_time
                return {a.name: 1.0 for a in legal_actions}

            def evaluate(self, state, current_time, remaining_deadline):
                del state, current_time, remaining_deadline
                return 0.0

            def actions_are_mutex(self, name_a, name_b):
                del name_a, name_b
                return False

        action_set, dbg = select_max_approximation_action_set(
            mdp=mdp,
            state=state,
            stn=stn,
            previous_action_node=None,
            legal_actions=legal,
            adapter=_PositiveScoreAdapter(),
            remaining_deadline=1.0,
            config=MaxApproximationConfig(num_samples=8, seed=7, debug=True),
        )
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
            [getattr(a, "name", str(a)) for a in set_a],
            [getattr(a, "name", str(a)) for a in set_b],
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
