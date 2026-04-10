import unittest
from dataclasses import dataclass
from typing import Mapping

import unified_planning as up
from unified_planning.shortcuts import BoolType, OverallPreconditionTiming

from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)


@dataclass(frozen=True)
class SyntheticAction:
    name: str
    pos_preconditions: frozenset[str]
    add_effects: frozenset[object]
    duration_steps: int
    probabilistic_effects: tuple = ()

    def duration_int(self) -> int:
        return self.duration_steps


@dataclass(frozen=True)
class SyntheticProbabilisticEffect:
    outcomes: Mapping[float, Mapping[object, object]]

    def probability_function(self, state, env):
        del state, env
        return self.outcomes


class TestTemporalProbabilisticRPG(unittest.TestCase):
    def test_duration_delays_effect_arrival(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b_d2",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=2,
                )
            ],
            facts={"A", "B"},
        )

        score_depth1 = heuristic.heuristic_score({"A"}, {"B"}, fixed_depth=1)
        score_depth2 = heuristic.heuristic_score({"A"}, {"B"}, fixed_depth=2)

        self.assertAlmostEqual(score_depth1, 0.0)
        self.assertAlmostEqual(score_depth2, 1.0)

    def test_precondition_gated_activation(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="seed_to_a",
                    pos_preconditions=frozenset({"SEED"}),
                    add_effects=frozenset({"A"}),
                    duration_steps=1,
                ),
                SyntheticAction(
                    name="a_to_b",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                ),
            ],
            facts={"SEED", "A", "B"},
        )

        score_depth1 = heuristic.heuristic_score({"SEED"}, {"B"}, fixed_depth=1)
        score_depth2 = heuristic.heuristic_score({"SEED"}, {"B"}, fixed_depth=2)

        self.assertAlmostEqual(score_depth1, 0.0)
        self.assertAlmostEqual(score_depth2, 1.0)

    def test_dp_cache_reuse(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                ),
                SyntheticAction(
                    name="a_to_c",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"C"}),
                    duration_steps=1,
                ),
            ],
            facts={"A", "B", "C"},
        )

        first = heuristic.heuristic_propagate({"A"}, fixed_depth=2)
        second = heuristic.heuristic_propagate({"A"}, fixed_depth=2)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)

    def test_split_durative_end_effect_keeps_original_duration(self):
        problem = up.model.Problem("split_duration_preserved")
        done = up.model.Fluent("done", BoolType())
        ready = up.model.Fluent("ready", BoolType())
        problem.add_fluent(done, default_initial_value=False)
        problem.add_fluent(ready, default_initial_value=True)

        finish = up.model.action.DurativeAction("finish")
        finish.set_fixed_duration(4)
        finish.add_precondition(OverallPreconditionTiming(), ready, True)
        finish.add_effect(done, True)
        problem.add_action(finish)
        problem.add_goal(done)

        ground_problem = up.engines.compilers.Grounder()._compile(problem).problem
        converted_problem = up.engines.Convert_problem(ground_problem)._converted_problem
        heuristic = TemporalProbabilisticRPGHeuristic.from_problem(converted_problem)

        initial_predicates = {
            fact
            for fact, value in converted_problem.initial_values.items()
            if value.bool_constant_value()
        }
        initial_state = up.engines.State(initial_predicates)
        score_depth3 = heuristic.heuristic_score(initial_state, converted_problem.goals, fixed_depth=3)
        score_depth4 = heuristic.heuristic_score(initial_state, converted_problem.goals, fixed_depth=4)

        self.assertAlmostEqual(score_depth3, 0.0)
        self.assertAlmostEqual(score_depth4, 1.0)

    def test_strategy_baseline_matches_default(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                )
            ],
            facts={"A", "B"},
        )

        default_score = heuristic.heuristic_score({"A"}, {"B"}, fixed_depth=1)
        baseline_score = heuristic.heuristic_score(
            {"A"},
            {"B"},
            fixed_depth=1,
            strategy="baseline",
        )
        self.assertAlmostEqual(default_score, baseline_score)

    def test_baseline_cached_matches_baseline_across_state_diffs(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="seed_to_a",
                    pos_preconditions=frozenset({"SEED"}),
                    add_effects=frozenset({"A"}),
                    duration_steps=1,
                ),
                SyntheticAction(
                    name="a_to_b",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                ),
                SyntheticAction(
                    name="b_to_goal",
                    pos_preconditions=frozenset({"B"}),
                    add_effects=frozenset({"GOAL"}),
                    duration_steps=1,
                ),
            ],
            facts={"SEED", "A", "B", "GOAL"},
        )

        states = [
            {"SEED"},
            {"SEED", "A"},
            {"SEED", "A", "B"},
            {"SEED", "B"},
        ]
        cached_table = None
        for state in states:
            baseline_score = heuristic.heuristic_score(
                state,
                {"GOAL"},
                fixed_depth=4,
                strategy="baseline",
            )
            cached_score, cached_table = heuristic.heuristic_score(
                state,
                {"GOAL"},
                fixed_depth=4,
                strategy="baseline_cached",
                cached_table=cached_table,
                return_cache_table=True,
            )
            self.assertAlmostEqual(cached_score, baseline_score)

    def test_atom_half_split_eligibility_single_precondition_only(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                )
            ],
            facts={"A", "B"},
        )

        eligibility = heuristic._build_atom_eligibility()
        self.assertIn("B", eligibility)

    def test_atom_half_split_rejects_conjunctive_preconditions(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="ac_to_b",
                    pos_preconditions=frozenset({"A", "C"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                )
            ],
            facts={"A", "C", "B"},
        )

        eligibility = heuristic._build_atom_eligibility()
        self.assertNotIn("B", eligibility)

    def test_atom_half_split_rejects_delayed_effects(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b_d2",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=2,
                )
            ],
            facts={"A", "B"},
        )

        eligibility = heuristic._build_atom_eligibility()
        self.assertNotIn("B", eligibility)

    def test_atom_half_split_matches_baseline_on_simple_case(self):
        effect = SyntheticProbabilisticEffect(
            outcomes={
                0.5: {"B": True},
                0.4: {},
            }
        )
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b_prob",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset(),
                    duration_steps=1,
                    probabilistic_effects=(effect,),
                )
            ],
            facts={"A", "B"},
        )

        score_depth2 = heuristic.heuristic_score(
            {"A"},
            {"B"},
            fixed_depth=2,
            strategy="atom_half_split",
        )
        score_depth3 = heuristic.heuristic_score(
            {"A"},
            {"B"},
            fixed_depth=3,
            strategy="atom_half_split",
        )

        baseline_depth2 = heuristic.heuristic_score(
            {"A"},
            {"B"},
            fixed_depth=2,
            strategy="baseline",
        )
        baseline_depth3 = heuristic.heuristic_score(
            {"A"},
            {"B"},
            fixed_depth=3,
            strategy="baseline",
        )

        self.assertAlmostEqual(score_depth2, baseline_depth2)
        self.assertAlmostEqual(score_depth3, baseline_depth3)

    def test_atom_half_split_conjunctive_case_falls_back_to_baseline(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="ac_to_b",
                    pos_preconditions=frozenset({"A", "C"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                )
            ],
            facts={"A", "C", "B"},
        )

        baseline_score = heuristic.heuristic_score(
            {"A", "C"},
            {"B"},
            fixed_depth=1,
            strategy="baseline",
        )
        atom_split_score = heuristic.heuristic_score(
            {"A", "C"},
            {"B"},
            fixed_depth=1,
            strategy="atom_half_split",
        )

        self.assertAlmostEqual(atom_split_score, baseline_score)

    def test_atom_half_split_reuses_fact_memo_across_queries(self):
        effect = SyntheticProbabilisticEffect(
            outcomes={
                0.3: {"B": True},
                0.7: {"B": False},
            }
        )
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b_prob",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset(),
                    duration_steps=1,
                    probabilistic_effects=(effect,),
                )
            ],
            facts={"A", "B"},
        )

        first = heuristic.heuristic_propagate(
            {"A"},
            fixed_depth=3,
            strategy="atom_half_split",
        )
        second = heuristic.heuristic_propagate(
            {"A"},
            fixed_depth=5,
            strategy="atom_half_split",
        )

        self.assertFalse(first.cache_hit)
        self.assertGreaterEqual(second.fact_cache_hits, 1)

    def test_atom_backtrack_exact_matches_closed_form_single_step_atom(self):
        p_add = 0.3
        effect = SyntheticProbabilisticEffect(
            outcomes={
                p_add: {"DOOR": True},
                1.0 - p_add: {},
            }
        )
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="search_and_open",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset(),
                    duration_steps=1,
                    probabilistic_effects=(effect,),
                )
            ],
            facts={"DOOR"},
        )

        depth = 5
        score = heuristic.heuristic_score(
            set(),
            {"DOOR"},
            fixed_depth=depth,
            strategy="atom_backtrack_exact",
        )
        expected = 1.0 - (1.0 - p_add) ** depth
        self.assertAlmostEqual(score, expected)

    def test_atom_backtrack_exact_matches_closed_form_two_step_chain(self):
        p_key = 0.4
        p_open_with_key = 0.6
        get_key_effect = SyntheticProbabilisticEffect(
            outcomes={
                p_key: {"KEY": True},
                1.0 - p_key: {},
            }
        )
        open_effect = SyntheticProbabilisticEffect(
            outcomes={
                p_open_with_key: {"DOOR": True},
                1.0 - p_open_with_key: {},
            }
        )
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="search_key",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset(),
                    duration_steps=1,
                    probabilistic_effects=(get_key_effect,),
                ),
                SyntheticAction(
                    name="open_with_key",
                    pos_preconditions=frozenset({"KEY"}),
                    add_effects=frozenset(),
                    duration_steps=1,
                    probabilistic_effects=(open_effect,),
                ),
            ],
            facts={"KEY", "DOOR"},
        )

        depth = 5
        score = heuristic.heuristic_score(
            set(),
            {"DOOR"},
            fixed_depth=depth,
            strategy="atom_backtrack_exact",
        )

        expected_failure = 1.0
        for i in range(1, depth + 1):
            key_by_prev = 1.0 - (1.0 - p_key) ** (i - 1)
            step_success = p_open_with_key * key_by_prev
            expected_failure *= 1.0 - step_success
        expected = 1.0 - expected_failure
        self.assertAlmostEqual(score, expected)


if __name__ == "__main__":
    unittest.main()
