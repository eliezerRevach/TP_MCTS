import unittest
from dataclasses import dataclass
from typing import Mapping

import unified_planning as up
from unified_planning.shortcuts import BoolType, OverallPreconditionTiming

from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
    build_resolution_delta_schedule,
    _raw_delta_steps_for_resolution_depth,
    _resolution_anchors_ascending,
    _resolution_completion_times,
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

    def test_resolution_raw_deltas_and_reorganized_anchors_depth_25(self):
        self.assertEqual(
            _raw_delta_steps_for_resolution_depth(25),
            [1, 1, 2, 2, 4, 4, 8, 3],
        )
        self.assertEqual(
            _resolution_anchors_ascending(25),
            [0, 1, 2, 4, 6, 9, 13, 17, 25],
        )

    def test_resolution_completion_times_depth_5(self):
        anchors = _resolution_anchors_ascending(5)
        self.assertEqual(anchors, [0, 1, 2, 3, 5])
        self.assertEqual(_resolution_completion_times(anchors, 1, 5), [1, 2, 3, 5])

    def test_build_resolution_forced_minimum_with_reference_t(self):
        deltas = build_resolution_delta_schedule(
            10,
            alpha=2.0,
            k_target=8,
            t_ref=25,
            delta_min=1,
            forced_minimum=True,
        )
        self.assertEqual(len(deltas), 8)
        self.assertEqual(sum(deltas), 10)
        self.assertTrue(all(d >= 1 for d in deltas))

    def test_build_resolution_custom_alpha_changes_widths(self):
        d_default = build_resolution_delta_schedule(25, alpha=2.0, k_target=8)
        d_alpha3 = build_resolution_delta_schedule(25, alpha=3.0, k_target=8)
        self.assertEqual(sum(d_default), 25)
        self.assertEqual(sum(d_alpha3), 25)
        self.assertNotEqual(d_default, d_alpha3)

    def test_resolution_alpha_none_same_as_two(self):
        d_none = build_resolution_delta_schedule(25, alpha=None, forced_minimum=False)
        d_two = build_resolution_delta_schedule(25, alpha=2.0, forced_minimum=False)
        self.assertEqual(d_none, d_two)

    def test_legacy_resolution_ignores_k_target_cap(self):
        """forced_minimum=False must not cap layer count at k_target."""
        wide = build_resolution_delta_schedule(25, k_target=3, forced_minimum=False)
        self.assertEqual(wide, [1, 1, 2, 2, 4, 4, 8, 3])

    def test_resolution_propagate_cache_distinguishes_alpha(self):
        p_add = 0.5
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
        state: set = set()
        goals = {"DOOR"}
        depth = 4
        r_a2 = heuristic.heuristic_propagate(
            state,
            goal_facts=goals,
            fixed_depth=depth,
            strategy="atom_backtrack_exact_resolution",
            resolution_alpha=2.0,
        )
        r_a3 = heuristic.heuristic_propagate(
            state,
            goal_facts=goals,
            fixed_depth=depth,
            strategy="atom_backtrack_exact_resolution",
            resolution_alpha=3.0,
        )
        self.assertFalse(r_a2.cache_hit)
        self.assertFalse(r_a3.cache_hit)
        r_a2_again = heuristic.heuristic_propagate(
            state,
            goal_facts=goals,
            fixed_depth=depth,
            strategy="atom_backtrack_exact_resolution",
            resolution_alpha=2.0,
        )
        self.assertTrue(r_a2_again.cache_hit)

    def test_normalize_strategy_atom_backtrack_exact_resolution(self):
        self.assertEqual(
            TemporalProbabilisticRPGHeuristic._normalize_strategy(
                "atom_backtrack_exact_resolution"
            ),
            "atom_backtrack_exact_resolution",
        )

    def test_atom_backtrack_exact_resolution_lower_than_exact_single_atom(self):
        """Fewer anchor completion trials than unit-step exact → lower success probability."""
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
        depth = 25
        score_exact = heuristic.heuristic_score(
            set(),
            {"DOOR"},
            fixed_depth=depth,
            strategy="atom_backtrack_exact",
        )
        score_res = heuristic.heuristic_score(
            set(),
            {"DOOR"},
            fixed_depth=depth,
            strategy="atom_backtrack_exact_resolution",
        )
        anchors = _resolution_anchors_ascending(depth)
        trials = len(_resolution_completion_times(anchors, 1, depth))
        expected_res = 1.0 - (1.0 - p_add) ** trials
        self.assertAlmostEqual(score_res, expected_res)
        self.assertLess(score_res, score_exact)

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

    def test_fast_atom_cache_matches_atom_backtrack_cached(self):
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
        score_fast = heuristic.heuristic_score(
            set(),
            {"DOOR"},
            fixed_depth=depth,
            strategy="fast_atom_cache",
        )
        score_cached = heuristic.heuristic_score(
            set(),
            {"DOOR"},
            fixed_depth=depth,
            strategy="atom_backtrack_cached",
        )
        self.assertAlmostEqual(score_fast, score_cached, places=10)

    def test_fast_atom_cache_cross_call_submemo_hits(self):
        """Depth 7 schedule revisits (KEY,h)/(DOOR,h) pairs already filled at depth 5."""
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
        state = set()
        goals = {"DOOR"}
        r5 = heuristic.heuristic_propagate(
            state,
            goal_facts=goals,
            fixed_depth=5,
            strategy="fast_atom_cache",
        )
        self.assertFalse(r5.cache_hit)
        r7 = heuristic.heuristic_propagate(
            state,
            goal_facts=goals,
            fixed_depth=7,
            strategy="fast_atom_cache",
        )
        self.assertFalse(r7.cache_hit)
        self.assertGreater(r7.fact_cache_hits + r7.action_cache_hits, 0)

    def test_query_cache_includes_goal_facts_for_atom_strategies(self):
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
        actions = [
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
        ]
        state = set()
        depth = 3
        for strat in ("atom_backtrack_exact", "atom_backtrack_exact_resolution"):
            with self.subTest(strategy=strat):
                heuristic = TemporalProbabilisticRPGHeuristic(
                    actions=actions,
                    facts={"KEY", "DOOR"},
                )
                r_door = heuristic.heuristic_propagate(
                    state,
                    goal_facts={"DOOR"},
                    fixed_depth=depth,
                    strategy=strat,
                )
                self.assertFalse(r_door.cache_hit)
                r_key = heuristic.heuristic_propagate(
                    state,
                    goal_facts={"KEY"},
                    fixed_depth=depth,
                    strategy=strat,
                )
                self.assertFalse(r_key.cache_hit)
                r_key_again = heuristic.heuristic_propagate(
                    state,
                    goal_facts={"KEY"},
                    fixed_depth=depth,
                    strategy=strat,
                )
                self.assertTrue(r_key_again.cache_hit)


class TestExpectedTime(unittest.TestCase):
    """Tests for TemporalProbabilisticRPGHeuristic.heuristic_expected_time."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _brute_force_expected_time(failure_at_t, max_t=50_000):
        """Sum failure_at_t(t) for t=0..max_t until convergence.

        The tail-sum formula E[T] = sum_{t=0}^{inf} P(T > t) includes the t=0
        term which is always 1.0 for any goal not yet in the initial state.
        """
        E_T = 0.0
        for t in range(0, max_t + 1):
            f = failure_at_t(t)
            E_T += f
            if f < 1e-14:
                break
        return E_T

    # ------------------------------------------------------------------
    # single atom (exact closed form: E[T] = 1/p)
    # ------------------------------------------------------------------

    def test_expected_time_single_atom(self):
        p_add = 0.3
        effect = SyntheticProbabilisticEffect(
            outcomes={p_add: {"DOOR": True}, 1.0 - p_add: {}}
        )
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="try_open",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset(),
                    duration_steps=1,
                    probabilistic_effects=(effect,),
                )
            ],
            facts={"DOOR"},
        )
        result = heuristic.heuristic_expected_time(set(), {"DOOR"})
        self.assertAlmostEqual(result, 1.0 / p_add, places=6)

    def test_expected_time_deterministic_atom(self):
        effect = SyntheticProbabilisticEffect(
            outcomes={1.0: {"DOOR": True}}
        )
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="open",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset(),
                    duration_steps=1,
                    probabilistic_effects=(effect,),
                )
            ],
            facts={"DOOR"},
        )
        result = heuristic.heuristic_expected_time(set(), {"DOOR"})
        self.assertAlmostEqual(result, 1.0, places=6)

    # ------------------------------------------------------------------
    # two-step chain A -> B: compare against brute-force sum
    # ------------------------------------------------------------------

    def test_expected_time_two_step_chain(self):
        p_key = 0.4
        p_open = 0.6
        get_key_effect = SyntheticProbabilisticEffect(
            outcomes={p_key: {"KEY": True}, 1.0 - p_key: {}}
        )
        open_effect = SyntheticProbabilisticEffect(
            outcomes={p_open: {"DOOR": True}, 1.0 - p_open: {}}
        )
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="get_key",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset(),
                    duration_steps=1,
                    probabilistic_effects=(get_key_effect,),
                ),
                SyntheticAction(
                    name="open_door",
                    pos_preconditions=frozenset({"KEY"}),
                    add_effects=frozenset(),
                    duration_steps=1,
                    probabilistic_effects=(open_effect,),
                ),
            ],
            facts={"KEY", "DOOR"},
        )

        # Brute-force reference: failure_door(t) = prod_{s=1}^{t} (1 - p_open*(1-(1-p_key)^(s-1)))
        # At t=0 the product is empty, so failure_door(0) = 1.0.
        def brute_failure(t):
            f = 1.0
            for s in range(1, t + 1):
                key_avail = 1.0 - (1.0 - p_key) ** (s - 1)
                f *= 1.0 - p_open * key_avail
            return f  # = 1.0 at t=0

        expected = self._brute_force_expected_time(brute_failure)
        result = heuristic.heuristic_expected_time(set(), {"DOOR"})
        self.assertAlmostEqual(result, expected, places=4)

    # ------------------------------------------------------------------
    # conjunctive goals: E[max(T_A, T_B)] for two independent atoms
    # ------------------------------------------------------------------

    def test_expected_time_conjunctive_goals(self):
        # Use p values that differ from 1-p to avoid duplicate dict keys in outcomes.
        p_a = 0.4
        p_b = 0.3
        effect_a = SyntheticProbabilisticEffect(
            outcomes={p_a: {"A": True}, 1.0 - p_a: {}}
        )
        effect_b = SyntheticProbabilisticEffect(
            outcomes={p_b: {"B": True}, 1.0 - p_b: {}}
        )
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="do_a",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset(),
                    duration_steps=1,
                    probabilistic_effects=(effect_a,),
                ),
                SyntheticAction(
                    name="do_b",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset(),
                    duration_steps=1,
                    probabilistic_effects=(effect_b,),
                ),
            ],
            facts={"A", "B"},
        )

        # Brute-force: joint_failure(t) = 1 - P(A by t)*P(B by t)
        #            = 1 - (1-(1-p_a)^t) * (1-(1-p_b)^t)
        # At t=0 both facts are unachieved, so joint_failure(0) = 1.0.
        def brute_joint_failure(t):
            p_a_by_t = 1.0 - (1.0 - p_a) ** t
            p_b_by_t = 1.0 - (1.0 - p_b) ** t
            return 1.0 - p_a_by_t * p_b_by_t  # = 1.0 at t=0

        expected = self._brute_force_expected_time(brute_joint_failure)
        result = heuristic.heuristic_expected_time(set(), {"A", "B"})
        self.assertAlmostEqual(result, expected, places=3)

    # ------------------------------------------------------------------
    # edge cases
    # ------------------------------------------------------------------

    def test_expected_time_unreachable_returns_inf(self):
        heuristic = TemporalProbabilisticRPGHeuristic(actions=[], facts={"DOOR"})
        result = heuristic.heuristic_expected_time(set(), {"DOOR"})
        self.assertEqual(result, float("inf"))

    def test_expected_time_already_achieved_returns_zero(self):
        heuristic = TemporalProbabilisticRPGHeuristic(actions=[], facts={"DOOR"})
        result = heuristic.heuristic_expected_time({"DOOR"}, {"DOOR"})
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_expected_time_empty_goals_returns_zero(self):
        heuristic = TemporalProbabilisticRPGHeuristic(actions=[], facts=set())
        result = heuristic.heuristic_expected_time(set(), set())
        self.assertAlmostEqual(result, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
