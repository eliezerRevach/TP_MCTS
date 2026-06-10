"""
Tests for the horizon-indexed Pattern Database (PDB) correction prototype.

Includes the four required acceptance tests:
  1. deterministic robot chain
  2. ignored precondition (optimistic projection)
  3. stochastic robot chain
  4. door-chain pattern growth
plus adapter, pattern-growth-policy and manager/fallback checks.

Actions are built with the same duck-typed interface the rest of the codebase
uses (``pos_preconditions`` / ``add_effects`` / ``del_effects`` /
``probabilistic_effects``), so the adapter (`build_pdb_actions`) is exercised
end-to-end.
"""

import unittest
from dataclasses import dataclass, field
from typing import Mapping

from comdp_plus_no_deadline.engines.pdb_correction import (
    PatternDatabase,
    PDBAction,
    PDBCorrection,
    PDBOutcome,
    build_pdb_actions,
    generate_patterns,
    grow_pattern,
)  # noqa: F401  (PDBOutcome used by concurrency tests)


@dataclass(frozen=True)
class SyntheticAction:
    name: str
    pos_preconditions: frozenset
    add_effects: frozenset = field(default_factory=frozenset)
    del_effects: frozenset = field(default_factory=frozenset)
    duration_steps: int = 1
    probabilistic_effects: tuple = ()

    def duration_int(self) -> int:
        return self.duration_steps


@dataclass(frozen=True)
class SyntheticProbabilisticEffect:
    outcomes: Mapping[float, Mapping[object, object]]

    def probability_function(self, state, env):
        del state, env
        return self.outcomes


def _robot_chain(n=10, *, with_battery=False):
    """move_i: pre {at_i (, battery_high)}, add at_{i+1}, duration 1, p=1."""
    actions = []
    for i in range(n):
        pre = {f"at_{i}"}
        if with_battery:
            pre.add("battery_high")
        actions.append(
            SyntheticAction(
                name=f"move_{i}",
                pos_preconditions=frozenset(pre),
                add_effects=frozenset({f"at_{i + 1}"}),
            )
        )
    return actions


class TestDeterministicRobot(unittest.TestCase):
    """Test 1: deterministic robot."""

    def setUp(self):
        actions = build_pdb_actions(_robot_chain(10))
        pattern = frozenset(f"at_{i}" for i in range(11))
        self.pdb = PatternDatabase(pattern, actions)
        self.goal = {"at_10"}

    def V(self, start, H):
        return self.pdb.value({start}, H, target=self.goal)

    def test_values(self):
        self.assertAlmostEqual(self.V("at_0", 9), 0.0)
        self.assertAlmostEqual(self.V("at_0", 10), 1.0)
        self.assertAlmostEqual(self.V("at_0", 15), 1.0)
        self.assertAlmostEqual(self.V("at_4", 5), 0.0)
        self.assertAlmostEqual(self.V("at_4", 6), 1.0)


class TestIgnoredPrecondition(unittest.TestCase):
    """Test 2: ignored precondition (optimistic projection)."""

    def test_pattern_without_battery_is_optimistic(self):
        actions = build_pdb_actions(_robot_chain(10, with_battery=True))
        pattern = frozenset(f"at_{i}" for i in range(11))  # battery_high NOT in P
        pdb = PatternDatabase(pattern, actions)
        # battery_high is projected away -> abstraction ignores it -> reachable.
        self.assertAlmostEqual(pdb.value({"at_0"}, 10, target={"at_10"}), 1.0)

    def test_pattern_with_battery_blocks(self):
        actions = build_pdb_actions(_robot_chain(10, with_battery=True))
        pattern = frozenset({f"at_{i}" for i in range(11)} | {"battery_high"})
        pdb = PatternDatabase(pattern, actions)
        # battery_high is false in the real state and has no achiever -> stuck.
        self.assertAlmostEqual(pdb.value({"at_0"}, 10, target={"at_10"}), 0.0)


class TestStochasticRobot(unittest.TestCase):
    """Test 3: stochastic robot (0.8 advance, 0.2 stay)."""

    def setUp(self):
        actions = []
        for i in range(2):
            actions.append(
                SyntheticAction(
                    name=f"move_{i}",
                    pos_preconditions=frozenset({f"at_{i}"}),
                    add_effects=frozenset(),
                    probabilistic_effects=(
                        SyntheticProbabilisticEffect(
                            outcomes={
                                0.8: {f"at_{i + 1}": True},
                                0.2: {f"at_{i}": True},
                            }
                        ),
                    ),
                )
            )
        pattern = frozenset({"at_0", "at_1", "at_2"})
        self.pdb = PatternDatabase(pattern, build_pdb_actions(actions))
        self.goal = {"at_2"}

    def V(self, H):
        return self.pdb.value({"at_0"}, H, target=self.goal)

    def test_values(self):
        self.assertAlmostEqual(self.V(0), 0.0)
        self.assertAlmostEqual(self.V(1), 0.0)
        self.assertAlmostEqual(self.V(2), 0.64)
        self.assertAlmostEqual(self.V(3), 0.896)


class TestDoorChainPatternGrowth(unittest.TestCase):
    """Test 4: door-chain pattern growth and per-pattern V values."""

    def setUp(self):
        self.raw_actions = [
            SyntheticAction("go_out", frozenset({"door_open"}), frozenset({"outside"})),
            SyntheticAction("open_door", frozenset({"has_key"}), frozenset({"door_open"})),
            SyntheticAction("find_key", frozenset(), frozenset({"has_key"})),
        ]
        self.actions = build_pdb_actions(self.raw_actions)
        self.goal = {"outside"}

    def test_pattern_outside_only(self):
        pdb = PatternDatabase(frozenset({"outside"}), self.actions)
        # door_open ignored -> go_out applicable immediately (optimistic).
        self.assertAlmostEqual(pdb.value(set(), 1, target=self.goal), 1.0)

    def test_pattern_outside_door(self):
        pdb = PatternDatabase(frozenset({"outside", "door_open"}), self.actions)
        self.assertAlmostEqual(pdb.value(set(), 1, target=self.goal), 0.0)
        self.assertAlmostEqual(pdb.value(set(), 2, target=self.goal), 1.0)

    def test_pattern_outside_door_key(self):
        pdb = PatternDatabase(
            frozenset({"outside", "door_open", "has_key"}), self.actions
        )
        self.assertAlmostEqual(pdb.value(set(), 2, target=self.goal), 0.0)
        self.assertAlmostEqual(pdb.value(set(), 3, target=self.goal), 1.0)

    def test_growth_builds_full_chain(self):
        # Goal-directed growth should walk outside -> door_open -> has_key.
        pattern = grow_pattern(
            self.goal,
            self.actions,
            max_facts_per_pattern=3,
            expansion_policy="max_prob",
        )
        self.assertEqual(pattern, frozenset({"outside", "door_open", "has_key"}))

    def test_growth_respects_max_facts(self):
        pattern = grow_pattern(
            self.goal,
            self.actions,
            max_facts_per_pattern=2,
            expansion_policy="max_prob",
        )
        self.assertEqual(pattern, frozenset({"outside", "door_open"}))


class TestConcurrentDurations(unittest.TestCase):
    """Concurrent durative semantics: independent durative actions overlap
    instead of serializing their durations."""

    @staticmethod
    def _two_independent(dur1, dur2, p1=0.8, p2=0.8):
        a1 = PDBAction(
            "A1", frozenset(),
            (PDBOutcome(p1, frozenset({"f1"}), frozenset()),
             PDBOutcome(1 - p1, frozenset(), frozenset())),
            duration=dur1,
        )
        a2 = PDBAction(
            "A2", frozenset(),
            (PDBOutcome(p2, frozenset({"f2"}), frozenset()),
             PDBOutcome(1 - p2, frozenset(), frozenset())),
            duration=dur2,
        )
        return PatternDatabase(frozenset({"f1", "f2"}), [a1, a2])

    def test_equal_durations_single_attempt(self):
        # A1, A2 both dur 2, single attempt each -> exactly succ(A1)*succ(A2).
        # The OLD sequential DP returned 0 here (2+2 > 2); concurrency fixes it.
        pdb = self._two_independent(2, 2, 0.8, 0.8)
        self.assertAlmostEqual(pdb.value(set(), 1.9, target={"f1", "f2"}), 0.0)
        self.assertAlmostEqual(pdb.value(set(), 2, target={"f1", "f2"}), 0.64)

    def test_fractional_durations_overlap(self):
        # The brief's example: A1 dur 0.5 -> f1, A2 dur 2 -> f2, both at root.
        # Old code returned 0 at H=2 (0.5 + 2 = 2.5 > 2); concurrency returns > 0.
        pdb = self._two_independent(0.5, 2, 0.8, 0.8)
        # Before A2 can finish (it needs 2.0) the target is unreachable.
        self.assertAlmostEqual(pdb.value(set(), 1.5, target={"f1", "f2"}), 0.0)
        v = pdb.value(set(), 2, target={"f1", "f2"})
        self.assertGreater(v, 0.0)                       # the bug fix
        self.assertGreaterEqual(v + 1e-9, 0.64)          # >= succ(A1)*succ(A2)
        # A1 (dur 0.5) gets up to 4 attempts by t=2, A2 one: 0.8*(1-0.2**4).
        self.assertAlmostEqual(v, 0.8 * (1 - 0.2 ** 4))

    def test_independent_not_serialized_vs_old(self):
        # Regression-direction: value must not DECREASE vs the old sequential
        # model. Old returned 0 at t=2; new returns > 0.
        pdb = self._two_independent(2, 2)
        self.assertGreater(pdb.value(set(), 2, target={"f1", "f2"}), 0.0)


def _sequential_value(pdb, state, H, target):
    """Reference: the OLD sequential DP (one action per recursion, charging its
    full duration). Used to assert the concurrent DP never returns LESS."""
    pattern = pdb.pattern
    actions = pdb.projected_actions
    target_proj = frozenset(target) & pattern
    memo = {}

    def rec(x, h):
        if target_proj <= x:
            return 1.0
        if h <= 0:
            return 0.0
        k = (x, h)
        if k in memo:
            return memo[k]
        best = 0.0
        for a in actions:
            if not a.preconditions <= x:
                continue
            dur = max(1, int(a.duration))
            if dur > h:
                continue
            v = 0.0
            for o in a.outcomes:
                v += o.probability * rec((x | o.add) - o.delete, h - dur)
            best = max(best, v)
        memo[k] = best
        return best

    return rec(frozenset(state) & pattern, int(H))


class TestNoValueDecrease(unittest.TestCase):
    """Concurrency only ADDS parallelism/retries, so the new DP must never score
    below the old sequential DP (add-only instances, integer durations)."""

    def test_random_instances_no_decrease(self):
        import random as _random

        rng = _random.Random(7)
        facts = ["p", "q", "r", "s"]
        for _ in range(60):
            n_actions = rng.randint(1, 4)
            actions = []
            for i in range(n_actions):
                pre = frozenset(rng.sample(facts, rng.randint(0, 2)))
                add = frozenset(rng.sample(facts, rng.randint(1, 2)))
                p = rng.choice([0.5, 0.8, 1.0])
                outcomes = (PDBOutcome(p, add, frozenset()),)
                if p < 1.0:
                    outcomes = outcomes + (PDBOutcome(1 - p, frozenset(), frozenset()),)
                actions.append(PDBAction(f"a{i}", pre, outcomes, duration=rng.randint(1, 3)))
            pattern = frozenset(facts)
            pdb = PatternDatabase(pattern, actions)
            target = frozenset(rng.sample(facts, rng.randint(1, 2)))
            for H in range(0, 7):
                new_v = pdb.value(set(), H, target=target)
                old_v = _sequential_value(pdb, set(), H, target)
                self.assertGreaterEqual(
                    new_v + 1e-9, old_v,
                    msg=f"decrease: H={H} new={new_v} old={old_v} target={target}",
                )


class TestAdapterJointOutcomes(unittest.TestCase):
    def test_residual_no_op_outcome(self):
        # Only the 0.8 branch is listed; the adapter must add a 0.2 no-op.
        action = SyntheticAction(
            name="flip",
            pos_preconditions=frozenset({"p"}),
            probabilistic_effects=(
                SyntheticProbabilisticEffect(outcomes={0.8: {"q": True}}),
            ),
        )
        (pdb_action,) = build_pdb_actions([action])
        total = sum(o.probability for o in pdb_action.outcomes)
        self.assertAlmostEqual(total, 1.0)
        self.assertAlmostEqual(pdb_action.achievement_probability("q"), 0.8)

    def test_deterministic_single_outcome(self):
        action = SyntheticAction("a", frozenset({"x"}), frozenset({"y"}))
        (pdb_action,) = build_pdb_actions([action])
        self.assertEqual(len(pdb_action.outcomes), 1)
        self.assertEqual(pdb_action.outcomes[0].add, frozenset({"y"}))
        self.assertAlmostEqual(pdb_action.outcomes[0].probability, 1.0)


class TestPDBCorrectionManager(unittest.TestCase):
    def setUp(self):
        # b needs a AND c jointly. Independent product would over/under-count;
        # the PDB gives the exact joint reachability of {a, c}.
        self.raw_actions = [
            SyntheticAction("make_a", frozenset({"seed"}), frozenset({"a"})),
            SyntheticAction("make_c", frozenset({"seed"}), frozenset({"c"})),
            SyntheticAction("use_ac", frozenset({"a", "c"}), frozenset({"goal"})),
        ]
        self.actions = build_pdb_actions(self.raw_actions)

    def test_covering_pattern_used(self):
        pattern = frozenset({"a", "c"})
        corr = PDBCorrection([pattern], self.actions)
        called = {"n": 0}

        def fallback():
            called["n"] += 1
            return 0.123

        # Concurrent durative semantics: make_a and make_c (independent, both
        # pre {seed}) start together at t=0 and both finish at t=1, so {a, c} is
        # jointly reachable within horizon 1 (the old sequential DP needed 2).
        self.assertAlmostEqual(corr.applicability({"seed"}, {"a", "c"}, 1, fallback), 1.0)
        self.assertEqual(corr.pdb_used, 1)
        self.assertEqual(corr.fallbacks, 0)
        self.assertEqual(called["n"], 0)

    def test_uncovered_preconditions_fallback(self):
        # Pattern does not contain precondition 'c' -> not a covering pattern.
        corr = PDBCorrection([frozenset({"a"})], self.actions)

        def fallback():
            return 0.42

        val = corr.applicability({"seed"}, {"a", "c"}, 5, fallback)
        self.assertAlmostEqual(val, 0.42)
        self.assertEqual(corr.pdb_used, 0)
        self.assertEqual(corr.fallbacks, 1)

    def test_horizon_too_short(self):
        corr = PDBCorrection([frozenset({"a", "c"})], self.actions)
        # Horizon 0: neither a nor c true yet -> 0.
        val = corr.applicability({"seed"}, {"a", "c"}, 0, lambda: 1.0)
        self.assertAlmostEqual(val, 0.0)

    def test_stats_and_from_actions(self):
        corr = PDBCorrection.from_actions(
            self.raw_actions,
            goal_facts={"goal"},
            num_patterns=2,
            max_facts_per_pattern=3,
            expansion_policy="max_prob",
        )
        stats = corr.stats()
        self.assertIn("num_patterns", stats)
        self.assertIn("pdb_table_total", stats)
        self.assertGreaterEqual(stats["num_patterns"], 1)


class TestHeuristicIntegration(unittest.TestCase):
    """The ``baseline_pdb`` strategy wired into the temporal heuristic."""

    def _heuristic(self):
        from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
            TemporalProbabilisticRPGHeuristic,
        )

        raw_actions = [
            SyntheticAction("make_a", frozenset({"seed"}), frozenset({"a"})),
            SyntheticAction("make_c", frozenset({"seed"}), frozenset({"c"})),
            SyntheticAction("use_ac", frozenset({"a", "c"}), frozenset({"goal"})),
        ]
        return TemporalProbabilisticRPGHeuristic(
            actions=raw_actions,
            facts={"seed", "a", "c", "goal"},
            initial_facts={"seed"},
            goal_facts={"goal"},
        )

    def test_autobuild_disabled_matches_baseline(self):
        heuristic = self._heuristic()
        # Opt out of the lazy auto-build -> baseline_pdb degrades to baseline.
        heuristic._PDB_AUTOBUILD = False
        pdb_score = heuristic.heuristic_score(
            {"seed"}, {"goal"}, fixed_depth=5, strategy="baseline_pdb"
        )
        base_score = heuristic.heuristic_score(
            {"seed"}, {"goal"}, fixed_depth=5, strategy="baseline"
        )
        self.assertAlmostEqual(pdb_score, base_score)
        self.assertIsNone(heuristic._pdb_correction)

    def test_autobuild_uses_pdb(self):
        heuristic = self._heuristic()
        # Default behaviour: selecting baseline_pdb lazily builds + uses a PDB,
        # no explicit build_pdb_correction() call required.
        score = heuristic.heuristic_score(
            {"seed"}, {"goal"}, fixed_depth=6, strategy="baseline_pdb"
        )
        self.assertGreater(score, 0.0)
        self.assertIsNotNone(heuristic._pdb_correction)
        self.assertGreater(heuristic._pdb_correction.stats()["pdb_used"], 0)

    def test_explicit_build_takes_precedence(self):
        heuristic = self._heuristic()
        correction = heuristic.build_pdb_correction(
            {"goal"}, num_patterns=2, max_facts_per_pattern=4, expansion_policy="max_prob"
        )
        score = heuristic.heuristic_score(
            {"seed"}, {"goal"}, fixed_depth=6, strategy="baseline_pdb"
        )
        # goal is reachable (seed -> a, c -> goal) within the horizon.
        self.assertGreater(score, 0.0)
        # The explicitly attached correction is the one consulted.
        self.assertIs(heuristic._pdb_correction, correction)
        self.assertGreater(correction.stats()["pdb_used"], 0)

    def test_class_level_pdb_config_is_read(self):
        # run_domain.py sets these class attrs from the CLI; confirm they flow
        # into the lazily auto-built correction.
        heuristic = self._heuristic()
        heuristic._PDB_NUM_PATTERNS = 1
        heuristic._PDB_MAX_FACTS_PER_PATTERN = 2
        heuristic._PDB_EXPANSION_POLICY = "max_prob"
        heuristic.heuristic_score(
            {"seed"}, {"goal"}, fixed_depth=6, strategy="baseline_pdb"
        )
        stats = heuristic._pdb_correction.stats()
        self.assertLessEqual(stats["num_patterns"], 1)
        self.assertTrue(all(s <= 2 for s in stats["pattern_sizes"]))


if __name__ == "__main__":
    unittest.main()
