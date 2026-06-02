"""
Tests for rollout-aligned common-horizon PTRPG:

  * the MDP-agnostic wrapper (RolloutAlignedEvaluator) with injected fakes, and
  * the baseline_survival_resolution suffix strategy (v3's PTRPG).

The wrapper logic is tested without any MDP by injecting raw_eval_fn /
prefix_rollout_fn fakes, so all branches (delta==0, goal-in-prefix, suffix
eval, redo averaging, budget fallback, caching) are deterministic.
"""

import unittest
from dataclasses import dataclass

from comdp_plus_no_deadline.engines.rollout_aligned import (
    FALLBACK_HORIZON_CAPPED,
    FALLBACK_RAW,
    RolloutAlignedConfig,
    RolloutAlignedEvaluator,
    frontier_score,
)
from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)


@dataclass
class FakeState:
    name: str
    current_time: float = 0.0
    predicates: frozenset = frozenset()


def _make_evaluator(cfg, prefix_fn=None):
    raw_calls = []

    def raw_eval(state, horizon):
        raw_calls.append((getattr(state, "name", state), horizon))
        return 0.5

    def default_prefix(state, delta):
        return FakeState("after", current_time=state.current_time + delta), False, 3, False

    evaluator = RolloutAlignedEvaluator(
        config=cfg,
        raw_eval_fn=raw_eval,
        prefix_rollout_fn=prefix_fn or default_prefix,
        state_hash_fn=lambda s: (frozenset(s.predicates), s.current_time),
    )
    return evaluator, raw_calls


class TestRolloutAlignedWrapper(unittest.TestCase):
    def test_delta_zero_uses_raw_at_R(self):
        # R=10 <= H=15 -> H=10, delta=0 -> raw PTRPG at R.
        cfg = RolloutAlignedConfig(common_horizon_H=15)
        ev, calls = _make_evaluator(cfg)
        v = ev.evaluate(FakeState("n"), remaining_horizon=10)
        self.assertEqual(v, 0.5)
        self.assertEqual(calls, [("n", 10)])
        self.assertEqual(ev.diagnostics.delta_zero_evaluations, 1)
        self.assertEqual(ev.diagnostics.prefix_rollouts, 0)

    def test_goal_in_prefix_returns_one(self):
        cfg = RolloutAlignedConfig(common_horizon_H=10, use_dynamic_H=False, redo=1)

        def prefix(state, delta):
            return FakeState("g", predicates=frozenset({"G"})), True, 2, False

        ev, calls = _make_evaluator(cfg, prefix_fn=prefix)
        v = ev.evaluate(FakeState("n"), remaining_horizon=30)  # delta=20
        self.assertEqual(v, 1.0)
        self.assertEqual(ev.diagnostics.prefix_rollouts_reached_goal, 1)
        self.assertEqual(calls, [])  # suffix PTRPG skipped when goal reached

    def test_no_goal_evaluates_suffix_at_H(self):
        cfg = RolloutAlignedConfig(common_horizon_H=10, use_dynamic_H=False, redo=1)
        ev, calls = _make_evaluator(cfg)  # default prefix never reaches goal
        v = ev.evaluate(FakeState("n"), remaining_horizon=30)  # delta=20
        self.assertEqual(v, 0.5)
        self.assertEqual(calls, [("after", 10)])  # common suffix horizon H=10

    def test_redo_averaging(self):
        cfg = RolloutAlignedConfig(common_horizon_H=10, use_dynamic_H=False, redo=4)
        seq = [True, False, True, False]  # -> (1 + 0.5 + 1 + 0.5)/4 = 0.75
        idx = {"i": 0}

        def prefix(state, delta):
            reached = seq[idx["i"] % len(seq)]
            idx["i"] += 1
            preds = frozenset({"G"}) if reached else frozenset()
            return FakeState("x", predicates=preds), reached, 1, False

        ev, _ = _make_evaluator(cfg, prefix_fn=prefix)
        v = ev.evaluate(FakeState("n"), remaining_horizon=30)
        self.assertAlmostEqual(v, 0.75)
        self.assertEqual(ev.diagnostics.prefix_rollouts, 4)
        self.assertEqual(ev.diagnostics.prefix_rollouts_reached_goal, 2)

    def test_budget_fallback_horizon_capped(self):
        cfg = RolloutAlignedConfig(
            common_horizon_H=10,
            use_dynamic_H=False,
            redo=5,
            max_prefix_rollouts_per_search=2,
            fallback_mode=FALLBACK_HORIZON_CAPPED,
        )
        ev, calls = _make_evaluator(cfg)
        # First node: affordable = min(redo=5, per_search=2) = 2 rollouts.
        ev.evaluate(FakeState("n1"), remaining_horizon=30)
        self.assertEqual(ev.diagnostics.prefix_rollouts, 2)
        calls.clear()
        # Second node: per-search budget exhausted -> fallback PTRPG(state, min(R,H)).
        v = ev.evaluate(FakeState("n2"), remaining_horizon=30)
        self.assertEqual(v, 0.5)
        self.assertEqual(calls, [("n2", 10)])  # min(30, 10)
        self.assertEqual(ev.diagnostics.budget_fallbacks, 1)

    def test_budget_fallback_raw_mode(self):
        cfg = RolloutAlignedConfig(
            common_horizon_H=10,
            use_dynamic_H=False,
            redo=5,
            max_prefix_rollouts_per_search=0,  # unlimited; force via time instead
            fallback_mode=FALLBACK_RAW,
        )
        # Force time exhaustion directly.
        cfg.max_prefix_rollout_time_per_search = 1e-9
        ev, calls = _make_evaluator(cfg)
        ev._search_time = 1.0  # pretend the search already spent its time budget
        v = ev.evaluate(FakeState("n"), remaining_horizon=30)
        self.assertEqual(v, 0.5)
        self.assertEqual(calls, [("n", 30)])  # raw fallback uses full R
        self.assertEqual(ev.diagnostics.budget_fallbacks, 1)

    def test_reset_search_budget(self):
        cfg = RolloutAlignedConfig(
            common_horizon_H=10, use_dynamic_H=False, redo=5, max_prefix_rollouts_per_search=2
        )
        ev, _ = _make_evaluator(cfg)
        ev.evaluate(FakeState("n1"), remaining_horizon=30)
        self.assertEqual(ev._search_rollouts, 2)
        ev.reset_search_budget()
        self.assertEqual(ev._search_rollouts, 0)
        # After reset the budget is available again.
        ev.evaluate(FakeState("n2"), remaining_horizon=30)
        self.assertEqual(ev._search_rollouts, 2)

    def test_caching(self):
        cfg = RolloutAlignedConfig(common_horizon_H=10, use_dynamic_H=False, redo=1, cache_aligned_values=True)
        ev, calls = _make_evaluator(cfg)
        s = FakeState("n", current_time=0.0)
        ev.evaluate(s, remaining_horizon=30)
        n_rollouts = ev.diagnostics.prefix_rollouts
        ev.evaluate(s, remaining_horizon=30)  # identical -> cache hit
        self.assertEqual(ev.diagnostics.cache_hits, 1)
        self.assertEqual(ev.diagnostics.prefix_rollouts, n_rollouts)  # no new rollout

    def test_diagnostics_dict(self):
        cfg = RolloutAlignedConfig(common_horizon_H=10, use_dynamic_H=False, redo=2)
        ev, _ = _make_evaluator(cfg)
        ev.evaluate(FakeState("n"), remaining_horizon=30)
        d = ev.diagnostics.as_dict()
        self.assertEqual(d["prefix_rollouts"], 2)
        self.assertIn("avg_prefix_rollout_length", d)
        self.assertIn("avg_aligned_value", d)

    # -- dynamic parent-local horizon --------------------------------------

    def test_dynamic_h_override_used(self):
        # use_dynamic_H + h_override=10 -> H_p=10 even though fixed cap is large.
        cfg = RolloutAlignedConfig(common_horizon_H=50, use_dynamic_H=True, redo=1)
        ev, calls = _make_evaluator(cfg)
        v = ev.evaluate(FakeState("n"), remaining_horizon=30, h_override=10)
        # delta = 30 - 10 = 20 -> prefix then suffix at H_p=10.
        self.assertEqual(calls, [("after", 10)])
        self.assertEqual(ev.diagnostics.dynamic_h_evaluations, 1)
        self.assertEqual(ev.diagnostics.fixed_h_evaluations, 0)

    def test_dynamic_h_not_capped_by_fixed(self):
        # The fixed horizon (common_horizon_H = ROLLOUT_ALIGNED_H) must NOT cap the
        # dynamic H_p; H_p is used as-is (capped only at the node's own R).
        cfg = RolloutAlignedConfig(common_horizon_H=12, use_dynamic_H=True, redo=1)
        ev, calls = _make_evaluator(cfg)
        ev.evaluate(FakeState("n"), remaining_horizon=40, h_override=30)
        self.assertEqual(calls, [("after", 30)])  # H_p=30, NOT capped down to 12

    def test_dynamic_no_override_uses_R_not_fixed(self):
        # Dynamic mode with no h_override evaluates at the node's own R, never at
        # common_horizon_H -> the fixed horizon is inert in dynamic mode.
        cfg = RolloutAlignedConfig(common_horizon_H=5, use_dynamic_H=True, redo=1)
        ev, calls = _make_evaluator(cfg)
        ev.evaluate(FakeState("n"), remaining_horizon=20, h_override=None)
        self.assertEqual(calls, [("n", 20)])  # H=R=20, NOT 5

    def test_fixed_h_when_dynamic_disabled(self):
        cfg = RolloutAlignedConfig(common_horizon_H=10, use_dynamic_H=False, redo=1)
        ev, calls = _make_evaluator(cfg)
        ev.evaluate(FakeState("n"), remaining_horizon=30, h_override=5)  # override ignored
        self.assertEqual(calls, [("after", 10)])  # fixed H=10
        self.assertEqual(ev.diagnostics.fixed_h_evaluations, 1)

    def test_dead_end_returns_zero(self):
        cfg = RolloutAlignedConfig(common_horizon_H=10, use_dynamic_H=False, redo=1)

        def prefix(state, delta):
            return FakeState("dead"), False, 1, True  # dead-end

        ev, calls = _make_evaluator(cfg, prefix_fn=prefix)
        v = ev.evaluate(FakeState("n"), remaining_horizon=30)
        self.assertEqual(v, 0.0)
        self.assertEqual(ev.diagnostics.prefix_rollouts_dead_end, 1)
        self.assertEqual(calls, [])  # no suffix eval on a dead-end

    def test_min_dynamic_horizon_fixed_fallback(self):
        # H_p below the floor -> fall back to the fixed common_horizon_H.
        cfg = RolloutAlignedConfig(
            common_horizon_H=12,
            use_dynamic_H=True,
            min_dynamic_horizon=5,
            fallback_if_H_too_small="fixed",
            redo=1,
        )
        ev, calls = _make_evaluator(cfg)
        ev.evaluate(FakeState("n"), remaining_horizon=40, h_override=2)  # H_p=2 < 5
        self.assertEqual(calls, [("after", 12)])  # explicit opt-in to fixed H=12
        self.assertEqual(ev.diagnostics.small_h_fallbacks, 1)

    def test_min_dynamic_horizon_raw_fallback(self):
        cfg = RolloutAlignedConfig(
            common_horizon_H=12,
            use_dynamic_H=True,
            min_dynamic_horizon=5,
            fallback_if_H_too_small="raw",
            redo=1,
        )
        ev, calls = _make_evaluator(cfg)
        v = ev.evaluate(FakeState("n"), remaining_horizon=40, h_override=2)
        self.assertEqual(calls, [("n", 40)])  # raw PTRPG at full R, no rollout
        self.assertEqual(ev.diagnostics.prefix_rollouts, 0)


@dataclass(frozen=True)
class SyntheticAction:
    name: str
    pos_preconditions: frozenset
    add_effects: frozenset
    duration_steps: int = 1
    del_effects: frozenset = frozenset()
    probabilistic_effects: tuple = ()

    def duration_int(self) -> int:
        return self.duration_steps


class TestBaselineSurvivalResolution(unittest.TestCase):
    def test_normalized(self):
        n = TemporalProbabilisticRPGHeuristic._normalize_strategy(
            "baseline_survival_resolution"
        )
        self.assertEqual(n, "baseline_survival_resolution")

    def test_runs_and_in_range(self):
        acts = [
            SyntheticAction("mkB", frozenset({"A"}), frozenset({"B"})),
            SyntheticAction("mkG", frozenset({"B"}), frozenset({"G"})),
        ]
        h = TemporalProbabilisticRPGHeuristic(
            acts, facts={"A", "B", "G"}, initial_facts={"A"}, goal_facts={"G"}
        )
        for d in (3, 8, 20):
            score = h.heuristic_score(
                {"A"}, {"G"}, fixed_depth=d, strategy="baseline_survival_resolution"
            )
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_reachable_goal_is_positive_at_depth(self):
        # G reachable in 2 sequential steps; deep enough horizon -> P(G) > 0.
        acts = [
            SyntheticAction("mkB", frozenset({"A"}), frozenset({"B"})),
            SyntheticAction("mkG", frozenset({"B"}), frozenset({"G"})),
        ]
        h = TemporalProbabilisticRPGHeuristic(
            acts, facts={"A", "B", "G"}, initial_facts={"A"}, goal_facts={"G"}
        )
        score = h.heuristic_score(
            {"A"}, {"G"}, fixed_depth=20, strategy="baseline_survival_resolution"
        )
        self.assertGreater(score, 0.0)


class TestFrontierScore(unittest.TestCase):
    """Option A frontier-aligned selection score (frontier_aligned_*)."""

    def test_lambda_one_uses_aligned_plus_exploration(self):
        # lambda=1.0 -> ignore Q, select on aligned value (+ exploration).
        s = frontier_score(existing_score=0.2, aligned_value=0.9, lambda_align=1.0, exploration=0.1)
        self.assertAlmostEqual(s, 0.9 + 0.1)

    def test_lambda_zero_recovers_existing(self):
        # lambda=0.0 -> standard UCT exploitation (Q) + exploration.
        s = frontier_score(existing_score=0.2, aligned_value=0.9, lambda_align=0.0, exploration=0.1)
        self.assertAlmostEqual(s, 0.2 + 0.1)

    def test_lambda_half_blends(self):
        s = frontier_score(existing_score=0.2, aligned_value=0.8, lambda_align=0.5)
        self.assertAlmostEqual(s, 0.5 * 0.2 + 0.5 * 0.8)

    def test_lambda_clamped(self):
        # Out-of-range lambda is clamped to [0, 1].
        s_hi = frontier_score(0.2, 0.8, lambda_align=5.0)
        self.assertAlmostEqual(s_hi, 0.8)  # clamped to 1.0
        s_lo = frontier_score(0.2, 0.8, lambda_align=-1.0)
        self.assertAlmostEqual(s_lo, 0.2)  # clamped to 0.0

    def test_argmax_prefers_higher_aligned_at_lambda_one(self):
        # With lambda=1 and equal exploration, the higher aligned value wins.
        a = frontier_score(0.9, 0.3, 1.0, exploration=0.05)  # high Q, low aligned
        b = frontier_score(0.1, 0.7, 1.0, exploration=0.05)  # low Q, high aligned
        self.assertGreater(b, a)


class TestOptionASanity(unittest.TestCase):
    """Spec sanity (#10): align frontier nodes to the deepest elapsed.

    Frontier A: elapsed=0, remaining=25 ; B: elapsed=10, remaining=15.
    deepest_elapsed=10 -> H_frontier=15. A must be scored by rolling 10 time
    units then PTRPG(.,15); B directly by PTRPG(B,15). It must NOT compare
    PTRPG(A,25) against PTRPG(B,15).
    """

    def test_ab_alignment_to_deepest(self):
        cfg = RolloutAlignedConfig(use_dynamic_H=True, redo=1)
        raw_calls = []
        rolled = []

        def raw_eval(state, horizon):
            raw_calls.append((getattr(state, "name", state), horizon))
            return 0.5

        def prefix(state, delta):
            rolled.append((getattr(state, "name", state), delta))
            return (
                FakeState("after_" + state.name, current_time=state.current_time + delta),
                False,
                int(delta),
                False,
            )

        ev = RolloutAlignedEvaluator(
            cfg, raw_eval, prefix,
            state_hash_fn=lambda s: (frozenset(s.predicates), s.current_time),
        )
        # H_frontier = deadline - deepest_elapsed = 25 - 10 = 15 (passed as h_override).
        v_a = ev.evaluate(FakeState("A", current_time=0.0), remaining_horizon=25, h_override=15)
        v_b = ev.evaluate(FakeState("B", current_time=10.0), remaining_horizon=15, h_override=15)
        # A: delta = 25 - 15 = 10 -> one prefix rollout of 10, then PTRPG at H=15.
        self.assertEqual(rolled, [("A", 10)])
        # B: delta = 0 -> PTRPG(B, 15) directly (no rollout).
        # Both suffixes are evaluated at the COMMON horizon 15, never 25.
        self.assertEqual(raw_calls, [("after_A", 15), ("B", 15)])
        self.assertEqual(v_a, 0.5)
        self.assertEqual(v_b, 0.5)


class TestFrontierStrategyRecognition(unittest.TestCase):
    def test_strategies_normalize(self):
        for name in (
            "frontier_aligned_baseline",
            "frontier_aligned_survival",
            "frontier_aligned_resolution_survival",
        ):
            # The frontier strategies are MCTS selection modes, not heuristic
            # propagation strategies, so the heuristic itself never normalizes
            # them; they are routed in mcts.py. Here we just assert the suffix
            # PTRPG each maps to is a valid heuristic strategy.
            pass
        from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
            TemporalProbabilisticRPGHeuristic,
        )
        for suffix in ("baseline", "baseline_survival", "baseline_survival_resolution"):
            self.assertEqual(
                TemporalProbabilisticRPGHeuristic._normalize_strategy(suffix), suffix
            )


if __name__ == "__main__":
    unittest.main()
