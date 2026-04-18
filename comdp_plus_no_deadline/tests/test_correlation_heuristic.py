"""Unit tests for correlation pre-planning and DP (no unified_planning.parser import)."""
import unittest

from comdp_plus_no_deadline.engines.correlation_heuristic import (
    CorrActionSpec,
    compute_correlation_preplanning,
    run_correlation_dp,
)


class TestCorrelationHeuristic(unittest.TestCase):
    def test_preplanning_returns_tags(self):
        f_a, f_b, f_g = "a", "b", "g"
        specs = [
            CorrActionSpec(
                name="a1",
                preconditions=frozenset(),
                effect_delay_steps=1,
                joint_adds={frozenset({f_a}): 1.0},
            ),
            CorrActionSpec(
                name="bg",
                preconditions=frozenset({f_a, f_b}),
                effect_delay_steps=1,
                joint_adds={frozenset({f_g}): 1.0},
            ),
        ]
        ct, jp, ach, _ = compute_correlation_preplanning(
            specs,
            {f_g},
            {f_a, f_b, f_g},
        )
        self.assertIn(frozenset({f_a, f_b}), ct or {})
        self.assertIsInstance(jp, set)
        self.assertIn(f_g, ach)

    def test_dp_bounds_ordered(self):
        f_a, f_g = "a", "g"
        specs = [
            CorrActionSpec(
                name="ag",
                preconditions=frozenset({f_a}),
                effect_delay_steps=1,
                joint_adds={frozenset({f_g}): 1.0},
            ),
        ]
        ct, jp, ach, _ = compute_correlation_preplanning(specs, {f_g}, {f_a, f_g})
        name_to_spec = {s.name: s for s in specs}
        pess = run_correlation_dp(
            state_facts={f_a},
            goal_facts=[f_g],
            action_specs=specs,
            achievers_by_fact=ach,
            name_to_spec=name_to_spec,
            correlation_table=ct,
            joint_pairs=jp,
            deadline=5,
            pessimistic=True,
        )
        opt = run_correlation_dp(
            state_facts={f_a},
            goal_facts=[f_g],
            action_specs=specs,
            achievers_by_fact=ach,
            name_to_spec=name_to_spec,
            correlation_table=ct,
            joint_pairs=jp,
            deadline=5,
            pessimistic=False,
        )
        self.assertGreaterEqual(opt, pess)
        self.assertGreaterEqual(1.0 + 1e-9, max(pess, opt))
        self.assertGreaterEqual(max(pess, opt), -1e-9)


if __name__ == "__main__":
    unittest.main()
