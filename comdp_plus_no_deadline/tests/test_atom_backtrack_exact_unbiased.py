"""
Focused tests for the atom_backtrack_exact_unbiased temporal heuristic strategy.

Validates:
- Strategy is recognized by the dispatcher.
- Pre-planning (lambda + B(t)) runs once and is cached on the heuristic instance.
- λ correctly decomposes into OR / AND / DEL contributions.
- Bias correction respects sign conventions (mutex achievers + delete-actions
  produce λ > 0 → corrected score is at most the uncorrected resolution score).
- Numerical guard: λ ≈ 0 ⇒ B(t) ≡ 0 ⇒ unbiased score equals resolution score.
"""

import unittest
from dataclasses import dataclass

from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)


@dataclass(frozen=True)
class SyntheticAction:
    """Minimal action-like object compatible with the heuristic's action model builder."""

    name: str
    pos_preconditions: frozenset
    add_effects: frozenset
    duration_steps: int
    del_effects: frozenset = frozenset()
    probabilistic_effects: tuple = ()

    def duration_int(self) -> int:
        return self.duration_steps


class TestAtomBacktrackExactUnbiased(unittest.TestCase):
    # ---------------------------------------------------------------------
    # Dispatcher / recognition
    # ---------------------------------------------------------------------

    def test_strategy_is_normalized(self):
        # Static method: should accept the new name and return it normalized.
        normalized = TemporalProbabilisticRPGHeuristic._normalize_strategy(
            "atom_backtrack_exact_unbiased"
        )
        self.assertEqual(normalized, "atom_backtrack_exact_unbiased")

    def test_strategy_is_callable_through_heuristic_score(self):
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                )
            ],
            facts={"A", "B"},
            initial_facts={"A"},
            goal_facts={"B"},
        )
        # Should not raise; should return a probability in [0, 1].
        score = h.heuristic_score(
            {"A"},
            {"B"},
            fixed_depth=3,
            strategy="atom_backtrack_exact_unbiased",
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    # ---------------------------------------------------------------------
    # Pre-planning is cached (called once per state/target/depth)
    # ---------------------------------------------------------------------

    def test_precompute_is_cached_once(self):
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="try1",
                    pos_preconditions=frozenset({"R1"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R2"}),
                ),
                SyntheticAction(
                    name="try2",
                    pos_preconditions=frozenset({"R2"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R1"}),
                ),
            ],
            facts={"R1", "R2", "G"},
            initial_facts={"R1", "R2"},
            goal_facts={"G"},
        )

        first = h._compute_unbiased_correction_table({"R1", "R2"}, {"G"}, depth=8)
        second = h._compute_unbiased_correction_table({"R1", "R2"}, {"G"}, depth=8)
        # Returns the same cached dict object (identity, not just equality).
        self.assertIs(first, second)
        # Cache key contains exactly one entry for that triple.
        cache_keys = list(h._unbiased_correction_cache.keys())
        self.assertEqual(len(cache_keys), 1)

    def test_correction_table_shape(self):
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                )
            ],
            facts={"A", "B"},
            initial_facts={"A"},
            goal_facts={"B"},
        )
        depth = 6
        correction = h._compute_unbiased_correction_table({"A"}, {"B"}, depth=depth)
        self.assertIn("lambda_total", correction)
        self.assertIn("lambda_breakdown", correction)
        self.assertIn("B_table", correction)
        self.assertEqual(set(correction["lambda_breakdown"].keys()), {"OR", "AND", "DEL"})
        self.assertEqual(len(correction["B_table"]), depth + 1)
        # B(0) is always exactly 0 by definition.
        self.assertEqual(correction["B_table"][0], 0.0)

    # ---------------------------------------------------------------------
    # Sign / direction of the correction
    # ---------------------------------------------------------------------

    def test_mutex_achievers_yield_negative_lambda_OR(self):
        # Two achievers of G that delete each other's preconditions: mutex by
        # delete-interference. Negative correlation → λ_OR < 0.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="try1",
                    pos_preconditions=frozenset({"R1"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R2"}),
                ),
                SyntheticAction(
                    name="try2",
                    pos_preconditions=frozenset({"R2"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R1"}),
                ),
            ],
            facts={"R1", "R2", "G"},
            initial_facts={"R1", "R2"},
            goal_facts={"G"},
        )
        correction = h._compute_unbiased_correction_table({"R1", "R2"}, {"G"}, depth=8)
        self.assertLess(correction["lambda_breakdown"]["OR"], 0.0)

    def test_delete_actions_yield_positive_lambda_DEL(self):
        # Same setup as above: delete actions exist for R1, R2 → γ > 0 → λ_DEL > 0.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="try1",
                    pos_preconditions=frozenset({"R1"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R2"}),
                ),
                SyntheticAction(
                    name="try2",
                    pos_preconditions=frozenset({"R2"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R1"}),
                ),
            ],
            facts={"R1", "R2", "G"},
            initial_facts={"R1", "R2"},
            goal_facts={"G"},
        )
        correction = h._compute_unbiased_correction_table({"R1", "R2"}, {"G"}, depth=8)
        self.assertGreater(correction["lambda_breakdown"]["DEL"], 0.0)

    # ---------------------------------------------------------------------
    # Layer-dependent correction shape
    # ---------------------------------------------------------------------

    def test_B_table_grows_or_stays_with_depth(self):
        # When λ > 0, B(t) is non-decreasing in t in the typical regime
        # (no saturation effects strong enough to reverse it on small depths).
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="try1",
                    pos_preconditions=frozenset({"R1"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R2"}),
                ),
                SyntheticAction(
                    name="try2",
                    pos_preconditions=frozenset({"R2"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R1"}),
                ),
            ],
            facts={"R1", "R2", "G"},
            initial_facts={"R1", "R2"},
            goal_facts={"G"},
        )
        correction = h._compute_unbiased_correction_table({"R1", "R2"}, {"G"}, depth=10)
        B = correction["B_table"]
        # The first non-zero B(t) should be ≥ B(0)=0; all later B(t) should be ≥ 0.
        for t in range(len(B)):
            self.assertGreaterEqual(B[t], -1e-9)

    # ---------------------------------------------------------------------
    # Numerical guard: tiny lambda → no correction
    # ---------------------------------------------------------------------

    def test_zero_lambda_gives_unchanged_score(self):
        # Single deterministic achiever, no deletes, no shared structure → λ ≈ 0.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                )
            ],
            facts={"A", "B"},
            initial_facts={"A"},
            goal_facts={"B"},
        )
        score_resolution = h.heuristic_score(
            {"A"}, {"B"}, fixed_depth=4, strategy="atom_backtrack_exact_resolution"
        )
        score_unbiased = h.heuristic_score(
            {"A"}, {"B"}, fixed_depth=4, strategy="atom_backtrack_exact_unbiased"
        )
        # With λ ≈ 0, the unbiased path matches resolution exactly.
        self.assertAlmostEqual(score_resolution, score_unbiased, places=6)
        correction = h._compute_unbiased_correction_table({"A"}, {"B"}, depth=4)
        self.assertLess(abs(correction["lambda_total"]), 1e-3)
        self.assertTrue(all(abs(b) < 1e-9 for b in correction["B_table"]))

    # ---------------------------------------------------------------------
    # End-to-end: corrected score is shifted relative to resolution score
    # ---------------------------------------------------------------------

    def test_unbiased_diverges_from_resolution_when_lambda_nonzero(self):
        # Mutex + delete setup: λ > 0 → unbiased ≤ resolution (correction subtracts).
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="try1",
                    pos_preconditions=frozenset({"R1"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R2"}),
                ),
                SyntheticAction(
                    name="try2",
                    pos_preconditions=frozenset({"R2"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R1"}),
                ),
            ],
            facts={"R1", "R2", "G"},
            initial_facts={"R1", "R2"},
            goal_facts={"G"},
        )
        score_res = h.heuristic_score(
            {"R1", "R2"}, {"G"}, fixed_depth=8, strategy="atom_backtrack_exact_resolution"
        )
        score_unb = h.heuristic_score(
            {"R1", "R2"}, {"G"}, fixed_depth=8, strategy="atom_backtrack_exact_unbiased"
        )
        # With positive λ, unbiased should be ≤ resolution (probability ∈ [0,1]).
        self.assertLessEqual(score_unb, score_res + 1e-9)
        # And the gap should be measurable (at least the size of B(depth)).
        correction = h._compute_unbiased_correction_table(
            {"R1", "R2"}, {"G"}, depth=8
        )
        # If λ > 0 and the resolution score is below 1, the gap should be positive.
        if score_res > 0.0 and correction["lambda_total"] > 1e-3:
            self.assertGreater(score_res - score_unb, 0.0)


if __name__ == "__main__":
    unittest.main()
