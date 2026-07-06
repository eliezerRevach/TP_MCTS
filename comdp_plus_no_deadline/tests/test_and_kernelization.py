"""
Tests for the enhanced OR layer (mutex-preserving sum + dominance) and the AND-
layer temporal-mutex kernelization in ``path_mutex`` — the multi-row AND design
worked out with the user.

Structure of the guarantees checked here:
- ``insert_path`` / ``table_or_hazard_paths``: never exceeds the union bound; the
  mutex-preserving sum keeps "a,b free of each other but both mutex to c" so the
  summed row is still certified mutex to c (the fix for the erase-on-sum data loss).
- ``dominance_prune``: value-preserving Pareto compaction.
- ``and_support_kernelized``: equals the Frechet min when nothing is mutex, drops
  below it exactly on a genuine (temporal) conflict, and — the core correctness
  property — always equals a brute-force max-over-feasible-selections of min(prob)
  over ALL facts jointly (i.e. the component decomposition is exact).
"""

import random
import unittest

from comdp_plus_no_deadline.engines.path_mutex import (
    Row,
    Segment,
    and_components,
    and_cumulative_bound,
    and_has_mutex,
    and_support_kernelized,
    cumulative_merge_truncate,
    dominance_prune,
    dominates,
    exact_component_value,
    guaranteed_mutex,
    insert_path,
    table_or_hazard_paths,
)


# Machine-contention mutex: two segments conflict iff they name the SAME machine
# and their windows overlap. mutex_fn answers "same resource?".
def _machine_mutex(a, b):
    return a == b


def _row(prob, machine=None, start=0, end=0, alts=None, complete=True):
    """A row occupying ``machine`` over ``[start,end)`` (or free if machine None)."""
    if machine is None:
        fp = frozenset()
    else:
        fp = frozenset({Segment(machine, start, end)})
    return Row(prob, fp, alternatives=tuple(alts) if alts else (), complete=complete)


class TestMutexPreservingSum(unittest.TestCase):
    def test_summed_row_stays_mutex_to_third(self):
        # The "a@[0,5], b@[10,15] both hit c@[0,20]" example: a and b occupy machine
        # mc at NON-overlapping times (free of each other), but c occupies mc over
        # [0,20] which overlaps BOTH. After summing a+b, the summed row must still
        # be certified mutex to c (every realization -> a or b -> conflicts with c).
        a = _row(0.3, "mc", 0, 5)
        b = _row(0.3, "mc", 10, 15)
        c = _row(0.4, "mc", 0, 20)
        # a and b are free of each other (mc[0,5) vs mc[10,15) don't overlap).
        self.assertFalse(guaranteed_mutex(a, b, _machine_mutex))
        # both a and b are mutex to c (distinct overlapping mc windows).
        self.assertTrue(guaranteed_mutex(a, c, _machine_mutex))
        self.assertTrue(guaranteed_mutex(b, c, _machine_mutex))
        # Sum a+b via insert_path (they are free -> Case 3 sum).
        table = [a]
        insert_path(table, b, k=3, mutex_fn=_machine_mutex)
        self.assertEqual(len(table), 1)  # summed into one row
        summed = table[0]
        self.assertAlmostEqual(summed.prob, 0.6)
        # THE POINT: summed row is still mutex to c (both members were) — the old
        # erase-on-sum would have made it free and LOST this.
        self.assertTrue(guaranteed_mutex(summed, c, _machine_mutex))

    def test_summed_row_not_mutex_when_only_one_member_is(self):
        # If only a is mutex to c and b is genuinely free, the summed row must NOT
        # be mutex to c (a realization via b escapes) — admissibility direction.
        a = _row(0.3, "mc", 0, 5)          # occupies mc -> mutex to c
        b = _row(0.3, "mb", 0, 5)          # occupies mb -> free of c
        c = _row(0.4, "mc", 0, 20)         # distinct overlapping mc window
        table = [a]
        insert_path(table, b, k=3, mutex_fn=_machine_mutex)
        summed = table[0]
        self.assertFalse(guaranteed_mutex(summed, c, _machine_mutex))

    def test_never_exceeds_union(self):
        rng = random.Random(7)
        for _ in range(200):
            supports = []
            for i in range(rng.randint(1, 5)):
                m = rng.choice(["m0", "m1", "m2"])
                s = rng.randint(0, 5)
                supports.append((f"a{i}", rng.random(), frozenset({Segment(m, s, s + rng.randint(1, 4))})))
            res = table_or_hazard_paths(supports, _machine_mutex, k=rng.randint(1, 4))
            self.assertLessEqual(res.value, res.union_value + 1e-12)

    def test_reduces_to_union_when_no_mutex(self):
        supports = [("a", 0.2, frozenset({Segment("m0", 0, 5)})),
                    ("b", 0.3, frozenset({Segment("m1", 0, 5)})),  # different machine
                    ("c", 0.1, frozenset({Segment("m2", 0, 5)}))]
        res = table_or_hazard_paths(supports, _machine_mutex, k=3)
        self.assertAlmostEqual(res.value, 0.6)
        self.assertAlmostEqual(res.union_value, 0.6)


class TestDominance(unittest.TestCase):
    def test_dominates_subset_footprint_higher_prob(self):
        a = _row(0.8, "m", 0, 5)                                   # fp = {m[0,5)}
        b = Row(0.6, frozenset({Segment("m", 0, 5), Segment("m2", 0, 5)}))  # superset fp, lower prob
        self.assertTrue(dominates(a, b))
        self.assertFalse(dominates(b, a))

    def test_dominance_prune_keeps_pareto_and_max(self):
        a = _row(0.8, "m", 0, 5)
        b = Row(0.6, frozenset({Segment("m", 0, 5), Segment("m2", 0, 5)}))  # dominated by a
        c = _row(0.7, "m3", 0, 5)  # incomparable (different machine)
        table = [a, b, c]
        before_max = max(r.prob for r in table)
        dominance_prune(table)
        self.assertIn(a, table)
        self.assertIn(c, table)
        self.assertNotIn(b, table)
        self.assertAlmostEqual(max(r.prob for r in table), before_max)  # value preserved

    def test_equal_rows_keep_exactly_one(self):
        a = _row(0.5, "m", 0, 5)
        b = _row(0.5, "m", 0, 5)  # identical
        table = [a, b]
        dominance_prune(table)
        self.assertEqual(len(table), 1)


class TestAndGateAndComponents(unittest.TestCase):
    def test_gate_no_mutex(self):
        fact_rows = {"f1": [_row(0.9, "m0", 0, 5)], "f2": [_row(0.8, "m1", 0, 5)]}
        self.assertFalse(and_has_mutex(fact_rows, _machine_mutex))

    def test_gate_mutex(self):
        fact_rows = {"f1": [_row(0.9, "m", 0, 5)], "f2": [_row(0.8, "m", 3, 8)]}
        self.assertTrue(and_has_mutex(fact_rows, _machine_mutex))

    def test_components_split(self):
        fact_rows = {
            "f1": [_row(0.9, "ma", 0, 5)],
            "f2": [_row(0.9, "ma", 2, 7)],   # mutex with f1 (machine ma)
            "f3": [_row(0.9, "mb", 0, 5)],
            "f4": [_row(0.9, "mb", 2, 7)],   # mutex with f3 (machine mb)
            "f5": [_row(0.9, "mc", 0, 5)],   # isolated
        }
        comps = and_components(fact_rows, _machine_mutex)
        comp_sets = sorted([sorted(c) for c in comps])
        self.assertEqual(comp_sets, [["f1", "f2"], ["f3", "f4"], ["f5"]])


class TestAndKernelizedValue(unittest.TestCase):
    def test_no_mutex_is_frechet_min(self):
        fact_rows = {"f1": [_row(0.9, "m0", 0, 5)], "f2": [_row(0.7, "m1", 0, 5)]}
        res = and_support_kernelized(fact_rows, _machine_mutex)
        self.assertAlmostEqual(res.value, 0.7)  # min(0.9, 0.7)
        self.assertFalse(res.tightened)

    def test_two_pieces_one_machine_tight_deadline_serialization(self):
        # Both single paths occupy machine m over OVERLAPPING windows -> the only
        # combo is infeasible -> value 0 (can't have both done via these paths).
        fact_rows = {"x1_done": [_row(0.9, "m", 0, 5)], "x2_done": [_row(0.8, "m", 3, 8)]}
        res = and_support_kernelized(fact_rows, _machine_mutex)
        self.assertAlmostEqual(res.value, 0.0)
        self.assertAlmostEqual(res.frechet_min, 0.8)
        self.assertTrue(res.tightened)

    def test_two_pieces_one_machine_loose_deadline_no_conflict(self):
        # Windows no longer overlap (serialized) -> no mutex -> Frechet min back.
        fact_rows = {"x1_done": [_row(0.9, "m", 0, 5)], "x2_done": [_row(0.8, "m", 5, 10)]}
        res = and_support_kernelized(fact_rows, _machine_mutex)
        self.assertAlmostEqual(res.value, 0.8)
        self.assertFalse(res.tightened)

    def test_free_fallback_prevents_zero(self):
        # Each fact also has a machine-free alternative -> feasible selections exist
        # -> value drops below Frechet but not to 0.
        fact_rows = {
            "x1_done": [_row(0.9, "m", 0, 5), _row(0.5, machine=None)],
            "x2_done": [_row(0.8, "m", 3, 8), _row(0.4, machine=None)],
        }
        res = and_support_kernelized(fact_rows, _machine_mutex)
        # best feasible: x1 free(0.5) with x2 machine(0.8) -> min 0.5; or x1 m(0.9) x2 free(0.4)->0.4
        self.assertAlmostEqual(res.value, 0.5)
        self.assertTrue(res.tightened)
        self.assertLess(res.value, res.frechet_min)

    def test_incomplete_rows_do_not_tighten(self):
        # Mark the machine rows incomplete -> they can never certify mutex -> no
        # tightening (safe no-op). This is the admissibility guard in action.
        fact_rows = {
            "x1_done": [_row(0.9, "m", 0, 5, complete=False)],
            "x2_done": [_row(0.8, "m", 3, 8, complete=False)],
        }
        res = and_support_kernelized(fact_rows, _machine_mutex)
        self.assertAlmostEqual(res.value, 0.8)  # Frechet min, untouched
        self.assertFalse(res.tightened)

    def test_vacuous_conjunction_is_one(self):
        res = and_support_kernelized({}, _machine_mutex)
        self.assertAlmostEqual(res.value, 1.0)


class TestKernelizedEqualsBruteForce(unittest.TestCase):
    """The core correctness property: the gated + component-decomposed pipeline
    equals a single brute force max-over-feasible-selections-of-min over ALL facts
    jointly. Randomized."""

    def test_random_instances(self):
        rng = random.Random(20240703)
        for _ in range(400):
            n_facts = rng.randint(1, 4)
            fact_rows = {}
            for fi in range(n_facts):
                rows = []
                for _ri in range(rng.randint(1, 3)):
                    if rng.random() < 0.35:
                        rows.append(_row(round(rng.random(), 3), machine=None))  # free
                    else:
                        m = rng.choice(["m0", "m1"])
                        s = rng.randint(0, 4)
                        rows.append(_row(round(rng.random(), 3), m, s, s + rng.randint(1, 3)))
                fact_rows[f"f{fi}"] = rows
            res = and_support_kernelized(fact_rows, _machine_mutex)
            brute = exact_component_value(list(fact_rows), fact_rows, _machine_mutex)
            self.assertAlmostEqual(res.value, brute, places=9,
                                   msg=f"{fact_rows}")
            # Never exceeds the Frechet min.
            self.assertLessEqual(res.value, res.frechet_min + 1e-12)


class TestAndCumulativeBound(unittest.TestCase):
    """The bound used by the table-flowing strategy: cumulative (free-summed)
    per-fact tables, union-bound over compatible cross-fact tuples, capped by
    Frechet."""

    def test_no_mutex_is_frechet(self):
        fr = {"f1": [_row(0.5, "m0", 0, 5), _row(0.4, machine=None)],
              "f2": [_row(0.7, "m1", 0, 5)]}
        marg = {"f1": 0.9, "f2": 0.7}
        self.assertAlmostEqual(and_cumulative_bound(fr, marg, _machine_mutex), 0.7)

    def test_all_paths_conflict_serialization_zero(self):
        # Each fact's ONLY path uses machine m over overlapping windows -> the only
        # cross tuple is infeasible -> union sum 0 -> bound 0.
        fr = {"x1": [_row(0.9, "m", 0, 5)], "x2": [_row(0.8, "m", 3, 8)]}
        marg = {"x1": 0.9, "x2": 0.8}
        self.assertAlmostEqual(and_cumulative_bound(fr, marg, _machine_mutex), 0.0)

    def test_loose_deadline_no_conflict_is_frechet(self):
        fr = {"x1": [_row(0.9, "m", 0, 5)], "x2": [_row(0.8, "m", 5, 10)]}  # touch, no overlap
        marg = {"x1": 0.9, "x2": 0.8}
        self.assertAlmostEqual(and_cumulative_bound(fr, marg, _machine_mutex), 0.8)

    def test_partial_conflict_union_of_compatible(self):
        # x1 has a machine path (0.6) AND a free path (0.3); x2 one machine path 0.8.
        # compatible tuples: (free 0.3, m 0.8) -> min 0.3. (m 0.6, m 0.8) infeasible.
        # sum = 0.3 -> bound = min(frechet 0.8, 0.3) = 0.3.
        fr = {"x1": [_row(0.6, "m", 0, 5), _row(0.3, machine=None)],
              "x2": [_row(0.8, "m", 3, 8)]}
        marg = {"x1": 0.9, "x2": 0.8}
        self.assertAlmostEqual(and_cumulative_bound(fr, marg, _machine_mutex), 0.3)

    def test_always_le_frechet_random(self):
        rng = random.Random(11)
        for _ in range(300):
            fr, marg = {}, {}
            for fi in range(rng.randint(1, 3)):
                rows = []
                tot = 0.0
                for _ri in range(rng.randint(1, 3)):
                    p = round(rng.random() * 0.5, 3)
                    tot += p
                    if rng.random() < 0.4:
                        rows.append(_row(p, machine=None))
                    else:
                        s = rng.randint(0, 4)
                        rows.append(_row(p, rng.choice(["m0", "m1"]), s, s + rng.randint(1, 3)))
                fr[f"f{fi}"] = rows
                marg[f"f{fi}"] = min(1.0, tot)
            frechet = min(marg.values())
            self.assertLessEqual(and_cumulative_bound(fr, marg, _machine_mutex), frechet + 1e-12)


class TestCumulativeMergeTruncate(unittest.TestCase):
    def test_preserves_sum_and_caps_rows(self):
        table = [_row(0.1, "m", 0, 5), _row(0.2, "m", 5, 10),
                 _row(0.3, machine=None), _row(0.05, "m", 10, 15)]
        before = sum(r.prob for r in table)
        cumulative_merge_truncate(table, k=2)
        self.assertLessEqual(len(table), 2)
        self.assertAlmostEqual(sum(r.prob for r in table), before)  # sum preserved
        # merged rows keep only the INTERSECTION footprint (guaranteed by both);
        # disagreeing merges degrade to empty fp, which certifies nothing.
        for r in table:
            self.assertIsInstance(r.footprint, frozenset)

    def test_dedupe_identical_footprints_sums(self):
        seg = frozenset({Segment("m", 0, 5)})
        table = [Row(0.2, seg), Row(0.3, seg), Row(0.1, frozenset())]
        cumulative_merge_truncate(table, k=5)
        self.assertEqual(len(table), 2)  # the two identical-fp rows merged
        merged = [r for r in table if r.footprint == seg][0]
        self.assertAlmostEqual(merged.prob, 0.5)
        self.assertTrue(merged.complete)  # both complete -> still certifies

    def test_intersection_merge_keeps_shared_evidence(self):
        shared = Segment("m", 0, 5)
        r1 = Row(0.1, frozenset({shared, Segment("a", 0, 2)}))
        r2 = Row(0.2, frozenset({shared, Segment("b", 3, 6)}))
        r3 = Row(0.9, frozenset({Segment("c", 0, 9)}))
        table = [r1, r2, r3]
        cumulative_merge_truncate(table, k=2)
        self.assertEqual(len(table), 2)
        weakest = min(table, key=lambda r: r.prob)
        self.assertEqual(weakest.footprint, frozenset({shared}))  # shared part survives
        self.assertAlmostEqual(weakest.prob, 0.3)


class TestTableStrategyEngine(unittest.TestCase):
    """End-to-end: the baseline_admissible_paths_table strategy in the engine."""

    def _two_pieces_one_machine(self):
        from comdp_plus_no_deadline.tests.test_path_mutex import SyntheticAction
        from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
            TemporalProbabilisticRPGHeuristic,
        )
        return TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction("use_m_x1", frozenset({"free_m"}), frozenset({"done_x1"}),
                                5, del_effects=frozenset({"free_m"})),
                SyntheticAction("use_m_x2", frozenset({"free_m"}), frozenset({"done_x2"}),
                                5, del_effects=frozenset({"free_m"})),
            ],
            facts={"free_m", "done_x1", "done_x2"},
            initial_facts={"free_m"},
            goal_facts={"done_x1", "done_x2"},
        )

    def test_kernelized_serialization_signal(self):
        # Two pieces, one machine, each machine-use = duration 5. Both-done can't
        # happen before layer 10 (serial). Kernelized must be 0 while too tight,
        # then rise to the baseline value once there is room to serialize.
        h = self._two_pieces_one_machine()
        s, g = {"free_m"}, {"done_x1", "done_x2"}
        tight = h.heuristic_score(s, g, fixed_depth=5, strategy="baseline_admissible_paths_table",
                                  aggregation="kernelized")
        just_short = h.heuristic_score(s, g, fixed_depth=7, strategy="baseline_admissible_paths_table",
                                       aggregation="kernelized")
        serial_ok = h.heuristic_score(s, g, fixed_depth=10, strategy="baseline_admissible_paths_table",
                                      aggregation="kernelized")
        self.assertAlmostEqual(tight, 0.0)        # can't both finish by 5
        self.assertAlmostEqual(just_short, 0.0)   # nor by 7 (serial needs 10)
        self.assertGreater(serial_ok, 0.5)        # serialized by 10 -> feasible

    def test_table_never_exceeds_baseline_product(self):
        h = self._two_pieces_one_machine()
        s, g = {"free_m"}, {"done_x1", "done_x2"}
        for depth in (5, 7, 10, 20):
            base = h.heuristic_score(s, g, fixed_depth=depth, strategy="baseline_admissible",
                                     aggregation="product")
            tbl = h.heuristic_score(s, g, fixed_depth=depth, strategy="baseline_admissible_paths_table",
                                    aggregation="product")
            self.assertLessEqual(tbl, base + 1e-9, msg=f"depth={depth}")

    def test_table_le_baseline_nonsaturated_probabilistic(self):
        # Probabilistic single-achiever goals (no mutex) -> tables must match
        # baseline exactly (degenerate), and never exceed it, in a NON-saturated
        # regime (probs stay < 1).
        from comdp_plus_no_deadline.tests.test_path_mutex import SyntheticAction
        from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
            TemporalProbabilisticRPGHeuristic, TemporalRelaxedActionModel,
        )
        h = TemporalProbabilisticRPGHeuristic(
            actions=[SyntheticAction("mk_x", frozenset({"a"}), frozenset({"x"}), 1),
                     SyntheticAction("mk_y", frozenset({"b"}), frozenset({"y"}), 1)],
            facts={"a", "b", "x", "y"}, initial_facts={"a", "b"}, goal_facts={"x", "y"},
        )
        # low success probs so nothing saturates
        h._action_models = [
            TemporalRelaxedActionModel("mk_x", frozenset({"a"}), {"x": 0.5}, 1),
            TemporalRelaxedActionModel("mk_y", frozenset({"b"}), {"y": 0.5}, 1),
        ]
        h._actions_by_effect_fact = h._build_actions_by_effect_fact()
        for depth in (2, 4, 8):
            base_prod = h.heuristic_score({"a", "b"}, {"x", "y"}, fixed_depth=depth,
                                          strategy="baseline_admissible", aggregation="product")
            base_min = h.heuristic_score({"a", "b"}, {"x", "y"}, fixed_depth=depth,
                                         strategy="baseline_admissible", aggregation="min")
            tbl = h.heuristic_score({"a", "b"}, {"x", "y"}, fixed_depth=depth,
                                    strategy="baseline_admissible_paths_table", aggregation="product")
            kern = h.heuristic_score({"a", "b"}, {"x", "y"}, fixed_depth=depth,
                                     strategy="baseline_admissible_paths_table", aggregation="kernelized")
            self.assertGreater(base_prod, 0.0)                  # non-saturated
            self.assertLessEqual(tbl, base_prod + 1e-9, msg=f"product depth={depth}")
            # kernelized is a Frechet-min-family bound; no mutex here -> it equals
            # the min of the table marginals, which is <= baseline's min.
            self.assertLessEqual(kern, base_min + 1e-9, msg=f"min depth={depth}")


class TestChainedFootprints(unittest.TestCase):
    """v2: the machine conflict sits ONE LEVEL ABOVE the goal facts — only
    transitive (chained) footprints can carry it to the goal AND."""

    def _two_level_machine(self):
        from comdp_plus_no_deadline.tests.test_path_mutex import SyntheticAction
        from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
            TemporalProbabilisticRPGHeuristic,
        )
        # use_m_xi consumes the shared machine (dur 3) -> prep_i; finish_i
        # (dur 1, NO resource) -> goal_i. The goals' immediate achievers are
        # resource-free; the contention is inherited from use_m_xi.
        return TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction("use_m_x1", frozenset({"free_m"}), frozenset({"prep_1"}),
                                3, del_effects=frozenset({"free_m"})),
                SyntheticAction("use_m_x2", frozenset({"free_m"}), frozenset({"prep_2"}),
                                3, del_effects=frozenset({"free_m"})),
                SyntheticAction("finish_1", frozenset({"prep_1"}), frozenset({"goal_1"}), 1),
                SyntheticAction("finish_2", frozenset({"prep_2"}), frozenset({"goal_2"}), 1),
            ],
            facts={"free_m", "prep_1", "prep_2", "goal_1", "goal_2"},
            initial_facts={"free_m"},
            goal_facts={"goal_1", "goal_2"},
        )

    def test_chain_carries_machine_conflict_to_goals(self):
        h = self._two_level_machine()
        s, g = {"free_m"}, {"goal_1", "goal_2"}
        # serial truth: use_m (3) + use_m (3) + finish (1) = 7 for both goals.
        tight = h.heuristic_score(s, g, fixed_depth=4, strategy="baseline_admissible_paths_table",
                                  aggregation="kernelized")
        chained = h._paths_table_chained_rows
        loose = h.heuristic_score(s, g, fixed_depth=12, strategy="baseline_admissible_paths_table",
                                  aggregation="kernelized")
        self.assertGreater(chained, 0, "no chained rows were emitted at all")
        self.assertAlmostEqual(tight, 0.0,
                               msg="chained footprints failed to zero the tight-deadline conjunction")
        self.assertGreater(loose, 0.5, "loose deadline must recover (serialized windows)")

    def test_chained_strategy_marginals_le_baseline(self):
        h = self._two_level_machine()
        s, g = {"free_m"}, {"goal_1", "goal_2"}
        for depth in (4, 7, 12):
            base = h.heuristic_score(s, g, fixed_depth=depth,
                                     strategy="baseline_admissible", aggregation="product")
            tbl = h.heuristic_score(s, g, fixed_depth=depth,
                                    strategy="baseline_admissible_paths_table", aggregation="product")
            self.assertLessEqual(tbl, base + 1e-9, msg=f"depth={depth}")


class TestCutAltsRegression(unittest.TestCase):
    """The a,b->(c,d) case under the FLAT default (GROUP_CAP=1, user-specified):
    OR-merging heterogeneous routes EMPTIES the path (keeps only the shared
    cut), so (a|b)-vs-(c|d) is NOT certified (both paths free -> sum, the
    accepted, sound loss). One-sided cases still certify via the surviving cut
    hitting the other row's non-empty path. Raising GROUP_CAP (>=2) restores
    the nested certification — tested via explicit alts_or with cap=2."""

    def _mk(self, action, w, cut_actions, cut_w=None):
        from comdp_plus_no_deadline.engines.path_mutex import CutRow
        cw = cut_w or w
        return CutRow(0.4, ({action: (w,)},), {x: (cw,) for x in cut_actions})

    def test_flat_ab_or_merge_empties_path_keeps_cut(self):
        # FLAT default (GROUP_CAP=1): a|b -> path free, mutex on (c,d).
        from comdp_plus_no_deadline.engines.path_mutex import (
            cutrow_mutex, cutrow_or_merge_into,
        )
        W = (0, 10)
        ra = self._mk("a", W, ("c", "d"))
        cutrow_or_merge_into(ra, self._mk("b", W, ("c", "d")), retry=False)
        self.assertEqual(ra.alts, ({},))                       # path = free
        self.assertEqual(set(ra.cut), {"c", "d"})              # shared cut kept
        # one-sided certification still works: (a|b)'s cut hits d's path.
        rd = self._mk("d", W, ("a", "b"))
        self.assertTrue(cutrow_mutex(ra, rd))
        # both-sides-merged: paths both free -> NOT certified -> sum (the
        # accepted flat-design loss; sound direction).
        rc = self._mk("c", W, ("a", "b"))
        cutrow_or_merge_into(rc, self._mk("d", W, ("a", "b")), retry=False)
        self.assertFalse(cutrow_mutex(ra, rc))

    def test_nested_cap2_restores_ab_vs_cd(self):
        # Raising the nest knob (cap=2) keeps both routes revealed -> certified.
        from comdp_plus_no_deadline.engines.path_mutex import (
            CutRow, alts_or, cutrow_mutex, map_or_merge,
        )
        W = (0, 10)
        ra, rb = self._mk("a", W, ("c", "d")), self._mk("b", W, ("c", "d"))
        rc, rd = self._mk("c", W, ("a", "b")), self._mk("d", W, ("a", "b"))
        r_ab = CutRow(0.8, alts_or(ra.alts, rb.alts, cap=2),
                      map_or_merge(ra.cut, rb.cut))
        r_cd = CutRow(0.8, alts_or(rc.alts, rd.alts, cap=2),
                      map_or_merge(rc.cut, rd.cut))
        self.assertEqual(len(r_ab.alts), 2)
        self.assertTrue(cutrow_mutex(r_ab, r_cd))

    def test_partial_cut_does_not_certify(self):
        from comdp_plus_no_deadline.engines.path_mutex import (
            cutrow_mutex, cutrow_or_merge_into,
        )
        W = (0, 10)
        ra = self._mk("a", W, ("c",))      # a mutex ONLY to c
        rb = self._mk("b", W, ("c", "d"))
        rc = self._mk("c", W, ("a", "b"))
        rd = self._mk("d", W, ("b",))      # d mutex ONLY to b
        cutrow_or_merge_into(ra, rb, retry=False)   # (a|b): shared cut = {c}
        cutrow_or_merge_into(rc, rd, retry=False)   # (c|d): shared cut = {b}
        # realization a + d is compatible -> must NOT certify
        self.assertFalse(cutrow_mutex(ra, rc))

    def test_and_bound_flat_semantics(self):
        from comdp_plus_no_deadline.engines.path_mutex import (
            cut_and_bound, cutrow_or_merge_into,
        )
        W = (0, 10)
        # One-sided case: (a|b) vs a plain d-row -> certified -> zero.
        ra = self._mk("a", W, ("c", "d"))
        cutrow_or_merge_into(ra, self._mk("b", W, ("c", "d")), retry=False)
        rd = self._mk("d", W, ("a", "b"))
        value, detected = cut_and_bound(
            {"f1": [ra], "f2": [rd]}, {"f1": 0.8, "f2": 0.4}
        )
        self.assertTrue(detected)
        self.assertAlmostEqual(value, 0.0)
        # Both-sides-merged (flat loss): paths free -> falls back to Frechet.
        rc = self._mk("c", W, ("a", "b"))
        cutrow_or_merge_into(rc, self._mk("d", W, ("a", "b")), retry=False)
        value2, detected2 = cut_and_bound(
            {"f1": [ra], "f2": [rc]}, {"f1": 0.8, "f2": 0.8}
        )
        self.assertFalse(detected2)
        self.assertAlmostEqual(value2, 0.8)


if __name__ == "__main__":
    unittest.main()
