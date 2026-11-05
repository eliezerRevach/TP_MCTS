"""rules = "stn": the state keeps the path's STN itself (``windows_lao``, module section "rules = stn").

The checks: the STN steps on small hand cases, the in-action retry loop, and on random toys
    true optimum  <=  STN table  <=  windows table (blur 0)
at the root for every r, plus every STN lookup from a reachable leaf (shifted fit) >= the truth.
The truth is a brute force over integer time: start any legal action, end a due one, or wait 1.
"""

import os
import random
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.temporal_stn_pdb import ToyModel  # noqa: E402
from comdp_plus_no_deadline.engines.windows_lao import (  # noqa: E402
    STN_EMPTY,
    WindowsLAO,
    WindowsTable,
    stn_e,
    stn_end,
    stn_root,
    stn_start,
)

F = Fraction


def stn_table(ops, goals, D, facts=(), rules="stn"):
    t = WindowsTable(WindowsLAO(ToyModel(ops, goals, D), heuristic="none", rules=rules, blur=0), D)
    assert t.build(set(facts))
    return t


# ---------------------------------------------------------------- the STN steps
def test_sequential_inside_a1_sums():
    k = stn_start(STN_EMPTY, 5, 0)                       # a1 (5)
    k = stn_start(k, 2, 0)                               # a2 (2) before a1
    k, c1 = stn_end(k, [5])
    assert c1 == 2 and stn_e(k) == (3,)
    k = stn_start(k, 3, 0)                               # a3 (3) before a1
    k, c2 = stn_end(k, [5])
    assert c2 == 3 and stn_e(k) == (0,)                  # charged 2 + 3, a1 has 0 left


def test_overlap_inside_a1_charges_the_max():
    k = stn_start(STN_EMPTY, 5, 0)                       # a1 (5)
    k = stn_start(k, 2, 0)                               # a2 (2) before a1
    k = stn_start(k, 3, 1)                               # a3 (3) after a2, before a1
    k, first = stn_end(k, [3, 5])
    k, second = stn_end(k, [5])
    assert first + second == 3                           # max(2, 3)


def test_a_longer_action_started_later_cannot_end_first():
    """The duration rule comes out of the STN: x (4) started after b (3) cannot end before it."""
    k = stn_start(STN_EMPTY, 3, 0)
    assert stn_start(k, 4, 0) is None
    assert stn_start(k, 3, 0) is not None                # equal length: they end together
    assert stn_start(k, 4, 1) is not None


def test_elapsed_time_is_seen():
    """b (3) runs across a's end (1): 2 left at most, so x (3) placed before b is impossible."""
    k = stn_start(STN_EMPTY, 3, 0)                       # b
    k = stn_start(k, 1, 0)                               # a (1) before b
    k, _ = stn_end(k, [3])
    assert stn_start(k, 3, 0) is None
    assert stn_start(k, 2, 0) is not None


def test_root_point():
    k = stn_root([2, 5], [3, 6])
    assert stn_e(k) == (2, 5)
    assert stn_root([F(1, 2)], [1]) is not None


# ---------------------------------------------------------------- values
def test_retry_inside_a_running_action_stops_at_three():
    ops = [
        {"name": "a1", "duration": 3, "pre": ("fresh",),
         "start": [(("door",), ("fresh",), 1.0)], "end": [((), ("door",), 1.0)]},
        {"name": "a2", "duration": 1, "pre": ("door",), "end_pre": ("door",),
         "end": [(("through",), (), 0.5), ((), (), 0.5)]},
    ]
    t = stn_table(ops, {"through"}, 10, {"fresh"})
    assert t.lookup({"fresh"}, 10)[0] == pytest.approx(0.875, abs=1e-9)
    # stn_lb: one state per (F, Q), the retry after three tries shares the state of the first -> not cut
    lb = stn_table(ops, {"through"}, 10, {"fresh"}, rules="stn_lb")
    assert lb.lookup({"fresh"}, 10)[0] == pytest.approx(1 - 0.5 ** 10, abs=1e-9)
    assert len(lb.states) < len(t.states) == lb.stats["stn_walked"]


def test_a_then_b():
    ops = [{"name": "a", "duration": 2, "end": [(("p",), (), 0.7), ((), (), 0.3)]},
           {"name": "b", "duration": 3, "pre": ("p",), "end": [(("g",), (), 0.8), ((), (), 0.2)]}]
    assert stn_table(ops, {"g"}, 8).lookup(set(), 8)[0] == pytest.approx(0.84, abs=1e-9)


def test_ilao_matches_the_table():
    ops = [{"name": "a", "duration": 2, "end": [(("p",), (), 0.7), ((), (), 0.3)]},
           {"name": "b", "duration": 3, "pre": ("p",), "end": [(("g",), (), 0.8), ((), (), 0.2)]},
           {"name": "c", "duration": 1, "end": [(("q",), (), 0.5), ((), (), 0.5)]}]
    t = stn_table(ops, {"g"}, 8)
    for r in range(9):
        lao = WindowsLAO(ToyModel(ops, {"g"}, 8), heuristic="none", rules="stn")
        assert lao.solve(set(), r) == pytest.approx(t.lookup(set(), r)[0], abs=1e-9)


# ---------------------------------------------------------------- against the truth
def truth(model, facts, running, r, memo):
    """Exact optimum over integer time: running = sorted ((key, time left), ...), r = time left."""
    if model.goals <= facts:
        return 1.0
    key = (facts, running, r)
    if key in memo:
        return memo[key]
    best = 0.0
    ops = model.ops()
    due = [k for k, rem in running if rem == 0]
    for k in due:
        if model.end_legal(k, facts):
            rest = tuple(x for x in running if x[0] != k)
            best = max(best, sum(p * truth(model, f2, rest, r, memo)
                                 for f2, p in model.outcomes((k, "E"), facts)))
    for k in model.legal_action_names(facts):
        d = int(ops[k].duration)
        nxt = tuple(sorted(running + ((k, d),)))
        best = max(best, sum(p * truth(model, f2, nxt, r, memo) for f2, p in model.outcomes((k, "S"), facts)))
    if running and not due and r >= 1:
        best = max(best, truth(model, facts, tuple((k, rem - 1) for k, rem in running), r - 1, memo))
    memo[key] = best
    return best


def random_toy(rng):
    pool = ["p", "q", "g"]
    ops = []
    for i in range(rng.randint(2, 3)):
        spec = {"name": f"a{i}", "duration": rng.randint(1, 3)}
        if rng.random() < 0.5:
            spec["pre"] = (rng.choice(pool[:2]),)
        if rng.random() < 0.3:
            spec["end_pre"] = (rng.choice(pool[:2]),)
        if rng.random() < 0.4:
            f = rng.choice(pool[:2])
            spec["start"] = [((f,), (), 1.0)] if rng.random() < 0.5 else [((), (f,), 1.0)]
        p = rng.choice([1.0, 0.7, 0.5])
        win = (rng.choice(pool),)
        spec["end"] = [(win, (), p)] + ([((), (), 1 - p)] if p < 1 else [])
        ops.append(spec)
    return ops


@pytest.mark.parametrize("seed", range(40))
def test_truth_le_stn_lb_and_stn_le_windows(seed):
    rng = random.Random(seed)
    ops = random_toy(rng)
    D = rng.randint(3, 6)
    facts = frozenset(rng.sample(["p", "q"], rng.randint(0, 1)))
    model = ToyModel(ops, {"g"}, D)
    stn = stn_table(ops, {"g"}, D, facts)
    win = stn_table(ops, {"g"}, D, facts, rules="windows")
    lb = stn_table(ops, {"g"}, D, facts, rules="stn_lb")
    memo = {}
    for r in range(D + 1):
        exact = truth(model, facts, (), r, memo)
        v_stn = stn.lookup(facts, r)[0]
        v_win = win.lookup(facts, r)[0]
        v_lb = lb.lookup(facts, r)[0]
        assert exact - 1e-9 <= v_stn <= v_win + 1e-9, (ops, D, facts, r, exact, v_stn, v_win)
        assert v_stn - 1e-9 <= v_lb, (ops, D, facts, r, v_stn, v_lb)       # merging by (F, Q) only loosens


@pytest.mark.parametrize("rules", ["stn", "stn_lb"])
@pytest.mark.parametrize("seed", range(25))
def test_every_shifted_lookup_is_above_the_truth(seed, rules):
    """Walk the true MDP at random; at every leaf the STN table's answer (exact or shifted fit) is
    >= the truth from that leaf."""
    rng = random.Random(1000 + seed)
    ops = random_toy(rng)
    D = rng.randint(4, 6)
    model = ToyModel(ops, {"g"}, D)
    t = stn_table(ops, {"g"}, D, rules=rules)
    memo, checked = {}, 0
    for _walk in range(30):
        facts, running, r = frozenset(), (), D
        for _step in range(12):
            if model.goals <= facts:
                break
            value, _kind = t.lookup(facts, r, running=list(running))
            exact = truth(model, facts, running, r, memo)
            if value is not None:
                assert value >= exact - 1e-9, (ops, D, facts, running, r, value, exact)
                checked += 1
            moves = []
            due = [k for k, rem in running if rem == 0]
            for k in due:
                if model.end_legal(k, facts):
                    moves.append(("E", k))
            moves += [("S", k) for k in model.legal_action_names(facts)]
            if running and not due and r >= 1:
                moves.append(("W", None))
            if not moves:
                break
            kind, k = rng.choice(moves)
            if kind == "W":
                running, r = tuple((x, rem - 1) for x, rem in running), r - 1
                continue
            outs = model.outcomes((k, kind), facts)
            facts = rng.choices([f for f, _p in outs], [p for _f, p in outs])[0]
            if kind == "S":
                running = tuple(sorted(running + ((k, int(model.ops()[k].duration)),)))
            else:
                running = tuple(x for x in running if x[0] != k)
    assert checked > 0
