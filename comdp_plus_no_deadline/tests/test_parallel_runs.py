"""Parallel Script 3 runs (experiment_common: workers > 1).

The merge is checked against the REAL ``evaluate.evaluation_loop``: the chunks are
printed by it, and the merged numbers must equal what it prints for all episodes
together.
"""

import contextlib
import io
import math
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import experiment_common as ec  # noqa: E402
import unified_planning.shortcuts  # noqa: E402,F401  (unified_planning.engines needs it first)
from unified_planning.engines.solvers.evaluate import evaluation_loop  # noqa: E402


def printed(episodes):
    """evaluation_loop's stdout for these (success, time) episodes."""
    it = iter(episodes)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluation_loop(len(episodes), lambda: next(it), ())
    return buf.getvalue()


def episodes(rng, n, p_win):
    return [(1, rng.randint(10, 25)) if rng.random() < p_win else (0, -math.inf) for _ in range(n)]


def same(a, b):
    if a is None or b is None:
        return a == b
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def check_merge(chunks):
    merged = ec.parse_run_metrics(ec.merge_run_domain_outputs([printed(c) for c in chunks]))
    whole = ec.parse_run_metrics(printed([e for c in chunks for e in c]))
    for key in ("runs_total", "amount_success", "avg_success_time", "std_success_time", "success_rate"):
        assert same(merged[key], whole[key]), (key, merged[key], whole[key], chunks)


def test_merge_equals_one_loop_over_all_episodes():
    rng = random.Random(0)
    for _ in range(300):
        sizes = [rng.randint(1, 8) for _ in range(rng.randint(1, 6))]
        p = rng.choice([0.0, 0.1, 0.5, 0.9, 1.0])
        check_merge([episodes(rng, n, p) for n in sizes])


def test_merge_edge_cases():
    win, lose = (1, 17), (0, -math.inf)
    check_merge([[lose], [lose, lose]])                 # no success: -inf, -1
    check_merge([[win], [lose]])                        # one success: std -1
    check_merge([[win], [(1, 20)]])                     # two workers with one success each
    check_merge([[win, (1, 21), (1, 18)], [lose]])


def test_a_crashed_worker_is_not_counted():
    out = ec.merge_run_domain_outputs([printed([(1, 12), (0, -math.inf)]), "Traceback (most recent call last):\n"])
    m = ec.parse_run_metrics(out)
    assert (m["runs_total"], m["amount_success"]) == (2, 1)
    assert "worker 1 printed no summary" in out


def test_split_runs():
    assert ec.split_runs(100, 16) == [7] * 4 + [6] * 12
    assert ec.split_runs(3, 16) == [1, 1, 1]
    assert ec.split_runs(10, 1) == [10]


def test_workers_get_their_own_seeds_and_runs(monkeypatch):
    calls = []

    def fake(cmd, env):
        calls.append(cmd)
        n = int(cmd[cmd.index("--runs") + 1])
        return printed([(1, 15)] * n), 0

    monkeypatch.setattr(ec, "_run_cmd", fake)
    out, rc = ec.run_domain_subprocess(
        domain="nasa_rover", object_amount=2, deadline=25, runs=7, seed=123, solver="mcts",
        heuristic_name="trpg", temporal_heuristic_strategy="baseline", temporal_heuristic_depth=25,
        max_approx_seed=5, workers=3)
    arg = lambda c, flag: int(c[c.index(flag) + 1])  # noqa: E731
    assert [arg(c, "--runs") for c in calls] == [3, 2, 2]
    assert [arg(c, "--seed") for c in calls] == [123, 1123, 2123]
    assert [arg(c, "--max-approx-seed") for c in calls] == [5, 1005, 2005]
    m = ec.parse_run_metrics(out)
    assert rc == 0 and (m["runs_total"], m["amount_success"], m["avg_success_time"]) == (7, 7, 15.0)


def test_one_worker_is_the_old_path(monkeypatch):
    calls = []
    monkeypatch.setattr(ec, "_run_cmd", lambda cmd, env: (calls.append(cmd), (printed([(0, -math.inf)]), 0))[1])
    out, _rc = ec.run_domain_subprocess(
        domain="nasa_rover", object_amount=2, deadline=25, runs=4, seed=123, solver="mcts",
        heuristic_name="trpg", temporal_heuristic_strategy="baseline", temporal_heuristic_depth=25)
    assert len(calls) == 1 and "worker" not in out
    assert calls[0][calls[0].index("--runs") + 1] == "4" and calls[0][calls[0].index("--seed") + 1] == "123"
