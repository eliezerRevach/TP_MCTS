"""
Per-call runtime: baseline_admissible vs baseline_admissible_paths_table (v3).

Times a SINGLE heuristic_score() call for each strategy from the initial state.
The heuristic memoizes propagation in self._query_cache, so to time real work
(not cache hits) we clear that cache before every timed call and average over N
repeats. Reports mean/median/min per call and the ratio.

Run with KMUTEX_K set via env (default here forces K=1):
    python scripts/time_paths_table_call.py --domain machine_shop --object_amount 2 --deadline 16 --k 1 --repeats 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean, median

_orig_argv = sys.argv[:]
sys.argv = [_orig_argv[0]]

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

import os
import random
import numpy as np

import unified_planning as up
import unified_planning.domains  # noqa: F401
from unified_planning.shortcuts import *  # noqa: F401,F403
from unified_planning.engines.convert_problem import Convert_problem
from unified_planning.engines.mdp import MDP
from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)

_DOMAIN_CLASS_MAP = {
    "nasa_rover": up.domains.Nasa_Rover,
    "machine_shop": up.domains.Machine_Shop,
    "stuck_car": up.domains.Stuck_Car,
    "stuck_car_1o": up.domains.Stuck_Car_1o,
    "conc": up.domains.Conc,
}


def build_mdp(domain, obj, deadline, seed):
    random.seed(seed)
    np.random.seed(seed)
    model = _DOMAIN_CLASS_MAP[domain](kind="regular", deadline=deadline, object_amount=obj)
    grounder = up.engines.Grounder(model.grounding_map()) if domain == "nasa_rover" else up.engines.Grounder()
    ground = grounder._compile(model.problem).problem
    converted = Convert_problem(ground)._converted_problem
    return MDP(converted, discount_factor=0.95, reward_mode="deadline", step_penalty=-0.05)


def time_strategy(h, state, goals, strategy, aggregation, depth, repeats):
    samples = []
    val = None
    for _ in range(repeats):
        h._query_cache.clear()  # force real propagation, not a cache hit
        t0 = time.perf_counter()
        val = h.heuristic_score(state, goals, aggregation=aggregation,
                                fixed_depth=depth, start_time=0.0, strategy=strategy)
        samples.append(time.perf_counter() - t0)
    return val, samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="machine_shop", choices=list(_DOMAIN_CLASS_MAP))
    p.add_argument("--object_amount", type=int, default=2)
    p.add_argument("--deadline", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=1, help="KMUTEX_K (path table row cap).")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args(_orig_argv[1:])

    os.environ["TP_MCTS_KMUTEX_K"] = str(args.k)
    depth = args.deadline

    print(f"domain={args.domain} obj={args.object_amount} dl={args.deadline} "
          f"depth={depth} K={args.k} repeats={args.repeats} (warmup={args.warmup})", flush=True)

    mdp = build_mdp(args.domain, args.object_amount, args.deadline, args.seed)
    state = mdp.initial_state()
    goals = mdp.problem.goals
    h = TemporalProbabilisticRPGHeuristic.from_problem(mdp.problem)

    # Warm up structural lazy-builds (mutex tables, specs) so they don't skew the
    # first timed sample. These are built once per instance and reused.
    for _ in range(args.warmup):
        h._query_cache.clear()
        h.heuristic_score(state, goals, aggregation="product", fixed_depth=depth,
                          start_time=0.0, strategy="baseline_admissible")
        h._query_cache.clear()
        h.heuristic_score(state, goals, aggregation="kernelized", fixed_depth=depth,
                          start_time=0.0, strategy="baseline_admissible_paths_table")

    base_val, base_s = time_strategy(h, state, goals, "baseline_admissible", "product", depth, args.repeats)
    tab_val, tab_s = time_strategy(h, state, goals, "baseline_admissible_paths_table", "kernelized", depth, args.repeats)

    def us(x):
        return x * 1e6

    print("\n" + "=" * 68)
    print(f"  {'strategy':<34}{'mean_ms':>10}{'median_ms':>11}{'min_ms':>9}")
    print("-" * 68)
    for name, s in (("baseline_admissible", base_s),
                    ("baseline_admissible_paths_table", tab_s)):
        print(f"  {name:<34}{mean(s)*1e3:>10.3f}{median(s)*1e3:>11.3f}{min(s)*1e3:>9.3f}")
    print("-" * 68)
    ratio = mean(tab_s) / mean(base_s) if mean(base_s) > 0 else float("inf")
    print(f"  paths_table / baseline  (mean)  = {ratio:.2f}x")
    print(f"  values: base={base_val:.6f}  table={tab_val:.6f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
    sys.argv = _orig_argv
