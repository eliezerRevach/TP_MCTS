"""
Sweep baseline_admissible vs baseline_admissible_paths_table (v3 cut-rows) across
domains / object counts / deadlines, looking for the BIGGEST value gap.

Single heuristic call per (domain, obj, deadline) from the initial state (no MDP
search loop). Reports, per config:
  base   = baseline_admissible value (product aggregation)
  table  = baseline_admissible_paths_table value (SAME product aggregation)
  gap    = base - table  (>0 means paths_table is tighter thanks to mutex)
  OR     = OR-layer max-instead-of-sum events
  ANDc   = AND nodes with a cross-fact mutex CERTIFIED
  ANDd   = AND nodes where value DROPPED below Frechet
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.argv = [sys.argv[0]]  # UP parser calls parse_args() at import.

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

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

# (domain, [object_amounts], [deadlines])
GRID = [
    ("machine_shop", [1, 2, 3], [10, 12, 14, 16, 18, 20]),
    ("nasa_rover",   [1, 2, 3], [10, 15, 20, 25]),
    ("stuck_car",    [1, 2],    [8, 10, 12, 15, 20]),
    ("stuck_car_1o", [1, 2],    [8, 10, 12, 15, 20]),
    ("conc",         [2, 3],    [6, 8, 10, 12]),
]

SEED = 42


def build_mdp(domain, object_amount, deadline, seed):
    random.seed(seed)
    np.random.seed(seed)
    model = _DOMAIN_CLASS_MAP[domain](kind="regular", deadline=deadline, object_amount=object_amount)
    if domain == "nasa_rover":
        grounder = up.engines.Grounder(model.grounding_map())
    else:
        grounder = up.engines.Grounder()
    ground = grounder._compile(model.problem).problem
    converted = Convert_problem(ground)._converted_problem
    return MDP(converted, discount_factor=0.95, reward_mode="deadline", step_penalty=-0.05)


def main():
    print(f"{'domain':<14}{'obj':>4}{'dl':>4}{'base':>10}{'table':>10}{'gap':>12}"
          f"{'OR':>5}{'ANDc':>6}{'ANDd':>6}")
    print("-" * 71)
    best = []
    for domain, objs, dls in GRID:
        for obj in objs:
            for dl in dls:
                try:
                    mdp = build_mdp(domain, obj, dl, SEED)
                    state = mdp.initial_state()
                    goals = mdp.problem.goals
                    h = TemporalProbabilisticRPGHeuristic.from_problem(mdp.problem)
                    base = h.heuristic_score(state, goals, aggregation="product",
                                             fixed_depth=dl, start_time=0.0,
                                             strategy="baseline_admissible")
                    table = h.heuristic_score(state, goals, aggregation="product",
                                              fixed_depth=dl, start_time=0.0,
                                              strategy="baseline_admissible_paths_table")
                    gap = base - table
                    orv = h._paths_table_or_mutex_events
                    andc = h._paths_table_and_potential
                    andd = h._paths_table_and_hits
                    flag = " <==" if gap > 1e-6 else ""
                    print(f"{domain:<14}{obj:>4}{dl:>4}{base:>10.5f}{table:>10.5f}"
                          f"{gap:>12.6f}{orv:>5}{andc:>6}{andd:>6}{flag}", flush=True)
                    best.append((gap, domain, obj, dl, base, table))
                except Exception as e:
                    print(f"{domain:<14}{obj:>4}{dl:>4}  ERROR: {type(e).__name__}: {e}", flush=True)

    print("\nTop gaps:")
    for gap, domain, obj, dl, base, table in sorted(best, reverse=True)[:10]:
        print(f"  gap={gap:.6f}  {domain} obj={obj} dl={dl}  base={base:.5f} table={table:.5f}")


if __name__ == "__main__":
    main()
