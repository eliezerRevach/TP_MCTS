"""
Single-call comparison: baseline_admissible_paths_table vs baseline_admissible.

Builds one MDP, constructs the PTRPG heuristic once, and makes a SINGLE
heuristic_score() call from the initial state for EACH strategy (no MDP/MCTS
search loop). Reports:

  * the heuristic value for each strategy, and
  * the path-table mutex instrumentation: how many times the OR layer used
    ``max`` (a certified mutex) instead of a flat ``sum``, plus the AND-layer
    cross-fact mutex detections / value drops.

The point: see whether path-mutex detection makes
``baseline_admissible_paths_table`` strictly TIGHTER (lower value) than plain
``baseline_admissible`` on a real domain state, and how often the mutex bound
actually fires.

Usage:
  python scripts/compare_paths_table_vs_admissible.py
  python scripts/compare_paths_table_vs_admissible.py --domain machine_shop --object_amount 2 --deadline 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_original_argv = sys.argv[:]
sys.argv = [_original_argv[0]]  # UP parser calls parse_args() at import time.

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
    "prob_conc": up.domains.Prob_Conc,
}


def build_mdp(domain, object_amount, deadline, seed):
    random.seed(seed)
    np.random.seed(seed)
    domain_cls = _DOMAIN_CLASS_MAP[domain]
    model = domain_cls(kind="regular", deadline=deadline, object_amount=object_amount)
    if domain == "nasa_rover":
        grounder = up.engines.Grounder(model.grounding_map())
    else:
        grounder = up.engines.Grounder()
    ground_problem = grounder._compile(model.problem).problem
    converted_problem = Convert_problem(ground_problem)._converted_problem
    return MDP(converted_problem, discount_factor=0.95, reward_mode="deadline", step_penalty=-0.05)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="machine_shop", choices=list(_DOMAIN_CLASS_MAP))
    p.add_argument("--object_amount", type=int, default=2)
    p.add_argument("--deadline", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depth", type=int, default=None, help="Heuristic depth (default = deadline).")
    args = p.parse_args(_original_argv[1:])

    depth = args.depth if args.depth is not None else args.deadline

    print(f"Building MDP: domain={args.domain} obj={args.object_amount} "
          f"deadline={args.deadline} depth={depth} seed={args.seed}", flush=True)
    mdp = build_mdp(args.domain, args.object_amount, args.deadline, args.seed)
    state = mdp.initial_state()
    goals = mdp.problem.goals

    heuristic = TemporalProbabilisticRPGHeuristic.from_problem(mdp.problem)

    def score(strategy, aggregation):
        return heuristic.heuristic_score(
            state, goals, aggregation=aggregation,
            fixed_depth=depth, start_time=0.0, strategy=strategy,
        )

    # Same-aggregation comparisons isolate the OR/AND PROPAGATION tightness from
    # the goal-aggregation choice (kernelized is a min-bound; product multiplies,
    # so kernelized >= product regardless of propagation).
    base_prod = score("baseline_admissible", "product")
    tab_prod = score("baseline_admissible_paths_table", "product")
    base_min = score("baseline_admissible", "min")
    tab_min = score("baseline_admissible_paths_table", "min")
    # The two strategies' NATIVE aggregations (what the search actually uses).
    base_native = base_prod  # baseline_admissible -> product
    tab_native = score("baseline_admissible_paths_table", "kernelized")

    print("\n" + "=" * 64)
    print("  SINGLE-CALL HEURISTIC VALUES  (same-aggregation = apples to apples)")
    print("=" * 64)
    print(f"  {'aggregation':<14}{'baseline_admissible':>22}{'paths_table':>16}{'base-table':>14}")
    for name, b, t in (("product", base_prod, tab_prod), ("min", base_min, tab_min)):
        d = b - t
        tag = "TIGHTER" if d > 1e-9 else ("LOOSER" if d < -1e-9 else "equal")
        print(f"  {name:<14}{b:>22.6f}{t:>16.6f}{d:>14.6f}  ({tag})")
    print("-" * 64)
    print(f"  native aggregations (what the search uses):")
    print(f"    baseline_admissible      (product)    = {base_native:.6f}")
    print(f"    paths_table              (kernelized) = {tab_native:.6f}")

    print("\n" + "=" * 64)
    print("  PATH-TABLE MUTEX INSTRUMENTATION (from the paths_table call)")
    print("=" * 64)
    print(f"  OR-layer mutex events (max used instead of sum) = {heuristic._paths_table_or_mutex_events}")
    print(f"  AND nodes (>=2 preconditions)                   = {heuristic._paths_table_and_nodes}")
    print(f"  AND nodes with a cross-fact mutex CERTIFIED     = {heuristic._paths_table_and_potential}")
    print(f"  AND nodes where the value DROPPED below Frechet = {heuristic._paths_table_and_hits}")
    print(f"  total AND mass shaved by mutex                  = {heuristic._paths_table_and_shaved:.6f}")
    print(f"  emitted rows / chained (inherited-footprint)    = "
          f"{heuristic._paths_table_emitted_rows} / {heuristic._paths_table_chained_rows}")
    print("=" * 64)

    # --- Self-mutex / partner diagnostic --------------------------------------
    # Does the strategy actually certify that an action conflicts with ITSELF
    # (a at [0,10] vs a at [5,15])? That is stored as the action being its OWN
    # partner in the cut map. Print the self-mutex set and the partner map.
    partners = heuristic._paths_partners()
    all_names = [m.name for m in heuristic._action_models]
    self_mutex = [n for n in all_names if heuristic._path_actions_mutex(n, n)]
    print("\n" + "=" * 64)
    print("  SELF-MUTEX / PARTNER DIAGNOSTIC")
    print("=" * 64)
    print(f"  total actions                                   = {len(all_names)}")
    print(f"  actions certified SELF-mutex (a conflicts a)    = {len(self_mutex)}")
    for n in self_mutex:
        print(f"      self-mutex: {n}")
    print(f"  actions with >=1 mutex partner (incl. self)     = {len(partners)}")
    shown = 0
    for name, plist in partners.items():
        has_self = name in plist
        print(f"      {name:<40} self={has_self} partners={list(plist)[:6]}"
              + (" ..." if len(plist) > 6 else ""))
        shown += 1
        if shown >= 20:
            print(f"      ... ({len(partners) - shown} more)")
            break
    print("=" * 64)


if __name__ == "__main__":
    main()
    sys.argv = _original_argv
