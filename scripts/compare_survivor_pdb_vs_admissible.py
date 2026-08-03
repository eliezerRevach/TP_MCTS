"""
Single-call comparison: baseline_admissible_survivor_pdb vs baseline_admissible.

Builds one MDP, constructs the PTRPG heuristic once, and makes a SINGLE
heuristic_score() call from the initial state for each strategy (no MDP/MCTS
search loop), across a range of deadlines.

The point is the DEADLINE SWEEP, not a single number. The marginal-only
cumulative bound ``min(1, sum_t H_t)`` accumulates hazard forever, so it
saturates to 1.0 as the deadline grows and every tree layer starts looking
identical to UCT. The survivor-PDB tracks a few facts JOINTLY, which makes the
per-layer conditional exact inside the pattern and removes the cap entirely.
What to look for:

  * value(survivor_pdb) <= value(admissible) at every deadline (admissibility
    is preserved by clamping to the min of the two bounds), and
  * the gap WIDENING with the deadline — that is the layer-bias the whole
    exercise is about.

Usage:
  python scripts/compare_survivor_pdb_vs_admissible.py
  python scripts/compare_survivor_pdb_vs_admissible.py --domain nasa_rover --deadlines 10,25,40
"""

from __future__ import annotations

import argparse
import sys
import time
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
    return MDP(
        converted_problem,
        discount_factor=0.95,
        reward_mode="deadline",
        step_penalty=-0.05,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="nasa_rover", choices=list(_DOMAIN_CLASS_MAP))
    p.add_argument("--object_amount", type=int, default=2)
    p.add_argument("--deadlines", default="10,15,25,40")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(_original_argv[1:])

    deadlines = [int(x) for x in str(args.deadlines).split(",") if x.strip()]

    print(f"domain={args.domain} obj={args.object_amount} seed={args.seed}", flush=True)
    print(
        f"{'deadline':>9} {'admissible':>12} {'survivor_pdb':>13} "
        f"{'gap':>9} {'adm_ms':>8} {'pdb_ms':>8}"
    )

    for deadline in deadlines:
        mdp = build_mdp(args.domain, args.object_amount, deadline, args.seed)
        state = mdp.initial_state()
        goals = mdp.problem.goals
        heuristic = TemporalProbabilisticRPGHeuristic.from_problem(mdp.problem)

        def score(strategy):
            started = time.perf_counter()
            value = heuristic.heuristic_score(
                state,
                goals,
                aggregation="product",
                fixed_depth=deadline,
                start_time=0.0,
                strategy=strategy,
            )
            return float(value), (time.perf_counter() - started) * 1000.0

        admissible, admissible_ms = score("baseline_admissible")
        survivor, survivor_ms = score("baseline_admissible_survivor_pdb")
        patterns = getattr(heuristic, "_survivor_pdb_patterns_used", 0)
        flag = "" if survivor <= admissible + 1e-9 else "  <-- ABOVE admissible!"
        print(
            f"{deadline:>9} {admissible:>12.6f} {survivor:>13.6f} "
            f"{admissible - survivor:>9.6f} {admissible_ms:>8.1f} {survivor_ms:>8.1f}"
            f"  patterns={patterns}{flag}"
        )


if __name__ == "__main__":
    main()
