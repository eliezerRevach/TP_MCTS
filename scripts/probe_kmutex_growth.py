"""
Probe whether the mutex-aware K-bounded OR layer (baseline_admissible_kmutex)
catches/keeps any mutex on a given domain — the "does the table grow?" check.

Builds the grounded+converted problem exactly like run_domain.run_regular, then
runs ONE forward propagation of the kmutex strategy from the initial state at the
requested depth and prints the survival instrumentation. It also compares against
baseline_admissible so you can see whether the bound was tightened anywhere.

Usage:
    python scripts/probe_kmutex_growth.py --domain nasa_rover --object_amount 2 --deadline 25
    python scripts/probe_kmutex_growth.py --domain machine_shop --object_amount 2 --deadline 25 -k 3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unified_planning as up
from unified_planning.shortcuts import *  # noqa: F401,F403  (matches run_domain import order)
import unified_planning.domains  # noqa: F401
from unified_planning.engines.convert_problem import Convert_problem
from unified_planning.engines.mdp import MDP
from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)

domains = {
    "nasa_rover": up.domains.Nasa_Rover,
    "machine_shop": up.domains.Machine_Shop,
    "stuck_car": up.domains.Stuck_Car,
}


def build_converted_problem(domain, deadline, object_amount, garbage_amount, domain_type):
    model = domains[domain](
        kind=domain_type,
        deadline=deadline,
        object_amount=object_amount,
        garbage_amount=garbage_amount,
    )
    if domain == "nasa_rover":
        grounder = up.engines.compilers.Grounder(model.grounding_map())
    else:
        grounder = up.engines.compilers.Grounder()
    ground_problem = grounder._compile(model.problem).problem
    converted = Convert_problem(ground_problem)._converted_problem
    return converted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="nasa_rover")
    ap.add_argument("--object_amount", type=int, default=2)
    ap.add_argument("--deadline", type=int, default=25)
    ap.add_argument("--garbage_amount", type=int, default=10)
    ap.add_argument("--domain_type", default="regular")
    ap.add_argument("--depth", type=int, default=None, help="propagation depth (default: deadline)")
    ap.add_argument("-k", "--kmutex_k", type=int, default=3)
    ap.add_argument(
        "--mutex_mode",
        default="exec",
        choices=["exec", "all"],
        help="exec = certified execution mutex (delete-interference + shared "
        "consumable precondition); all = also shared add-effect (degenerate)",
    )
    args = ap.parse_args()

    os.environ["TP_MCTS_KMUTEX_K"] = str(args.kmutex_k)
    os.environ["TP_MCTS_KMUTEX_MUTEX_MODE"] = str(args.mutex_mode)

    depth = args.depth if args.depth is not None else args.deadline
    print(
        f"domain={args.domain} object_amount={args.object_amount} "
        f"deadline={args.deadline} depth={depth} K={args.kmutex_k} "
        f"mutex_mode={args.mutex_mode}",
        flush=True,
    )

    converted = build_converted_problem(
        args.domain, args.deadline, args.object_amount, args.garbage_amount, args.domain_type
    )
    mdp = MDP(converted, discount_factor=1.0, reward_mode="terminal", step_penalty=0.0)
    state = mdp.initial_state()
    print(
        f"actions={len(converted.actions)} goals={len(converted.goals)} "
        f"initial_true_facts={len(state.predicates)}",
        flush=True,
    )

    heuristic = TemporalProbabilisticRPGHeuristic.from_problem(converted)
    print(f"kmutex K in use = {heuristic._kmutex_k}", flush=True)

    admissible = heuristic.heuristic_score(
        state, converted.goals, fixed_depth=depth, strategy="baseline_admissible"
    )
    kmutex = heuristic.heuristic_score(
        state, converted.goals, fixed_depth=depth, strategy="baseline_admissible_kmutex"
    )

    print("\n=== per-cell kmutex table growth (one forward propagation) ===", flush=True)
    print(heuristic.log_kmutex_summary(), flush=True)
    print(
        f"baseline_admissible score = {admissible:.6f}   "
        f"baseline_admissible_kmutex score = {kmutex:.6f}   "
        f"(delta = {kmutex - admissible:+.6f})",
        flush=True,
    )

    paths = heuristic.heuristic_score(
        state, converted.goals, fixed_depth=depth, strategy="baseline_admissible_paths"
    )
    print("\n=== temporal path-mutex growth (one forward propagation) ===", flush=True)
    print(heuristic.log_pathmutex_summary(), flush=True)
    print(
        f"baseline_admissible_paths score = {paths:.6f}", flush=True,
    )

    kinstr = heuristic._kmutex_instr
    pinstr = heuristic._pathmutex_instr
    print("\n[result] summary", flush=True)
    print(
        f"  per-cell kmutex : {kinstr.or_nodes_clique_survived} clique(s) survived "
        f"of {kinstr.or_nodes_total} OR-nodes (tightened<union at "
        f"{kinstr.tightened_below_union}).",
        flush=True,
    )
    print(
        f"  temporal paths  : {pinstr.or_nodes_clique_survived} clique(s) survived "
        f"of {pinstr.or_nodes_total} OR-nodes "
        f"(survival={pinstr.clique_survival_fraction:.3f}, "
        f"AND_blocked={pinstr.and_paths_blocked}).",
        flush=True,
    )
    if pinstr.or_nodes_clique_survived > kinstr.or_nodes_clique_survived:
        print(
            "  => the TEMPORAL path-mutex caught resource/self-mutex that the "
            "per-cell bound missed (overlapping re-fires collapse via max).",
            flush=True,
        )


if __name__ == "__main__":
    main()
