"""
Inspect TP-MCTS tree growth one selection iteration at a time.

This is a diagnostic tool, not an experiment runner.  It builds one TP-MCTS
root for a single scenario, optionally runs a fixed number of tree iterations,
and prints root-action statistics after each requested milestone.

Example:
  python scripts/inspect_mcts_tree.py ^
    --domain nasa_rover --object_amount 2 --deadline 25 ^
    --heuristic atomic_exact --selection_type max --value_mode greedy_matched ^
    --search_depth 0 --k 9999 --exploration_constant 0 ^
    --discount_factor 1 --step_penalty -0.01 ^
    --milestones 0 1 2 5 10 --top_n 12

Compare baseline avg against top-K UCT:
  python scripts/inspect_mcts_tree.py --selection_type avg --seed 123 --milestones 0 1 2 5 10
  python scripts/inspect_mcts_tree.py --selection_type avg_topk --uct_initial_k 3 --seed 123 --milestones 0 1 2 5 10
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from comdp_plus_no_deadline.engines.frontier_aligned_option_a import (
    is_option_a_strategy,
    option_a_ptrpg_suffix,
)
from experiment_common import HEURISTIC_ALIASES

# The local unified_planning package parses sys.argv at import time.  Hide this
# script's diagnostic flags during import, then restore them for our own parser.
_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
try:
    import unified_planning as up
    import unified_planning.domains
    from unified_planning.engines.solvers.greedy_parallel import greedy_matched_value_target
    from unified_planning.engines.solvers.mcts import C_MCTS
    from unified_planning.shortcuts import Convert_problem, Grounder, MDP, create_init_stn
finally:
    sys.argv = _ORIGINAL_ARGV


DOMAINS = {
    "machine_shop": up.domains.Machine_Shop,
    "nasa_rover": up.domains.Nasa_Rover,
    "stuck_car_1o": up.domains.Stuck_Car_1o,
    "stuck_car": up.domains.Stuck_Car,
    "conc": up.domains.Conc,
    "full_conc": up.domains.Full_Conc,
    "prob_conc": up.domains.Prob_Conc,
    "best_no_parallel": up.domains.Best_No_Parallel,
    "simple": up.domains.Simple,
    "hosting": up.domains.Hosting,
    "prob_match_cellar": up.domains.Prob_MatchCellar,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grow one TP-MCTS tree by fixed iterations and print root diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--domain", default="nasa_rover", choices=sorted(DOMAINS))
    p.add_argument("--domain_type", default="regular")
    p.add_argument("--object_amount", type=int, default=2)
    p.add_argument("--garbage_amount", type=int, default=0)
    p.add_argument("--deadline", type=int, default=25)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--search_depth", type=int, default=0)
    p.add_argument(
        "--selection_type",
        choices=["avg", "avg_topk", "avg_pw", "max", "rootInterval"],
        default="max",
    )
    p.add_argument(
        "--uct_initial_k",
        type=int,
        default=3,
        help="Top-K window used by avg_topk and as the initial width for avg_pw.",
    )
    p.add_argument("--k", type=int, default=9999)
    p.add_argument("--exploration_constant", "--C", type=float, default=0.0)
    p.add_argument("--discount_factor", "--gamma", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=-0.01)
    p.add_argument("--reward_mode", choices=["deadline", "terminal"], default="deadline")
    p.add_argument(
        "--value_mode",
        choices=[
            "tp_mcts",
            "greedy_matched",
            "ptrpg_guided_terminal_rollout",
            "fixed_tail_ptrpg_rollout",
        ],
        default="greedy_matched",
        help=(
            "MCTS leaf/backup target: tp_mcts (PTRPG heuristic), greedy_matched, "
            "ptrpg_guided_terminal_rollout, or fixed_tail_ptrpg_rollout."
        ),
    )
    p.add_argument(
        "--ptrpg-guided-rollout-policy",
        dest="ptrpg_guided_rollout_policy",
        default=None,
        choices=(
            "baseline_survival_resolution",
            "atomic_exact_resolution",
            "atom_backtrack_exact_resolution",
        ),
        help="PTRPG policy for ptrpg_guided_terminal_rollout (defaults from heuristic alias).",
    )
    p.add_argument(
        "--ptrpg-guided-rollout-max-steps",
        dest="ptrpg_guided_rollout_max_steps",
        type=int,
        default=None,
        help="Safety cap on MDP transitions in rollout (default: 32×90).",
    )
    p.add_argument(
        "--ptrpg-guided-rollout-epsilon",
        dest="ptrpg_guided_rollout_epsilon",
        type=float,
        default=None,
        help="Rollout epsilon (0 = greedy).",
    )
    p.add_argument(
        "--ptrpg-guided-rollout-debug",
        dest="ptrpg_guided_rollout_debug",
        action="store_true",
        help="Log the first rollout per search.",
    )
    p.add_argument(
        "--fixed-tail-h",
        dest="fixed_tail_h",
        type=int,
        default=None,
        help="fixed_tail_ptrpg_rollout: common PTRPG suffix horizon H (default 10 if omitted).",
    )
    p.add_argument(
        "--fixed-tail-debug",
        dest="fixed_tail_debug",
        action="store_true",
        help="Log the first 5 fixed-tail leaf evaluations per MCTS search.",
    )
    p.add_argument(
        "--heuristic",
        default="atomic_exact",
        choices=sorted(HEURISTIC_ALIASES),
        help="User-facing heuristic key from experiment_common.py.",
    )
    p.add_argument(
        "--heuristic_depth",
        type=int,
        default=None,
        help="Defaults to the deadline, matching the experiment scripts.",
    )
    p.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=[0, 1, 2, 5, 10],
        help="Cumulative selection-iteration counts to print.",
    )
    p.add_argument("--top_n", type=int, default=12)
    p.add_argument(
        "--expected",
        action="store_true",
        help="Also print the one-step expected greedy-matched target for visible rows.",
    )
    p.add_argument(
        "--resolution-alpha",
        type=float,
        default=None,
        help=(
            "Override unified_planning.args.resolution_alpha for atom_backtrack_exact_resolution "
            "(omit flag to keep the value from unified_planning import-time defaults)."
        ),
    )
    p.add_argument(
        "--resolution-forced-minimum",
        action="store_true",
        help="Set unified_planning.args.resolution_forced_minimum for atom_backtrack_exact_resolution.",
    )
    p.add_argument(
        "--rollout-aligned-h",
        type=int,
        default=None,
        help="rollout_aligned_* / frontier_aligned_* / Option A: common suffix horizon H.",
    )
    p.add_argument(
        "--rollout-aligned-redo",
        type=int,
        default=None,
        help="Prefix rollouts averaged per aligned node (Option A uses this).",
    )
    p.add_argument(
        "--rollout-aligned-boundary-mode",
        default=None,
        help="Prefix rollout boundary: overshoot | wait_no_overshoot | expected_stochastic_rounding.",
    )
    p.add_argument(
        "--rollout-aligned-lambda-align",
        type=float,
        default=None,
        help="frontier_aligned_* (non-Option-A): blend aligned value with UCT Q.",
    )
    p.add_argument(
        "--rollout-aligned-fixed-h",
        action="store_true",
        help="rollout_aligned_* only: use fixed ROLLOUT_ALIGNED_H instead of dynamic H.",
    )
    return p.parse_args(argv)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def build_mdp(args: argparse.Namespace) -> MDP:
    domain_cls = DOMAINS[args.domain]
    model = domain_cls(
        kind=args.domain_type,
        deadline=args.deadline,
        object_amount=args.object_amount,
        garbage_amount=args.garbage_amount,
    )

    grounder = Grounder(model.grounding_map()) if args.domain == "nasa_rover" else Grounder()
    grounded = grounder._compile(model.problem).problem
    converted = Convert_problem(grounded)._converted_problem
    return MDP(
        converted,
        discount_factor=args.discount_factor,
        reward_mode=args.reward_mode,
        step_penalty=args.step_penalty,
    )


def apply_heuristic_alias_overrides(args: argparse.Namespace) -> None:
    """Resolve value_mode / PTRPG rollout policy from experiment_common aliases."""
    alias = HEURISTIC_ALIASES[args.heuristic]
    args.value_mode = alias.get("value_mode", args.value_mode)
    if args.ptrpg_guided_rollout_policy is None:
        args.ptrpg_guided_rollout_policy = alias.get("ptrpg_guided_rollout_policy")
    if getattr(args, "fixed_tail_h", None) is None and "fixed_tail_h" in alias:
        args.fixed_tail_h = int(alias["fixed_tail_h"])


def strategy_for_greedy_matched_expected(strategy: str) -> str:
    """Map MCTS-only strategy names to a PTRPG strategy greedy_matched can evaluate."""
    if is_option_a_strategy(strategy):
        return option_a_ptrpg_suffix(strategy)
    return strategy


def configure_rollout_aligned_cli(args: argparse.Namespace) -> None:
    """Mirror run_domain / experiment alignment flags on up.args."""
    a = getattr(up, "args", None)
    if a is None:
        return
    if args.rollout_aligned_h is not None:
        a.rollout_aligned_h = int(args.rollout_aligned_h)
    if args.rollout_aligned_redo is not None:
        a.rollout_aligned_redo = int(args.rollout_aligned_redo)
    if args.rollout_aligned_boundary_mode is not None:
        a.rollout_aligned_boundary_mode = str(args.rollout_aligned_boundary_mode)
    if args.rollout_aligned_lambda_align is not None:
        a.rollout_aligned_lambda_align = float(args.rollout_aligned_lambda_align)
    if args.rollout_aligned_fixed_h:
        a.rollout_aligned_fixed_h = True


def configure_ptrpg_rollout_cli(args: argparse.Namespace) -> None:
    """Mirror run_domain parser fields on up.args for direct C_MCTS construction."""
    if args.ptrpg_guided_rollout_policy is not None:
        up.args.ptrpg_guided_rollout_policy = args.ptrpg_guided_rollout_policy
    if args.ptrpg_guided_rollout_max_steps is not None:
        up.args.ptrpg_guided_rollout_max_steps = args.ptrpg_guided_rollout_max_steps
    if args.ptrpg_guided_rollout_epsilon is not None:
        up.args.ptrpg_guided_rollout_epsilon = args.ptrpg_guided_rollout_epsilon
    if args.ptrpg_guided_rollout_debug:
        up.args.ptrpg_guided_rollout_debug = True


def configure_fixed_tail_cli(args: argparse.Namespace) -> None:
    """Mirror fixed-tail flags on up.args for direct C_MCTS construction."""
    a = getattr(up, "args", None)
    if a is None:
        return
    if getattr(args, "fixed_tail_h", None) is not None:
        a.fixed_tail_h = int(args.fixed_tail_h)
    if getattr(args, "fixed_tail_debug", False):
        a.fixed_tail_debug = True


def build_mcts(args: argparse.Namespace) -> C_MCTS:
    mdp = build_mdp(args)
    alias = HEURISTIC_ALIASES[args.heuristic]
    depth = args.heuristic_depth if args.heuristic_depth is not None else args.deadline
    return C_MCTS(
        mdp=mdp,
        root_node=None,
        root_state=mdp.initial_state(),
        search_depth=args.search_depth,
        exploration_constant=args.exploration_constant,
        stn=create_init_stn(mdp),
        selection_type=args.selection_type,
        k=args.k,
        uct_initial_k=args.uct_initial_k,
        heuristic_name=alias["heuristic_name"],
        temporal_heuristic_depth=depth,
        temporal_heuristic_strategy=alias["temporal_heuristic_strategy"],
        value_mode=args.value_mode,
    )


def run_one_iteration(mcts: C_MCTS, selection_type: str) -> float:
    if selection_type in {"avg", "avg_topk", "avg_pw"}:
        return mcts.selection(mcts.root_node)
    if selection_type == "max":
        return mcts.selection_max(mcts.root_node)
    reward, *_ = mcts.selection_root_interval(mcts.root_node)
    return reward


def subtree_stats_from_snode(snode: Any) -> dict[str, int]:
    state_nodes = 1
    action_nodes = len(snode.children)
    max_depth = int(snode.depth)
    expanded_actions = 0

    for anode in snode.children.values():
        if anode.children:
            expanded_actions += 1
        for child_snode in anode.children.values():
            child = subtree_stats_from_snode(child_snode)
            state_nodes += child["state_nodes"]
            action_nodes += child["action_nodes"]
            expanded_actions += child["expanded_actions"]
            max_depth = max(max_depth, child["max_depth"])

    return {
        "state_nodes": state_nodes,
        "action_nodes": action_nodes,
        "expanded_actions": expanded_actions,
        "max_depth": max_depth,
    }


def subtree_stats_from_anode(anode: Any) -> dict[str, int]:
    state_nodes = 0
    action_nodes = 0
    expanded_actions = 0
    max_depth = int(anode.parent.depth)

    for child_snode in anode.children.values():
        child = subtree_stats_from_snode(child_snode)
        state_nodes += child["state_nodes"]
        action_nodes += child["action_nodes"]
        expanded_actions += child["expanded_actions"]
        max_depth = max(max_depth, child["max_depth"])

    return {
        "state_nodes": state_nodes,
        "action_nodes": action_nodes,
        "expanded_actions": expanded_actions,
        "max_depth": max_depth,
    }


def transition_terminal_probability(mdp: MDP, state: Any, action: Any) -> float:
    total = 0.0
    for next_state, prob in mdp.transition_function(state, action):
        if mdp.is_terminal(next_state):
            total += prob
    return total


def expected_target(mcts: C_MCTS, action: Any, args: argparse.Namespace) -> float:
    alias = HEURISTIC_ALIASES[args.heuristic]
    anode = mcts.root_node.children[action]
    current_time = anode.stn.get_current_end_time()
    strategy = strategy_for_greedy_matched_expected(alias["temporal_heuristic_strategy"])
    return greedy_matched_value_target(
        mdp=mcts.mdp,
        state=mcts.root_node.state,
        action=action,
        current_time=current_time,
        heuristic_name=alias["heuristic_name"],
        temporal_heuristic_depth=args.heuristic_depth if args.heuristic_depth is not None else args.deadline,
        temporal_heuristic_strategy=strategy,
    )


def uct_score(root: Any, anode: Any, exploration_constant: float) -> float | None:
    if anode.count == 0:
        return None
    if root.count <= 0:
        return anode.value
    return anode.value + exploration_constant * math.sqrt(math.log(root.count) / anode.count)


def next_uct_action_name(mcts: C_MCTS, exploration_constant: float) -> str:
    root = mcts.root_node
    if len(root.possible_actions) == 0:
        return "<none>"
    return mcts.uct(root, exploration_constant).name


def root_rows(mcts: C_MCTS, args: argparse.Namespace) -> list[dict[str, Any]]:
    root = mcts.root_node
    rows = []
    for action in root.possible_actions:
        anode = root.children[action]
        stats = subtree_stats_from_anode(anode)
        rows.append(
            {
                "action": action,
                "name": action.name,
                "visits": int(anode.count),
                "value": float(anode.value),
                "uct": uct_score(root, anode, args.exploration_constant),
                "state_children": len(anode.children),
                "subtree_states": stats["state_nodes"],
                "subtree_actions": stats["action_nodes"],
                "max_depth": stats["max_depth"],
                "heuristic_only": anode.count > 0 and not anode.children,
                "terminal_prob": transition_terminal_probability(mcts.mdp, root.state, action),
                "expected": None,
            }
        )

    rows.sort(key=lambda row: (row["value"], row["visits"]), reverse=True)
    visible = rows[: args.top_n]
    if args.expected:
        for row in visible:
            row["expected"] = expected_target(mcts, row["action"], args)
    return visible


def best_action_name(mcts: C_MCTS) -> str:
    best_action = None
    best_value = -math.inf
    for action in mcts.root_node.possible_actions:
        anode = mcts.root_node.children[action]
        if anode.count > 0 and anode.value > best_value:
            best_value = anode.value
            best_action = action
    if best_action is None:
        return "<none>"
    return best_action.name


def fmt_float(value: float | None, width: int = 9) -> str:
    if value is None:
        return "unvisited".rjust(width)
    if not math.isfinite(value):
        return str(value).rjust(width)
    return f"{value:{width}.4f}"


def print_snapshot(mcts: C_MCTS, args: argparse.Namespace, milestone: int) -> None:
    root = mcts.root_node
    all_anodes = list(root.children.values())
    unvisited = sum(1 for anode in all_anodes if anode.count == 0)
    heuristic_only = sum(1 for anode in all_anodes if anode.count > 0 and not anode.children)
    expanded = sum(1 for anode in all_anodes if anode.children)
    stats = subtree_stats_from_snode(root)
    best = best_action_name(mcts)
    next_uct = next_uct_action_name(mcts, args.exploration_constant)
    visited_values = [anode.value for anode in all_anodes if anode.count > 0]
    value_spread = (
        max(visited_values) - min(visited_values)
        if visited_values
        else 0.0
    )

    print()
    print("=" * 110)
    print(
        f"after_iterations={milestone}  root_visits={int(root.count)}  "
        f"best_action={best}  next_uct_action={next_uct}"
    )
    print(
        f"root_actions={len(all_anodes)}  unvisited={unvisited}  "
        f"heuristic_only={heuristic_only}  expanded={expanded}  "
        f"tree_state_nodes={stats['state_nodes']}  tree_action_nodes={stats['action_nodes']}  "
        f"max_depth={stats['max_depth']}  visited_value_spread={value_spread:.6f}"
    )
    print("-" * 110)

    expected_header = " expected" if args.expected else ""
    print(
        f"{'#':>2} {'action':<44} {'visits':>6} {'value':>9} {'uct':>9}"
        f"{expected_header:>10} {'childS':>6} {'subS':>6} {'maxD':>5} {'termP':>7} flag"
    )
    for idx, row in enumerate(root_rows(mcts, args), start=1):
        flag = "heuristic-only" if row["heuristic_only"] else ""
        name = row["name"][:44]
        expected_text = f"{row['expected']:9.4f}" if args.expected else ""
        print(
            f"{idx:>2} {name:<44} {row['visits']:>6} "
            f"{fmt_float(row['value'])} {fmt_float(row['uct'])}"
            f"{expected_text:>10} {row['state_children']:>6} "
            f"{row['subtree_states']:>6} {row['max_depth']:>5} "
            f"{row['terminal_prob']:>7.3f} {flag}"
        )

    best_row = next((row for row in root_rows(mcts, args) if row["name"] == best), None)
    if best_row and best_row["heuristic_only"]:
        print()
        print(
            "[diagnostic] Current returned action is heuristic-only: it has a positive visit "
            "count from initialization, but no expanded successor state under the root."
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    apply_heuristic_alias_overrides(args)
    configure_ptrpg_rollout_cli(args)
    configure_fixed_tail_cli(args)
    configure_rollout_aligned_cli(args)
    if args.resolution_alpha is not None:
        up.args.resolution_alpha = args.resolution_alpha
    if args.resolution_forced_minimum:
        up.args.resolution_forced_minimum = True
    set_seed(args.seed)
    milestones = sorted(set(max(0, m) for m in args.milestones))

    mcts = build_mcts(args)
    tail_h = getattr(args, "fixed_tail_h", None)
    if args.value_mode == "fixed_tail_ptrpg_rollout" and tail_h is None:
        tail_h = getattr(getattr(up, "args", None), "fixed_tail_h", 10)
    print(
        "TP-MCTS tree inspection: "
        f"domain={args.domain} obj={args.object_amount} deadline={args.deadline} "
        f"heuristic={args.heuristic} selection={args.selection_type} "
        f"value_mode={args.value_mode} depth={args.search_depth} "
        f"k={args.k} uct_initial_k={args.uct_initial_k} C={args.exploration_constant} seed={args.seed}"
        + (f" fixed_tail_h={tail_h}" if args.value_mode == "fixed_tail_ptrpg_rollout" else "")
    )
    print(
        f"Milestones (cumulative selection iterations): {milestones}  top_n={args.top_n}"
    )
    if args.expected:
        strat = HEURISTIC_ALIASES[args.heuristic]["temporal_heuristic_strategy"]
        if is_option_a_strategy(strat):
            suffix = option_a_ptrpg_suffix(strat)
            print(
                "[note] --expected column is greedy_matched one-step score using PTRPG "
                f"suffix {suffix!r}; it is not the global Option-A aligned selection value."
            )

    completed = 0
    for milestone in milestones:
        while completed < milestone:
            run_one_iteration(mcts, args.selection_type)
            completed += 1
        print_snapshot(mcts, args, milestone)


if __name__ == "__main__":
    main()
