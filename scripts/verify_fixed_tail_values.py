"""Quick check: fixed-tail leaf eval and MCTS backups are not all zero."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import unified_planning as up
from experiment_common import HEURISTIC_ALIASES
from inspect_mcts_tree import build_mdp, set_seed
from unified_planning.engines.utils import create_init_stn
from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
    fixed_tail_config_from_args,
    fixed_tail_ptrpg_value,
)
from unified_planning.engines.solvers.mcts import C_MCTS
from unified_planning.engines.solvers.ptrpg_guided_rollout import remaining_deadline


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="machine_shop")
    p.add_argument("--object_amount", type=int, default=2)
    p.add_argument("--deadline", type=int, default=25)
    p.add_argument("--fixed_tail_h", type=int, default=23)
    p.add_argument("--search_depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--iterations", type=int, default=50)
    args = p.parse_args()

    set_seed(args.seed)
    up.args = argparse.Namespace(
        fixed_tail_h=args.fixed_tail_h,
        resolution_alpha=2.0,
        resolution_forced_minimum=False,
        resolution_reference_t=None,
    )

    alias = HEURISTIC_ALIASES["fixed_tail_atomic_exact_resolution"]
    ns = argparse.Namespace(
        domain=args.domain,
        domain_type="combination",
        object_amount=args.object_amount,
        garbage_amount=0,
        deadline=args.deadline,
        discount_factor=1.0,
        step_penalty=0.0,
        reward_mode="terminal",
    )
    mdp = build_mdp(ns)
    stn = create_init_stn(mdp)
    state = mdp.initial_state()
    cfg = fixed_tail_config_from_args()
    strat = alias["temporal_heuristic_strategy"]

    R = remaining_deadline(mdp, stn)
    root_val = fixed_tail_ptrpg_value(
        mdp, state, stn, None, cfg, tail_strategy=strat
    )
    print(f"Root R={R} H={cfg.fixed_tail_h} direct fixed_tail_ptrpg_value={root_val:.6f}")

    mcts = C_MCTS(
        mdp,
        None,
        state,
        args.search_depth,
        10.0,
        stn,
        "avg",
        10,
        previous_chosen_action_node=None,
        heuristic_name=alias["heuristic_name"],
        temporal_heuristic_strategy=strat,
        value_mode="fixed_tail_ptrpg_rollout",
    )
    for _ in range(args.iterations):
        mcts.search(1, "avg", "q")

    vals = []
    for action, anode in mcts.root_node.children.items():
        # Node.update stores the running mean in anode.value — do not divide by count again.
        backup = float(anode.value)
        vals.append(backup)
        print(
            f"  {getattr(action, 'name', action)}: visits={int(anode.count)} "
            f"backup_mean={backup:.6f}"
        )

    spread = max(vals) - min(vals) if vals else 0.0
    nonzero = sum(1 for v in vals if abs(v) > 1e-6)
    print(f"visited_value_spread={spread:.6f}  nonzero_children={nonzero}/{len(vals)}")

    ok = root_val > 1e-6 and nonzero > 0 and spread > 1e-6
    print("PASS" if ok else "FAIL (need root eval > 0 and differing nonzero backups)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
