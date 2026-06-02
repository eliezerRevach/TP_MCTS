"""Quick check: fixed-tail prefix-frac bootstrap and MCTS backups are sensible."""
from __future__ import annotations

import argparse
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
    build_fixed_tail_search_context,
    crossed_cutoff,
    elapsed_from_root,
    fixed_tail_bootstrap_value,
    fixed_tail_config_from_args,
    node_remaining,
)
from unified_planning.engines.solvers.mcts import C_MCTS


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="machine_shop")
    p.add_argument("--object_amount", type=int, default=2)
    p.add_argument("--deadline", type=int, default=25)
    p.add_argument("--fixed_tail_prefix_frac", type=float, default=0.10)
    p.add_argument("--search_depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--iterations", type=int, default=30)
    args = p.parse_args()

    set_seed(args.seed)
    up.args = argparse.Namespace(
        fixed_tail_prefix_frac=args.fixed_tail_prefix_frac,
        fixed_tail_debug=False,
        resolution_alpha=2.0,
        resolution_forced_minimum=False,
        resolution_reference_t=None,
    )

    alias = HEURISTIC_ALIASES["fixed_tail_atomic_exact_resolution"]
    strat = alias["temporal_heuristic_strategy"]
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
    ctx = build_fixed_tail_search_context(mdp, state, stn, cfg)

    root_rem = node_remaining(mdp, state, stn)
    root_val = fixed_tail_bootstrap_value(mdp, state, stn, strat, ctx=ctx)
    print(
        f"Root remaining={root_rem} prefix_frac={cfg.prefix_frac} "
        f"prefix_budget={ctx.prefix_budget} bootstrap={root_val:.6f}"
    )

    mcts = C_MCTS(
        mdp,
        None,
        state,
        args.search_depth,
        10.0,
        stn,
        "avg",
        10,
        heuristic_name=alias["heuristic_name"],
        temporal_heuristic_strategy=strat,
        value_mode="fixed_tail_ptrpg_rollout",
    )
    assert mcts._fixed_tail_ctx.prefix_budget == ctx.prefix_budget

    for _ in range(args.iterations):
        mcts.search(1, "avg", "q")

    vals = []
    for action, anode in mcts.root_node.children.items():
        backup = float(anode.value)
        vals.append(backup)
        print(
            f"  {getattr(action, 'name', action)}: visits={int(anode.count)} "
            f"backup_mean={backup:.6f}"
        )

    spread = max(vals) - min(vals) if vals else 0.0
    nonzero = sum(1 for v in vals if abs(v) > 1e-6)
    print(f"visited_value_spread={spread:.6f}  nonzero_children={nonzero}/{len(vals)}")

    ok = root_val > 1e-6 and nonzero > 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
