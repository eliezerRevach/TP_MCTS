"""
Heuristic Per-Call Runtime Benchmark
======================================

Measures the cost of each heuristic call using the built-in timing hooks
from ``unified_planning.engines.heuristic_timing``.

Runs ``greedy_parallel`` on a single scenario for each heuristic variant
and reports:

  wrapper_first_call_sec    -- wall time for the first _heuristic_value() call
  wrapper_avg_call_sec      -- total wrapper time / total wrapper calls
  worker_first_call_sec     -- wall time for the first heuristic_score() call
  worker_avg_call_sec       -- total worker time / total worker calls
  worker_cache_hit_avg_sec  -- avg time for cache-hit worker calls
  worker_cache_miss_avg_sec -- avg time for cache-miss worker calls
  worker_cache_hits         -- number of cache hits
  worker_cache_misses       -- number of cache misses

Heuristics benchmarked by default
-----------------------------------
  ptrpg_old           -- classical PTRPG / trpg (from TP-MCTS paper)
  baseline            -- temporal_probabilistic_rpg / baseline
  baseline_cached     -- temporal_probabilistic_rpg / baseline_cached
  atomic_exact        -- temporal_probabilistic_rpg / atom_backtrack_exact
  atomic_exact_cached -- temporal_probabilistic_rpg / atom_backtrack_cached
  fast_atom_cache     -- temporal_probabilistic_rpg / fast_atom_cache

Default scenario
-----------------
  Domain  : nasa_rover
  Objects : 2
  Deadline: 25
  Heuristic depth: same as deadline when ``--heuristic_depth`` is omitted
  Max steps: 90

Output
------
  results/heuristic_runtime_per_call.csv   (overwritten each run)
  Terminal report after each heuristic + ranking table at the end.

Usage (Colab or local)
-----------------------
  # From the repo root:
  python scripts/run_heuristic_runtime_per_call.py

  # Custom settings:
  python scripts/run_heuristic_runtime_per_call.py \\
      --domain nasa_rover \\
      --object_amount 3 \\
      --deadline 35 \\
      --heuristic_depth 35 \\
      --max_steps 90 \\
      --seed 42 \\
      --heuristics ptrpg_old baseline baseline_cached atomic_exact atomic_exact_cached \\
      --output results/my_timing.csv

Notes
-----
- Worker-level timing (cache hit/miss) is only available for
  temporal_probabilistic_rpg strategies; ptrpg_old shows wrapper timing only.
- Each heuristic variant gets a fresh MDP so caches do not bleed between runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.argv must be cleared BEFORE importing unified_planning because
# unified_planning/parser.py calls argparse.parse_args() at module level.
# We save the original argv so parse_args() below can still see CLI flags
# (e.g. --help, --domain, etc.) after UP is imported.
# ---------------------------------------------------------------------------
_original_argv = sys.argv[:]
sys.argv = [_original_argv[0]]  # empty-ish so UP parser sees no unknown flags

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

import random
import numpy as np

import unified_planning as up
import unified_planning.domains  # ensure up.domains.* is accessible
from unified_planning.shortcuts import *  # noqa: F401,F403
from unified_planning.engines.convert_problem import Convert_problem
from unified_planning.engines.mdp import MDP
from unified_planning.engines.heuristic_timing import reset_metrics, get_metrics
import unified_planning.engines.solvers.greedy_parallel as gp_solver

from experiment_common import (
    ALL_HEURISTICS,
    DEFAULT_HEURISTICS_RUNTIME,
    HEURISTIC_ALIASES,
    REPO_ROOT,
    print_summary_table,
    validate_heuristics,
    write_csv,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DOMAIN = "nasa_rover"
DEFAULT_OBJECT_AMOUNT = 2
DEFAULT_DEADLINE = 25
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 90
# When --heuristic_depth is omitted, use the same value as --deadline (see main()).
DEFAULT_HEURISTIC_DEPTH_FALLBACK = 25
DEFAULT_REWARD_MODE = "deadline"
DEFAULT_DISCOUNT_FACTOR = 0.95
DEFAULT_STEP_PENALTY = -0.05
DEFAULT_OUTPUT = str(REPO_ROOT / "results" / "heuristic_runtime_per_call.csv")

CSV_FIELDNAMES = [
    "heuristic",
    "heuristic_label",
    "heuristic_name_internal",
    "strategy_internal",
    "domain",
    "object_amount",
    "deadline",
    "reward_mode",
    "discount_factor",
    "step_penalty",
    "seed",
    "heuristic_depth",
    "max_steps",
    "plan_success",
    "plan_makespan",
    "wrapper_total_calls",
    "wrapper_total_time_sec",
    "wrapper_first_call_sec",
    "wrapper_avg_call_sec",
    "worker_total_calls",
    "worker_total_time_sec",
    "worker_first_call_sec",
    "worker_avg_call_sec",
    "worker_cache_hits",
    "worker_cache_misses",
    "worker_cache_hit_avg_sec",
    "worker_cache_miss_avg_sec",
]

RANKING_COLUMNS = [
    "heuristic",
    "wrapper_avg_call_sec",
    "worker_avg_call_sec",
    "worker_cache_hit_avg_sec",
    "worker_cache_miss_avg_sec",
    "worker_cache_hits",
    "worker_cache_misses",
    "plan_success",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark heuristic per-call runtime using greedy_parallel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--heuristics",
        nargs="+",
        default=DEFAULT_HEURISTICS_RUNTIME,
        metavar="H",
        help=(
            "Heuristics to benchmark. "
            f"Choices: {', '.join(ALL_HEURISTICS)}. "
            f"Default: {' '.join(DEFAULT_HEURISTICS_RUNTIME)}"
        ),
    )
    p.add_argument(
        "--domain",
        default=DEFAULT_DOMAIN,
        help=f"Domain name. Default: {DEFAULT_DOMAIN}",
    )
    p.add_argument(
        "--object_amount",
        type=int,
        default=DEFAULT_OBJECT_AMOUNT,
        help=f"Number of objects. Default: {DEFAULT_OBJECT_AMOUNT}",
    )
    p.add_argument(
        "--deadline",
        type=int,
        default=DEFAULT_DEADLINE,
        help=f"Problem deadline. Default: {DEFAULT_DEADLINE}",
    )
    p.add_argument(
        "--heuristic_depth",
        type=int,
        default=None,
        help=(
            "Temporal heuristic lookahead depth. "
            "Default: same as --deadline (or "
            f"{DEFAULT_HEURISTIC_DEPTH_FALLBACK} if deadline is unset)."
        ),
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Max plan steps (greedy_parallel horizon). Default: {DEFAULT_MAX_STEPS}",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )
    p.add_argument(
        "--reward_mode",
        choices=["deadline", "terminal"],
        default=DEFAULT_REWARD_MODE,
        help=f"MDP reward mode. Default: {DEFAULT_REWARD_MODE}",
    )
    p.add_argument(
        "--discount_factor",
        "--gamma",
        type=float,
        default=DEFAULT_DISCOUNT_FACTOR,
        dest="discount_factor",
        help=f"MDP discount factor. Default: {DEFAULT_DISCOUNT_FACTOR}",
    )
    p.add_argument(
        "--step_penalty",
        type=float,
        default=DEFAULT_STEP_PENALTY,
        help=f"Per-step MDP reward penalty. Default: {DEFAULT_STEP_PENALTY}",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Domain builders
# ---------------------------------------------------------------------------

_DOMAIN_CLASS_MAP = {
    "nasa_rover": up.domains.Nasa_Rover,
    "machine_shop": up.domains.Machine_Shop,
    "stuck_car": up.domains.Stuck_Car,
    "stuck_car_1o": up.domains.Stuck_Car_1o,
    "conc": up.domains.Conc,
    "prob_conc": up.domains.Prob_Conc,
}


def build_mdp(
    domain: str,
    object_amount: int,
    deadline: int,
    seed: int,
    reward_mode: str = "deadline",
    discount_factor: float = 0.95,
    step_penalty: float = -0.05,
) -> "up.engines.MDP":
    """Build and compile an MDP from scratch (fresh caches on each call)."""
    random.seed(seed)
    np.random.seed(seed)

    if domain not in _DOMAIN_CLASS_MAP:
        raise ValueError(
            f"Domain '{domain}' not supported for the Python API path. "
            f"Supported: {list(_DOMAIN_CLASS_MAP)}"
        )

    domain_cls = _DOMAIN_CLASS_MAP[domain]
    model = domain_cls(kind="regular", deadline=deadline, object_amount=object_amount)

    if domain == "nasa_rover":
        grounder = up.engines.Grounder(model.grounding_map())
    else:
        grounder = up.engines.Grounder()

    grounding_result = grounder._compile(model.problem)
    ground_problem = grounding_result.problem
    convert_problem = Convert_problem(ground_problem)
    converted_problem = convert_problem._converted_problem
    return MDP(
        converted_problem,
        discount_factor=discount_factor,
        reward_mode=reward_mode,
        step_penalty=step_penalty,
    )


# ---------------------------------------------------------------------------
# Single heuristic benchmark
# ---------------------------------------------------------------------------

def run_timing(
    *,
    heuristic_key: str,
    domain: str,
    object_amount: int,
    deadline: int,
    heuristic_depth: int,
    max_steps: int,
    seed: int,
    reward_mode: str,
    discount_factor: float,
    step_penalty: float,
) -> dict[str, Any]:
    alias = HEURISTIC_ALIASES[heuristic_key]
    h_name = alias["heuristic_name"]
    strategy = alias["temporal_heuristic_strategy"]

    print(f"\n{'='*60}", flush=True)
    print(f"  Heuristic : {heuristic_key}  ({alias['label']})", flush=True)
    print(f"  Internal  : heuristic_name={h_name}  strategy={strategy}", flush=True)
    print(f"  Scenario  : {domain} obj={object_amount}  deadline={deadline}  depth={heuristic_depth}", flush=True)
    print(f"{'='*60}", flush=True)

    mdp = build_mdp(
        domain,
        object_amount,
        deadline,
        seed,
        reward_mode=reward_mode,
        discount_factor=discount_factor,
        step_penalty=step_penalty,
    )
    reset_metrics()

    result = gp_solver.plan(
        mdp,
        steps=max_steps,
        search_time=1,
        search_depth=40,
        exploration_constant=10.0,
        selection_type="avg",
        k=10,
        heuristic_name=h_name,
        temporal_heuristic_depth=heuristic_depth,
        temporal_heuristic_strategy=strategy,
    )

    plan_success = bool(result[0])
    plan_makespan = result[1]

    if plan_makespan == float("-inf"):
        print(f"\n  Plan result: no plan found", flush=True)
    else:
        print(f"\n  Plan result: success={plan_success}  makespan={plan_makespan:.2f}", flush=True)

    m = get_metrics()
    s = m.summary() if m is not None else {}
    print(f"\n{m.report() if m is not None else '(no timing data)'}", flush=True)

    row: dict[str, Any] = {
        "heuristic": heuristic_key,
        "heuristic_label": alias["label"],
        "heuristic_name_internal": h_name,
        "strategy_internal": strategy if h_name == "temporal_probabilistic_rpg" else "n/a",
        "domain": domain,
        "object_amount": object_amount,
        "deadline": deadline,
        "reward_mode": reward_mode,
        "discount_factor": discount_factor,
        "step_penalty": step_penalty,
        "seed": seed,
        "heuristic_depth": heuristic_depth,
        "max_steps": max_steps,
        "plan_success": plan_success,
        "plan_makespan": plan_makespan,
    }
    # Flatten summary dict into row (keys already match CSV_FIELDNAMES)
    for field in CSV_FIELDNAMES:
        if field in s and field not in row:
            row[field] = s[field]
    # Fill any missing timing fields with empty string
    for field in CSV_FIELDNAMES:
        row.setdefault(field, "")

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else _original_argv[1:])
    validate_heuristics(args.heuristics)

    print(f"\n{'='*60}", flush=True)
    print(f"  Heuristic Per-Call Runtime Benchmark", flush=True)
    print(f"  Scenario  : {args.domain}  obj={args.object_amount}  deadline={args.deadline}", flush=True)
    heuristic_depth = (
        args.heuristic_depth
        if args.heuristic_depth is not None
        else (args.deadline if args.deadline is not None else DEFAULT_HEURISTIC_DEPTH_FALLBACK)
    )
    print(
        f"  H-depth   : {heuristic_depth}"
        + (
            ""
            if args.heuristic_depth is not None
            else " (default: same as deadline)"
        )
        + f"  max_steps={args.max_steps}  seed={args.seed}  "
        f"reward={args.reward_mode}  gamma={args.discount_factor}  step_penalty={args.step_penalty}",
        flush=True,
    )
    print(f"  Heuristics: {args.heuristics}", flush=True)
    print(f"  Output    : {args.output}", flush=True)
    print(f"{'='*60}", flush=True)

    rows: list[dict[str, Any]] = []

    for h_key in args.heuristics:
        row = run_timing(
            heuristic_key=h_key,
            domain=args.domain,
            object_amount=args.object_amount,
            deadline=args.deadline,
            heuristic_depth=heuristic_depth,
            max_steps=args.max_steps,
            seed=args.seed,
            reward_mode=args.reward_mode,
            discount_factor=args.discount_factor,
            step_penalty=args.step_penalty,
        )
        rows.append(row)
        write_csv(rows, Path(args.output), fieldnames=CSV_FIELDNAMES)

    # Final ranking sorted by wrapper_avg_call_sec ascending
    print(f"\n\n{'='*60}", flush=True)
    print(f"  RANKING (by wrapper_avg_call_sec, fastest first)", flush=True)
    print(f"{'='*60}", flush=True)

    def _sort_key(r: dict) -> float:
        v = r.get("wrapper_avg_call_sec", "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")

    sorted_rows = sorted(rows, key=_sort_key)
    print_summary_table(sorted_rows, RANKING_COLUMNS)

    write_csv(rows, Path(args.output), fieldnames=CSV_FIELDNAMES)
    print(f"\nDone. Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    # Restore original argv so any downstream module that rechecks it is unaffected.
    main()
    sys.argv = _original_argv
