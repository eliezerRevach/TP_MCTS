"""
MCTS Heuristic Comparison Script
=================================

Runs TP-MCTS across the four NASA rover scenarios and multiple heuristic
variants, then saves a flat CSV results table.

Default scenario grid
---------------------
  (nasa_rover, objects=2, deadline=25)
  (nasa_rover, objects=2, deadline=35)
  (nasa_rover, objects=3, deadline=25)
  (nasa_rover, objects=3, deadline=35)

Default heuristics
------------------
  ptrpg_old           -- classical PTRPG from TP-MCTS paper (heuristic_name=trpg)
  baseline            -- temporal_probabilistic_rpg / baseline strategy
  baseline_cached     -- temporal_probabilistic_rpg / baseline_cached
  atomic_exact        -- temporal_probabilistic_rpg / atom_backtrack_exact
  atomic_exact_cached -- temporal_probabilistic_rpg / atom_backtrack_cached
  fast_atom_cache     -- temporal_probabilistic_rpg / fast_atom_cache

Output
------
  results/mcts_heuristic_comparison.csv   (overwritten each run)
  Terminal summary table after all experiments finish.

Usage (Colab or local)
----------------------
  # From the repo root:
  python scripts/run_mcts_heuristic_comparison.py

  # Custom settings:
  python scripts/run_mcts_heuristic_comparison.py \\
      --runs 10 \\
      --seed 42 \\
      --heuristics ptrpg_old baseline baseline_cached \\
      --objects 2 3 \\
      --deadlines 25 35 \\
      --search_time 1 \\
      --search_depth 0 \\
      --k 9999 \\
      --output results/my_run.csv \\
      --verbose

  # Override heuristic depth (default: same as deadline):
      --heuristic_depth 30

  # Use reward_mode=terminal instead of deadline:
      --reward_mode terminal
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow running from any working directory.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
# Insert both so that `unified_planning` package and sibling script are found.
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

from experiment_common import (
    ALL_HEURISTICS,
    DEFAULT_HEURISTICS_MCTS,
    HEURISTIC_ALIASES,
    REPO_ROOT,
    parse_run_metrics,
    print_summary_table,
    run_domain_subprocess,
    validate_heuristics,
    write_csv,
)

# ---------------------------------------------------------------------------
# Default scenario grid (matches professor request)
# ---------------------------------------------------------------------------

DEFAULT_OBJECTS = [2, 3]
DEFAULT_DEADLINES = [25, 35]
DEFAULT_DOMAIN = "nasa_rover"
DEFAULT_RUNS = 20
DEFAULT_SEED = 123
DEFAULT_SEARCH_TIME = 1
DEFAULT_SEARCH_DEPTH = 40  # Matches run_domain.py default.
DEFAULT_K = 10  # Matches run_domain.py default for selection_type=max child sampling.
DEFAULT_EXPLORATION_CONSTANT = 10.0
DEFAULT_SELECTION_TYPE = "avg"
DEFAULT_REWARD_MODE = "deadline"
DEFAULT_DISCOUNT_FACTOR = 0.95
DEFAULT_STEP_PENALTY = -0.05
DEFAULT_VALUE_MODE = "tp_mcts"
DEFAULT_OUTPUT = str(REPO_ROOT / "results" / "mcts_heuristic_comparison.csv")

CSV_FIELDNAMES = [
    "domain",
    "object_amount",
    "deadline",
    "heuristic",
    "heuristic_label",
    "runs_total",
    "amount_success",
    "success_rate",
    "avg_success_time",
    "std_success_time",
    "seed",
    "search_time",
    "search_depth",
    "k",
    "selection_type",
    "exploration_constant",
    "discount_factor",
    "step_penalty",
    "reward_mode",
    "value_mode",
    "heuristic_depth",
    "returncode",
]

SUMMARY_COLUMNS = [
    "domain",
    "object_amount",
    "deadline",
    "heuristic",
    "amount_success",
    "success_rate",
    "avg_success_time",
    "std_success_time",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run TP-MCTS comparison across NASA rover scenarios and heuristics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--heuristics",
        nargs="+",
        default=DEFAULT_HEURISTICS_MCTS,
        metavar="H",
        help=(
            "Heuristics to compare. "
            f"Choices: {', '.join(ALL_HEURISTICS)}. "
            f"Default: {' '.join(DEFAULT_HEURISTICS_MCTS)}"
        ),
    )
    p.add_argument(
        "--objects",
        nargs="+",
        type=int,
        default=DEFAULT_OBJECTS,
        metavar="N",
        help=f"Object amounts to test. Default: {DEFAULT_OBJECTS}",
    )
    p.add_argument(
        "--deadlines",
        nargs="+",
        type=int,
        default=DEFAULT_DEADLINES,
        metavar="D",
        help=f"Deadlines to test. Default: {DEFAULT_DEADLINES}",
    )
    p.add_argument(
        "--domain",
        default=DEFAULT_DOMAIN,
        help=f"Domain name. Default: {DEFAULT_DOMAIN}",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Runs per experiment. Default: {DEFAULT_RUNS}",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )
    p.add_argument(
        "--search_time",
        type=int,
        default=DEFAULT_SEARCH_TIME,
        help=f"MCTS search time (seconds per step). Default: {DEFAULT_SEARCH_TIME}",
    )
    p.add_argument(
        "--search_depth",
        type=int,
        default=DEFAULT_SEARCH_DEPTH,
        help=f"MCTS search depth. Default: {DEFAULT_SEARCH_DEPTH}",
    )
    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=(
            "Number of random child actions sampled by selection_type=max. "
            f"Use a large value to evaluate all actions. Default: {DEFAULT_K}"
        ),
    )
    p.add_argument(
        "--exploration_constant",
        type=float,
        default=DEFAULT_EXPLORATION_CONSTANT,
        help=f"MCTS exploration constant (UCT C parameter). Default: {DEFAULT_EXPLORATION_CONSTANT}",
    )
    p.add_argument(
        "--selection_type",
        choices=["avg", "rootInterval", "max"],
        default=DEFAULT_SELECTION_TYPE,
        help=f"MCTS selection/scheduling variant. Default: {DEFAULT_SELECTION_TYPE}",
    )
    p.add_argument(
        "--heuristic_depth",
        type=int,
        default=None,
        help=(
            "Temporal heuristic lookahead depth. "
            "Default: same as deadline (matches demo.ipynb behavior)."
        ),
    )
    p.add_argument(
        "--reward_mode",
        choices=["deadline", "terminal"],
        default=DEFAULT_REWARD_MODE,
        help=f"Reward mode passed to the solver. Default: {DEFAULT_REWARD_MODE}",
    )
    p.add_argument(
        "--discount_factor",
        "--gamma",
        type=float,
        default=DEFAULT_DISCOUNT_FACTOR,
        dest="discount_factor",
        help=f"MDP discount factor (gamma) for MCTS. Default: {DEFAULT_DISCOUNT_FACTOR}",
    )
    p.add_argument(
        "--step_penalty",
        type=float,
        default=DEFAULT_STEP_PENALTY,
        help=f"Per-step MDP reward penalty (each transition). Default: {DEFAULT_STEP_PENALTY}",
    )
    p.add_argument(
        "--value_mode",
        choices=["tp_mcts", "greedy_matched"],
        default=DEFAULT_VALUE_MODE,
        help=f"MCTS backup target mode. Default: {DEFAULT_VALUE_MODE}",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print subprocess command lines and full stdout.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    *,
    domain: str,
    object_amount: int,
    deadline: int,
    heuristic_key: str,
    runs: int,
    seed: int,
    search_time: int,
    search_depth: int,
    k: int,
    selection_type: str,
    exploration_constant: float,
    heuristic_depth: int | None,
    reward_mode: str,
    discount_factor: float,
    step_penalty: float,
    value_mode: str,
    verbose: bool,
) -> dict:
    alias = HEURISTIC_ALIASES[heuristic_key]
    depth = heuristic_depth if heuristic_depth is not None else deadline

    label = f"[{domain} obj={object_amount} dl={deadline}] heuristic={heuristic_key}"
    print(f"\n{'─'*60}", flush=True)
    print(f"  {label}", flush=True)
    print(
        f"  heuristic_depth={depth}  search_time={search_time}  search_depth={search_depth}  "
        f"k={k}  runs={runs}  seed={seed}  C={exploration_constant}  "
        f"selection={selection_type}  gamma={discount_factor}  step_penalty={step_penalty}  "
        f"reward_mode={reward_mode}  value_mode={value_mode}",
        flush=True,
    )
    print(f"{'─'*60}", flush=True)

    t0 = time.perf_counter()
    output, returncode = run_domain_subprocess(
        domain=domain,
        object_amount=object_amount,
        deadline=deadline,
        runs=runs,
        seed=seed,
        solver="mcts",
        heuristic_name=alias["heuristic_name"],
        temporal_heuristic_strategy=alias["temporal_heuristic_strategy"],
        temporal_heuristic_depth=depth,
        search_time=search_time,
        search_depth=search_depth,
        k=k,
        selection_type=selection_type,
        exploration_constant=exploration_constant,
        reward_mode=reward_mode,
        discount_factor=discount_factor,
        step_penalty=step_penalty,
        value_mode=value_mode,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0

    if verbose:
        print(output, flush=True)

    metrics = parse_run_metrics(output)

    if returncode != 0:
        print(f"  [WARN] Process exited with code {returncode}", flush=True)
        # Show last few lines for quick debugging
        lines = [l for l in output.splitlines() if l.strip()]
        for line in lines[-6:]:
            print(f"    {line}", flush=True)

    print(
        f"  => success={metrics.get('amount_success')}/{metrics.get('runs_total')}  "
        f"rate={metrics.get('success_rate')}  "
        f"avg_time={metrics.get('avg_success_time')}  "
        f"wall={elapsed:.1f}s",
        flush=True,
    )

    row = {
        "domain": domain,
        "object_amount": object_amount,
        "deadline": deadline,
        "heuristic": heuristic_key,
        "heuristic_label": alias["label"],
        "seed": seed,
        "search_time": search_time,
        "search_depth": search_depth,
        "k": k,
        "selection_type": selection_type,
        "exploration_constant": exploration_constant,
        "discount_factor": discount_factor,
        "step_penalty": step_penalty,
        "reward_mode": reward_mode,
        "value_mode": value_mode,
        "heuristic_depth": depth,
        "returncode": returncode,
        **metrics,
    }
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    validate_heuristics(args.heuristics)

    scenarios = [
        (obj, dl)
        for obj in args.objects
        for dl in args.deadlines
    ]

    n_heur = len(args.heuristics)
    total = len(scenarios) * n_heur
    print(f"\n{'='*60}", flush=True)
    print(f"  TP-MCTS Heuristic Comparison", flush=True)
    print(f"  Domain    : {args.domain}", flush=True)
    print(f"  Scenarios : {scenarios}", flush=True)
    print(f"  Heuristics: {args.heuristics}", flush=True)
    print(f"  Per block : --runs {args.runs} (MCTS episodes inside run_domain for ONE row)", flush=True)
    print(f"  Outer grid: {len(scenarios)} scenarios x {n_heur} heuristics = {total} rows in CSV", flush=True)
    print(
        f"  MCTS args : search_time={args.search_time}  search_depth={args.search_depth}  "
        f"k={args.k}  C={args.exploration_constant}  selection={args.selection_type}",
        flush=True,
    )
    print(
        f"  Seed / gamma / reward : seed={args.seed}  gamma={args.discount_factor}  "
        f"step_penalty={args.step_penalty}  reward_mode={args.reward_mode}  "
        f"value_mode={args.value_mode}",
        flush=True,
    )
    print(f"  Output    : {args.output}", flush=True)
    print(f"{'='*60}", flush=True)

    rows: list[dict] = []
    completed = 0

    for obj, dl in scenarios:
        for h in args.heuristics:
            completed += 1
            print(
                f"\n[CSV row {completed}/{total}: obj={obj} deadline={dl} heuristic={h} | "
                f"inner loop = {args.runs} MCTS runs]",
                flush=True,
            )
            row = run_experiment(
                domain=args.domain,
                object_amount=obj,
                deadline=dl,
                heuristic_key=h,
                runs=args.runs,
                seed=args.seed,
                search_time=args.search_time,
                search_depth=args.search_depth,
                k=args.k,
                selection_type=args.selection_type,
                exploration_constant=args.exploration_constant,
                heuristic_depth=args.heuristic_depth,
                reward_mode=args.reward_mode,
                discount_factor=args.discount_factor,
                step_penalty=args.step_penalty,
                value_mode=args.value_mode,
                verbose=args.verbose,
            )
            rows.append(row)

            # Write incrementally so partial results survive interruption.
            write_csv(rows, Path(args.output), fieldnames=CSV_FIELDNAMES)

    print(f"\n\n{'='*60}", flush=True)
    print(f"  RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print_summary_table(rows, SUMMARY_COLUMNS)

    write_csv(rows, Path(args.output), fieldnames=CSV_FIELDNAMES)
    print(f"\nDone. Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
