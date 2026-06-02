"""
Shared utilities for experiment scripts.

Provides:
- Heuristic alias mapping (user-facing names -> internal strategy keys)
- Subprocess runner that captures stdout and parses solver metrics
- CSV writer helper

Resolution heuristic (`atomic_exact_resolution` / `atom_backtrack_exact_resolution`):
pass `resolution_alpha`, `resolution_forced_minimum`, `resolution_reference_t`
into `run_domain_subprocess`, or append the matching `--resolution-*` flags via
`extra_args`.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Heuristic naming
# ---------------------------------------------------------------------------

# Map user-facing short names to internal (strategy, heuristic_name) pairs.
# heuristic_name is either "temporal_probabilistic_rpg" or "trpg".
HEURISTIC_ALIASES: dict[str, dict[str, str]] = {
    "baseline": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline",
        "label": "baseline",
    },
    "baseline_cached": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_cached",
        "label": "baseline_cached",
    },
    # Delete/survival-aware baseline: forward DP with a per-step survival factor
    # S_t(f) so deletable facts (e.g. free(m)) decay below 1 instead of being
    # pinned at 1. NOT monotone, NOT admissible.
    "baseline_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_survival",
        "label": "baseline_survival",
    },
    # Same survival propagation as baseline_survival, but scored with the
    # variance-aware "meanvar" goal aggregation (mean - alpha*sqrt(k-1)*std over
    # per-goal areas). Kept separate so the two can be compared head-to-head.
    "baseline_survival_meanvar": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_survival_meanvar",
        "label": "baseline_survival_meanvar",
    },
    # Survival propagation with the component-wise AND-layer gamma correction
    # replacing the flat precondition product R(a). Collapses to baseline_survival
    # when no static precondition dependency is detected. NOT a calibrated
    # probability; the goal is better AND-layer ranking direction.
    "baseline_survival_and_gamma": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_survival_and_gamma",
        "label": "baseline_survival_and_gamma",
    },
    # Resolution backtrack (log-spaced / exponential-width layers) with the same
    # component-wise AND-layer gamma correction as baseline_survival_and_gamma.
    # Collapses to atomic_exact_resolution when no precondition dependency exists.
    "atomic_exact_resolution_and_gamma": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution_and_gamma",
        "label": "atomic_exact_resolution_and_gamma",
    },
    # Synonym: internal temporal_heuristic_strategy name.
    "atom_backtrack_exact_resolution_and_gamma": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution_and_gamma",
        "label": "atom_backtrack_exact_resolution_and_gamma",
    },
    # Survival/delete forward DP over log-spaced (exponential-width) resolution
    # layers: P_{t-k} with k = exponential gap. Standalone, and the suffix
    # evaluator of rollout_aligned_resolution_survival.
    "baseline_survival_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_survival_resolution",
        "label": "baseline_survival_resolution",
    },
    # Rollout-aligned common-horizon PTRPG (3 versions). Each aligns a node's
    # remaining horizon to a shared suffix horizon H via real prefix rollouts,
    # then scores the common suffix with the named underlying PTRPG.
    #   v1: baseline (pure testing) | v2: baseline_survival | v3: survival+resolution
    "rollout_aligned_baseline": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "rollout_aligned_baseline",
        "label": "rollout_aligned_baseline",
    },
    "rollout_aligned_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "rollout_aligned_survival",
        "label": "rollout_aligned_survival",
    },
    "rollout_aligned_resolution_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "rollout_aligned_resolution_survival",
        "label": "rollout_aligned_resolution_survival",
    },
    # Option A: frontier-aligned SELECTION. Same per-node aligned value as the
    # rollout-aligned strategies, but used as a frontier selection score (blended
    # with Q via lambda_align) to choose which child to expand; the original node
    # is expanded (no rollout endpoints inserted). Compare against rollout_aligned_*.
    "frontier_aligned_baseline": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_baseline",
        "label": "frontier_aligned_baseline",
    },
    "frontier_aligned_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_survival",
        "label": "frontier_aligned_survival",
    },
    "frontier_aligned_resolution_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_resolution_survival",
        "label": "frontier_aligned_resolution_survival",
    },
    # Fresh global Option A (selection-only aligned value; no lambda blend).
    # First-test CLI: --rollout-aligned-redo 1 --rollout-aligned-boundary-mode wait_no_overshoot
    "frontier_aligned_option_a": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_option_a",
        "label": "frontier_aligned_option_a",
    },
    "frontier_aligned_option_a_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_option_a_survival",
        "label": "frontier_aligned_option_a_survival",
    },
    "frontier_aligned_option_a_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_option_a_resolution",
        "label": "frontier_aligned_option_a_resolution",
    },
    "atomic_exact": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact",
        "label": "atomic_exact",
    },
    "atomic_exact_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution",
        "label": "atomic_exact_resolution",
    },
    # Synonym: internal temporal_heuristic_strategy name (same mapping as atomic_exact_resolution).
    "atom_backtrack_exact_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution",
        "label": "atom_backtrack_exact_resolution",
    },
    # Bias-corrected variant: same base scoring as atomic_exact_resolution + structural
    # per-layer correction B(t). Pre-planning is amortized; per-call cost is one lookup.
    "atomic_exact_unbiased": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_unbiased",
        "label": "atomic_exact_unbiased",
    },
    "atom_backtrack_exact_unbiased": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_unbiased",
        "label": "atom_backtrack_exact_unbiased",
    },
    "atomic_exact_cached": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_cached",
        "label": "atomic_exact_cached",
    },
    "fast_atom_cache": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "fast_atom_cache",
        "label": "fast_atom_cache",
    },
    # Correlation-aware DP leaf; same as --heuristic_name baseline_pessimistic on run_domain.
    "baseline_pessimistic": {
        "heuristic_name": "baseline_pessimistic",
        "temporal_heuristic_strategy": "baseline",
        "label": "baseline_pessimistic",
    },
    # Historical typo alias (parser + MCTS accept it).
    "baseline_passmistic": {
        "heuristic_name": "baseline_pessimistic",
        "temporal_heuristic_strategy": "baseline",
        "label": "baseline_pessimistic",
    },
    "ptrpg_old": {
        "heuristic_name": "trpg",
        "temporal_heuristic_strategy": "baseline",  # unused for trpg
        "label": "ptrpg_old (trpg)",
    },
    # MCTS leaf: real stochastic rollout to terminal 0/1; PTRPG only guides action choice.
    "ptrpg_guided_rollout_baseline_survival_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_survival_resolution",
        "value_mode": "ptrpg_guided_terminal_rollout",
        "ptrpg_guided_rollout_policy": "baseline_survival_resolution",
        "label": "ptrpg_guided_rollout_baseline_survival_resolution",
    },
    "ptrpg_guided_rollout_atomic_exact_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution",
        "value_mode": "ptrpg_guided_terminal_rollout",
        "ptrpg_guided_rollout_policy": "atomic_exact_resolution",
        "label": "ptrpg_guided_rollout_atomic_exact_resolution",
    },
    # MCTS leaf: PTRPG-guided prefix to fixed tail horizon, then PTRPG(state, H).
    "fixed_tail_atomic_exact_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution",
        "value_mode": "fixed_tail_ptrpg_rollout",
        "ptrpg_guided_rollout_policy": "atomic_exact_resolution",
        "label": "fixed_tail_atomic_exact_resolution",
    },
}

ALL_HEURISTICS = list(HEURISTIC_ALIASES.keys())

DEFAULT_HEURISTICS_MCTS = [
    "ptrpg_old",
    "baseline",
    "baseline_cached",
    "atomic_exact",
    "atomic_exact_resolution",
    "atomic_exact_cached",
    "fast_atom_cache",
]

DEFAULT_HEURISTICS_RUNTIME = [
    "ptrpg_old",
    "baseline",
    "baseline_cached",
    "atomic_exact",
    "atomic_exact_resolution",
    "atomic_exact_cached",
    "fast_atom_cache",
]

# ---------------------------------------------------------------------------
# Repository root detection
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
RUN_DOMAIN_PY = REPO_ROOT / "unified_planning" / "run_domain.py"


def validate_heuristics(names: list[str]) -> None:
    for n in names:
        if n not in HEURISTIC_ALIASES:
            valid = ", ".join(ALL_HEURISTICS)
            raise ValueError(f"Unknown heuristic '{n}'. Valid options: {valid}")


# ---------------------------------------------------------------------------
# Subprocess runner for run_domain.py
# ---------------------------------------------------------------------------

def run_domain_subprocess(
    *,
    domain: str,
    object_amount: int,
    deadline: int,
    runs: int,
    seed: int,
    solver: str,
    heuristic_name: str,
    temporal_heuristic_strategy: str,
    temporal_heuristic_depth: int,
    search_time: int = 1,
    search_depth: int = 40,
    k: int = 10,
    selection_type: str = "avg",
    exploration_constant: float = 10.0,
    reward_mode: str = "deadline",
    discount_factor: float = 0.95,
    step_penalty: float = -0.05,
    value_mode: str = "tp_mcts",
    final_selection: str = "q",
    ptrpg_guided_rollout_policy: str | None = None,
    ptrpg_guided_rollout_max_steps: int | None = None,
    ptrpg_guided_rollout_epsilon: float | None = None,
    ptrpg_guided_rollout_debug: bool = False,
    fixed_tail_prefix_frac: float | None = None,
    fixed_tail_debug: bool = False,
    garbage_amount: int = 0,
    resolution_alpha: float | None = None,
    resolution_forced_minimum: bool = False,
    resolution_reference_t: int | None = None,
    and_gamma_rollout_calibration: bool = False,
    rollout_aligned_h: int | None = None,
    rollout_aligned_redo: int | None = None,
    rollout_aligned_policy: str | None = None,
    rollout_aligned_cache: bool = False,
    rollout_aligned_max_rollouts_per_node: int | None = None,
    rollout_aligned_max_rollouts_per_search: int | None = None,
    rollout_aligned_max_time_per_search: float | None = None,
    rollout_aligned_fallback: str | None = None,
    rollout_aligned_fixed_h: bool = False,
    rollout_aligned_boundary_mode: str | None = None,
    rollout_aligned_min_dynamic_horizon: int | None = None,
    rollout_aligned_fallback_if_small: str | None = None,
    rollout_aligned_lambda_align: float | None = None,
    frontier_option_a_debug: bool = False,
    extra_args: list[str] | None = None,
    verbose: bool = False,
) -> tuple[str, int]:
    """
    Launch run_domain.py as a subprocess and return (stdout_text, returncode).

    Uses the current Python interpreter so Colab/venv paths are respected.
    """
    cmd = [
        sys.executable,
        str(RUN_DOMAIN_PY),
        "--domain", domain,
        "--object_amount", str(object_amount),
        "--garbage_amount", str(garbage_amount),
        "--deadline", str(deadline),
        "--runs", str(runs),
        "--search_time", str(search_time),
        "--search_depth", str(search_depth),
        "--k", str(k),
        "--selection_type", selection_type,
        "--exploration_constant", str(exploration_constant),
        "--reward_mode", reward_mode,
        "--discount_factor", str(discount_factor),
        "--step_penalty", str(step_penalty),
        "--value_mode", value_mode,
        "--final_selection", final_selection,
        "--seed", str(seed),
        "--solver", solver,
        "--heuristic_name", heuristic_name,
        "--temporal_heuristic_depth", str(temporal_heuristic_depth),
        "--temporal_heuristic_strategy", temporal_heuristic_strategy,
    ]
    if resolution_alpha is not None:
        cmd.extend(["--resolution-alpha", str(resolution_alpha)])
    if resolution_forced_minimum:
        cmd.append("--resolution-forced-minimum")
    if resolution_reference_t is not None:
        cmd.extend(["--resolution-reference-t", str(resolution_reference_t)])
    if and_gamma_rollout_calibration:
        cmd.append("--and-gamma-rollout-calibration")
    if rollout_aligned_h is not None:
        cmd.extend(["--rollout-aligned-h", str(rollout_aligned_h)])
    if rollout_aligned_redo is not None:
        cmd.extend(["--rollout-aligned-redo", str(rollout_aligned_redo)])
    if rollout_aligned_policy is not None:
        cmd.extend(["--rollout-aligned-policy", str(rollout_aligned_policy)])
    if rollout_aligned_cache:
        cmd.append("--rollout-aligned-cache")
    if rollout_aligned_max_rollouts_per_node is not None:
        cmd.extend(["--rollout-aligned-max-rollouts-per-node", str(rollout_aligned_max_rollouts_per_node)])
    if rollout_aligned_max_rollouts_per_search is not None:
        cmd.extend(["--rollout-aligned-max-rollouts-per-search", str(rollout_aligned_max_rollouts_per_search)])
    if rollout_aligned_max_time_per_search is not None:
        cmd.extend(["--rollout-aligned-max-time-per-search", str(rollout_aligned_max_time_per_search)])
    if rollout_aligned_fallback is not None:
        cmd.extend(["--rollout-aligned-fallback", str(rollout_aligned_fallback)])
    if rollout_aligned_fixed_h:
        cmd.append("--rollout-aligned-fixed-h")
    if rollout_aligned_boundary_mode is not None:
        cmd.extend(["--rollout-aligned-boundary-mode", str(rollout_aligned_boundary_mode)])
    if rollout_aligned_min_dynamic_horizon is not None:
        cmd.extend(["--rollout-aligned-min-dynamic-horizon", str(rollout_aligned_min_dynamic_horizon)])
    if rollout_aligned_fallback_if_small is not None:
        cmd.extend(["--rollout-aligned-fallback-if-small", str(rollout_aligned_fallback_if_small)])
    if rollout_aligned_lambda_align is not None:
        cmd.extend(["--rollout-aligned-lambda-align", str(rollout_aligned_lambda_align)])
    if frontier_option_a_debug:
        cmd.append("--frontier-option-a-debug")
    if ptrpg_guided_rollout_policy is not None:
        cmd.extend(["--ptrpg-guided-rollout-policy", str(ptrpg_guided_rollout_policy)])
    if ptrpg_guided_rollout_max_steps is not None:
        cmd.extend(["--ptrpg-guided-rollout-max-steps", str(ptrpg_guided_rollout_max_steps)])
    if ptrpg_guided_rollout_epsilon is not None:
        cmd.extend(["--ptrpg-guided-rollout-epsilon", str(ptrpg_guided_rollout_epsilon)])
    if ptrpg_guided_rollout_debug:
        cmd.append("--ptrpg-guided-rollout-debug")
    if fixed_tail_prefix_frac is not None:
        cmd.extend(["--fixed-tail-prefix-frac", str(fixed_tail_prefix_frac)])
    if fixed_tail_debug:
        cmd.append("--fixed-tail-debug")
    if extra_args:
        cmd.extend(extra_args)

    if verbose:
        print(f"  CMD: {' '.join(cmd)}", flush=True)

    # Prefer the repo's `unified_planning` over any same-named site-packages install.
    env = os.environ.copy()
    repo = str(REPO_ROOT)
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo + os.pathsep + prev_pp if prev_pp else repo

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    output = proc.stdout + proc.stderr
    return output, proc.returncode


# ---------------------------------------------------------------------------
# Metric parsing from run_domain.py stdout
# ---------------------------------------------------------------------------

_METRIC_PATTERNS: dict[str, re.Pattern] = {
    "runs_total": re.compile(r"Completed\s*=\s*(\S+)"),
    "amount_success": re.compile(r"Amount of success\s*=\s*(\S+)"),
    "avg_success_time": re.compile(r"Average success time\s*=\s*(\S+)"),
    "std_success_time": re.compile(r"STD success time\s*=\s*(\S+)"),
}


def parse_run_metrics(output: str) -> dict[str, Any]:
    """Extract key metrics from run_domain.py stdout."""
    result: dict[str, Any] = {}
    for key, pattern in _METRIC_PATTERNS.items():
        m = pattern.search(output)
        if m:
            raw = m.group(1)
            try:
                result[key] = int(raw)
            except ValueError:
                try:
                    result[key] = float(raw)
                except ValueError:
                    result[key] = raw
        else:
            result[key] = None

    # Derived: success rate as float 0..1
    amt = result.get("amount_success")
    tot = result.get("runs_total")
    if amt is not None and tot is not None and tot > 0:
        try:
            result["success_rate"] = round(int(amt) / int(tot), 4)
        except (TypeError, ValueError):
            result["success_rate"] = None
    else:
        result["success_rate"] = None

    return result


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str] | None = None) -> None:
    """Write a list of dicts to a CSV file, auto-detecting columns if not given."""
    if not rows:
        print(f"[warn] No rows to write to {path}")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Pretty terminal table
# ---------------------------------------------------------------------------

def print_summary_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Print a simple fixed-width table to stdout."""
    col_widths = [max(len(str(c)), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in columns]
    sep = "  ".join("-" * w for w in col_widths)
    header = "  ".join(str(c).ljust(w) for c, w in zip(columns, col_widths))
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(columns, col_widths))
        print(line)
