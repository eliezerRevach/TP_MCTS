"""
Shared utilities for experiment scripts.

Provides:
- Heuristic alias mapping (user-facing names -> internal strategy keys)
- Subprocess runner that captures stdout and parses solver metrics
- CSV writer helper

Resolution heuristic (`atomic_exact_resolution` / `atom_backtrack_exact_resolution`):
pass `resolution_alpha`, `resolution_forced_minimum`, `resolution_k_target`,
`resolution_reference_t` into `run_domain_subprocess`, or append the matching
`--resolution-*` flags via `extra_args`.
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
    garbage_amount: int = 0,
    resolution_alpha: float | None = None,
    resolution_forced_minimum: bool = False,
    resolution_k_target: int | None = None,
    resolution_reference_t: int | None = None,
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
    if resolution_k_target is not None:
        cmd.extend(["--resolution-k-target", str(resolution_k_target)])
    if resolution_reference_t is not None:
        cmd.extend(["--resolution-reference-t", str(resolution_reference_t)])
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
