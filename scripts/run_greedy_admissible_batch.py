"""Batch greedy_parallel runtime: baseline vs resolution backward/forward (alpha=2)."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))
_orig_argv = sys.argv[:]
sys.argv = [_orig_argv[0]]

import unified_planning as up  # noqa: E402
import unified_planning.domains  # noqa: E402
from run_heuristic_runtime_per_call import run_timing  # noqa: E402

HEURISTICS = [
    "baseline_admissible",
    "baseline_admissible_resolution",
    "baseline_admissible_resolution_forward",
]

up.args.resolution_alpha = 2.0


def main() -> None:
    print("greedy_parallel avg call time (alpha=2 for resolution variants)\n")
    header = (
        f"{'obj':>3} {'dl':>3} {'heuristic':<45} "
        f"{'wrapper_avg':>12} {'worker_avg':>12} {'success':>8}"
    )
    print(header)
    print("-" * len(header))

    for obj in (2, 3):
        for dl in (25, 50):
            for h in HEURISTICS:
                row = run_timing(
                    heuristic_key=h,
                    domain="nasa_rover",
                    object_amount=obj,
                    deadline=dl,
                    heuristic_depth=dl,
                    max_steps=90,
                    seed=42,
                    reward_mode="deadline",
                    discount_factor=0.95,
                    step_penalty=-0.05,
                )
                w = float(row.get("wrapper_avg_call_sec") or 0)
                k = float(row.get("worker_avg_call_sec") or 0)
                s = row.get("plan_success", "")
                print(
                    f"{obj:>3} {dl:>3} {h:<45} {w:>12.6f} {k:>12.6f} {str(s):>8}"
                )
            print()


if __name__ == "__main__":
    main()
    sys.argv = _orig_argv
