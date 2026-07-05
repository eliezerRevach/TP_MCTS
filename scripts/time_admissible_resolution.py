"""One-off: time a SINGLE heuristic call (no MCTS) for baseline_admissible
(dense) vs baseline_admissible_resolution (2^(k/2) backward) on NASA Rover 2.

Builds the real domain, takes the initial state, and times one fresh
(cache-miss) heuristic_score call at each deadline. Reports wall time + value
so you can see the 2^(k/2) speedup and confirm resolution >= dense (admissible).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))
_orig_argv = sys.argv[:]
sys.argv = [_orig_argv[0]]

import unified_planning as up  # noqa: E402
import unified_planning.domains  # noqa: E402
from unified_planning.engines.convert_problem import Convert_problem  # noqa: E402
from unified_planning.engines.mdp import MDP  # noqa: E402
from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (  # noqa: E402
    TemporalProbabilisticRPGHeuristic,
)


def build_mdp(object_amount: int, deadline: int) -> MDP:
    model = up.domains.Nasa_Rover(
        kind="regular", deadline=deadline, object_amount=object_amount
    )
    grounder = up.engines.Grounder(model.grounding_map())
    ground_problem = grounder._compile(model.problem).problem
    converted = Convert_problem(ground_problem)._converted_problem
    return MDP(converted, discount_factor=0.95, reward_mode="deadline", step_penalty=-0.05)


def time_once(strategy: str, deadline: int, object_amount: int = 2, alpha: float = 2.0):
    mdp = build_mdp(object_amount, deadline)
    state = mdp.initial_state()
    goals = mdp.problem.goals
    heuristic = TemporalProbabilisticRPGHeuristic.from_problem(mdp.problem)
    kwargs = {}
    if strategy in (
        "baseline_admissible_resolution",
        "baseline_admissible_resolution_forward",
        "atom_backtrack_exact_resolution",
    ):
        kwargs = {"resolution_alpha": alpha}
    # First (cache-miss) call is the real cost.
    t0 = time.perf_counter()
    value = heuristic.heuristic_score(
        state,
        goals,
        aggregation="product",
        fixed_depth=deadline,
        start_time=0.0,
        strategy=strategy,
        **kwargs,
    )
    elapsed = time.perf_counter() - t0
    return float(value), elapsed


def main():
    object_amount = 2
    print(f"NASA Rover {object_amount} — single heuristic call (no MCTS)\n")
    header = f"{'deadline':>8} {'variant':>42} {'value':>12} {'call_sec':>12}"
    print(header)
    print("-" * len(header))
    # variants: (label, strategy, alpha)
    #   alpha=1 -> backward pass but EVERY layer is an anchor (no skip): isolates
    #             the backward effect from the log-skip effect.
    #   alpha=2 -> backward pass + 2^(k/2) log skip.
    variants = [
        ("baseline_admissible (dense)", "baseline_admissible", None),
        ("backward resolution alpha=2^(k/2)", "baseline_admissible_resolution", 2.0),
        ("forward resolution alpha=2^(k/2)", "baseline_admissible_resolution_forward", 2.0),
    ]
    for deadline in (3, 4, 5, 6, 8, 10, 12, 15, 20, 25):
        for label, strategy, alpha in variants:
            if alpha is None:
                value, elapsed = time_once(strategy, deadline, object_amount)
            else:
                value, elapsed = time_once(strategy, deadline, object_amount, alpha=alpha)
            print(f"{deadline:>8} {label:>42} {value:>12.6f} {elapsed:>12.4f}")
        print()


if __name__ == "__main__":
    main()
    sys.argv = _orig_argv
