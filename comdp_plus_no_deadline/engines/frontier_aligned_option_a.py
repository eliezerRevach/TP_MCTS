"""
Option A: global open-frontier MCTS selection (frontier_aligned_option_a family).

At each search iteration, collect all expandable leaf nodes F across the tree,
align every candidate to the same suffix horizon H_frontier = deadline - max(elapsed),
score via prefix rollout + PTRPG, pick argmax aligned_value, expand ONLY that node
with the standard heuristic/backprop pipeline. Aligned values are selection-only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

from comdp_plus_no_deadline.engines.rollout_aligned import (
    BOUNDARY_WAIT_NO_OVERSHOOT,
    RolloutAlignedConfig,
    RolloutAlignedEvaluator,
)

# MCTS temporal_heuristic_strategy -> underlying PTRPG suffix strategy.
OPTION_A_STRATEGIES: Dict[str, str] = {
    "frontier_aligned_option_a": "baseline",
    "frontier_aligned_option_a_survival": "baseline_survival",
    "frontier_aligned_option_a_resolution": "baseline_survival_resolution",
}


def is_option_a_strategy(strategy: Optional[str]) -> bool:
    return (strategy or "") in OPTION_A_STRATEGIES


def option_a_ptrpg_suffix(strategy: str) -> str:
    return OPTION_A_STRATEGIES[strategy]


@dataclass
class OptionAConfig:
    """Fixed defaults for Option A; ROLLOUT_ALIGNED_H / fixed-H must not apply."""

    redo: int = 1
    boundary_mode: str = BOUNDARY_WAIT_NO_OVERSHOOT
    prefix_rollout_policy: str = "random"
    use_dynamic_H: bool = True
    # Deliberately high so tests prove it is never read in dynamic mode.
    common_horizon_H: int = 999

    def to_rollout_config(self) -> RolloutAlignedConfig:
        return RolloutAlignedConfig(
            use_dynamic_H=True,
            common_horizon_H=int(self.common_horizon_H),
            redo=max(1, int(self.redo)),
            prefix_rollout_policy=str(self.prefix_rollout_policy),
            boundary_mode=str(self.boundary_mode),
            cache_aligned_values=False,
            lambda_align=1.0,
        )


def option_a_config_from_cli() -> OptionAConfig:
    """Build Option A config; only redo/boundary are optionally overridden from CLI."""
    cfg = OptionAConfig()
    try:
        import unified_planning as up

        a = getattr(up, "args", None)
    except Exception:
        a = None
    if a is None:
        return cfg
    redo = getattr(a, "rollout_aligned_redo", None)
    if redo is not None:
        cfg.redo = max(1, int(redo))
    boundary = getattr(a, "rollout_aligned_boundary_mode", None)
    if boundary:
        cfg.boundary_mode = str(boundary)
    return cfg


def node_elapsed(snode, root_stn=None) -> float:
    """Elapsed time at an S-node (STN end time of the incoming action edge)."""
    parent = getattr(snode, "parent", None)
    if parent is not None:
        stn = getattr(parent, "stn", None)
        if stn is not None:
            try:
                return float(stn.get_current_end_time())
            except Exception:
                pass
    if root_stn is not None:
        try:
            return float(root_stn.get_current_end_time())
        except Exception:
            pass
    return float(getattr(getattr(snode, "state", None), "current_time", 0.0))


def collect_global_frontier(
    root,
    *,
    search_depth: int,
    deadline: Optional[float] = None,
    root_stn=None,
) -> List[Any]:
    """All open/expandable S-nodes in the tree (global frontier F)."""
    frontier: List[Any] = []
    stack = [root]
    seen = set()
    while stack:
        snode = stack.pop()
        sid = id(snode)
        if sid in seen:
            continue
        seen.add(sid)
        if not getattr(snode, "possible_actions", None):
            continue
        depth = int(getattr(snode, "depth", 0))
        if depth > search_depth:
            continue
        if deadline is not None:
            elapsed = node_elapsed(snode, root_stn=root_stn)
            if elapsed > float(deadline):
                continue
        children = getattr(snode, "children", None) or {}
        expandable = any(
            (getattr(anode, "count", 0) == 0 or not getattr(anode, "children", None))
            for anode in children.values()
        )
        if expandable:
            frontier.append(snode)
        for anode in children.values():
            for child in (getattr(anode, "children", None) or {}).values():
                stack.append(child)
    return frontier


def compute_H_frontier(
    frontier: Sequence[Any],
    deadline: Optional[float],
    *,
    root_stn=None,
) -> Tuple[float, Optional[float], Dict[int, float]]:
    """deepest_elapsed, H_frontier, elapsed_map keyed by id(snode)."""
    if not frontier:
        return 0.0, None, {}
    elapsed_map = {id(n): node_elapsed(n, root_stn=root_stn) for n in frontier}
    deepest_elapsed = max(elapsed_map.values())
    if deadline is None:
        return deepest_elapsed, None, elapsed_map
    H_frontier = float(deadline) - deepest_elapsed
    assert abs(H_frontier - (float(deadline) - deepest_elapsed)) < 1e-9, "H_frontier mismatch"
    return deepest_elapsed, H_frontier, elapsed_map


def remaining_horizon(deadline: Optional[float], elapsed: float, configured_depth: int) -> int:
    if deadline is not None:
        return max(0, int(math.floor(float(deadline) - elapsed)))
    return max(0, int(configured_depth))


def aligned_value_for_node(
    evaluator: RolloutAlignedEvaluator,
    state,
    *,
    remaining: int,
    H_frontier: Optional[float],
    deepest_elapsed: float,
    elapsed: float,
) -> float:
    """Option A aligned value: prefix to delta then PTRPG at H_frontier."""
    if H_frontier is None:
        return evaluator.evaluate(state, remaining, h_override=None)
    H = max(0, int(math.floor(float(H_frontier))))
    delta = float(deepest_elapsed) - float(elapsed)
    assert abs(delta - (remaining - H)) < 1e-6 or remaining >= H, (
        f"delta mismatch: delta={delta} remaining={remaining} H={H}"
    )
    return evaluator.evaluate(state, remaining, h_override=H)


def select_frontier_node(
    frontier: Sequence[Any],
    scores: Dict[int, float],
) -> Any:
    """argmax aligned_value; ties -> lowest node_id."""
    best = None
    best_score = -float("inf")
    best_nid = float("inf")
    for n in frontier:
        nid = getattr(n, "_option_a_node_id", id(n))
        sc = scores.get(id(n), -float("inf"))
        if sc > best_score or (abs(sc - best_score) < 1e-12 and nid < best_nid):
            best_score = sc
            best_nid = nid
            best = n
    return best


def format_option_a_debug_row(
    *,
    node_id: int,
    parent_id: Optional[int],
    depth: int,
    elapsed: float,
    remaining: float,
    deepest_elapsed: float,
    H_frontier: Optional[float],
    delta: float,
    raw_ptrpg: float,
    aligned_value: float,
    selected: bool,
) -> str:
    hf = "None" if H_frontier is None else f"{H_frontier:.4f}"
    pid = "None" if parent_id is None else str(parent_id)
    sel = "yes" if selected else "no"
    return (
        f"node_id={node_id} parent_id={pid} depth={depth} "
        f"elapsed={elapsed:.4f} remaining={remaining:.4f} "
        f"deepest_elapsed={deepest_elapsed:.4f} H_frontier={hf} delta={delta:.4f} "
        f"raw_PTRPG={raw_ptrpg:.4f} aligned_value={aligned_value:.4f} selected={sel}"
    )


def build_option_a_evaluator(
    heuristic_mdp,
    heuristic,
    suffix_strategy: str,
    *,
    aggregation: str = "product",
    resolution_kwargs: Optional[dict] = None,
    config: Optional[OptionAConfig] = None,
) -> RolloutAlignedEvaluator:
    """Cached Option A evaluator; never uses fixed ROLLOUT_ALIGNED_H."""
    cache = getattr(heuristic_mdp, "_option_a_evaluators", None)
    if cache is None:
        cache = {}
        setattr(heuristic_mdp, "_option_a_evaluators", cache)
    cache_key = (suffix_strategy, "option_a_v1")
    existing = cache.get(cache_key)
    if existing is not None:
        return existing

    cfg = (config or option_a_config_from_cli()).to_rollout_config()
    res_kwargs = resolution_kwargs or {}
    deadline = heuristic_mdp.deadline()
    goals = set(heuristic_mdp.problem.goals)

    def _is_goal(s):
        preds = getattr(s, "predicates", None)
        return preds is not None and goals.issubset(set(preds))

    def raw_eval_fn(s, horizon):
        ct = float(getattr(s, "current_time", 0.0))
        if deadline is not None:
            eff = min(int(horizon), max(0, int(math.floor(deadline - ct))))
        else:
            eff = int(horizon)
        return heuristic.heuristic_score(
            s,
            goals,
            aggregation=aggregation,
            fixed_depth=eff,
            start_time=ct,
            strategy=suffix_strategy,
            **res_kwargs,
        )

    def prefix_rollout_fn(s, delta):
        sim = s
        start_ct = float(getattr(sim, "current_time", 0.0))
        reached = False
        dead_end = False
        length = 0
        no_overshoot = cfg.boundary_mode == BOUNDARY_WAIT_NO_OVERSHOOT
        while (float(getattr(sim, "current_time", 0.0)) - start_ct) < delta:
            legal = heuristic_mdp.legal_actions(sim)
            if not legal:
                dead_end = True
                break
            if no_overshoot:
                committed = False
                candidates = list(legal)
                random.shuffle(candidates)
                for action in candidates:
                    terminal, next_state, _r = heuristic_mdp.step(sim, action)
                    new_elapsed = float(getattr(next_state, "current_time", 0.0)) - start_ct
                    if new_elapsed <= delta:
                        sim = next_state
                        committed = True
                        length += 1
                        if _is_goal(sim):
                            reached = True
                        elif terminal:
                            dead_end = True
                        break
                if not committed:
                    break
                if reached or dead_end:
                    break
            else:
                action = random.choice(legal)
                terminal, next_state, _r = heuristic_mdp.step(sim, action)
                sim = next_state
                length += 1
                if _is_goal(sim):
                    reached = True
                    break
                if terminal:
                    dead_end = True
                    break
        return sim, reached, length, dead_end

    def state_hash_fn(s):
        preds = getattr(s, "predicates", None)
        ct = getattr(s, "current_time", 0.0)
        if preds is not None:
            return (frozenset(preds), ct)
        return (id(s), ct)

    evaluator = RolloutAlignedEvaluator(
        config=cfg,
        raw_eval_fn=raw_eval_fn,
        prefix_rollout_fn=prefix_rollout_fn,
        state_hash_fn=state_hash_fn,
    )
    cache[cache_key] = evaluator
    return evaluator
