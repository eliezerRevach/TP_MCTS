"""
Rollout-aligned common-horizon PTRPG evaluation.

Fixes the cross-depth / cross-deadline bias of PTRPG inside MCTS: raw PTRPG
evaluates each node with its *own* remaining horizon, so a shallow node with
remaining deadline 30 gets more relaxed RPG layers than a deep node with
remaining 15 and looks more optimistic only because it had more layers to
accumulate optimism in. Comparing ``PTRPG(state, 30)`` against
``PTRPG(state, 15)`` is therefore unfair.

This wrapper compares every node through a **common suffix horizon ``H``**. When a
node's remaining horizon ``R`` exceeds ``H``, the extra prefix ``Δ = R - H`` is
consumed by *real* MDP rollouts (legal actions, durations, stochastic effects,
STN-feasible transitions), and only the resulting state is scored by raw PTRPG
over the common ``H``:

    Δ = max(0, R - H)
    Δ == 0:  h_aligned = PTRPG(state, R)
    Δ  > 0:  h_aligned = mean over `redo` prefix rollouts of
                 1.0                       if the rollout reached the goal
                 PTRPG(state_after, H)     otherwise

Rollouts are used ONLY for the unmatched prefix; the common suffix is still the
real PTRPG. The PTRPG itself (noisy-OR, survival/delete, the underlying strategy)
is untouched — this is a wrapper around it.

This module is deliberately MDP-agnostic: the caller injects
  * ``raw_eval_fn(state, horizon) -> float``       (e.g. heuristic_score(strategy, depth))
  * ``prefix_rollout_fn(state, delta) -> (state2, reached_goal, length)``
  * ``state_hash_fn(state) -> Hashable``            (optional, for caching)
so it is fully unit-testable with fakes and carries no import dependency on the
MCTS / STN machinery.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, Optional, Tuple

State = object
RawEvalFn = Callable[[State, int], float]
# Returns (resulting_state, reached_goal, rollout_length_in_steps, dead_end).
# dead_end == True means the prefix hit a dead-end / STN-infeasible state before
# the boundary without reaching the goal (value 0.0).
PrefixRolloutFn = Callable[[State, int], Tuple[State, bool, int, bool]]
StateHashFn = Callable[[State], Hashable]

FALLBACK_HORIZON_CAPPED = "horizon_capped"
FALLBACK_RAW = "raw"

# fallback_if_H_too_small modes (when the dynamic H_p < min_dynamic_horizon).
SMALL_H_USE_ANYWAY = "use_anyway"  # use H_p as-is (pure correctness)
SMALL_H_FIXED = "fixed"  # fall back to the fixed common_horizon_H
SMALL_H_RAW = "raw"  # fall back to raw PTRPG(state, R)

# Boundary modes for landing the prefix rollout on delta (durations vary).
BOUNDARY_OVERSHOOT = "overshoot"  # allow the last action to overshoot delta
BOUNDARY_WAIT_NO_OVERSHOOT = "wait_no_overshoot"  # never execute an overshooting action
BOUNDARY_EXPECTED_ROUNDING = "expected_stochastic_rounding"


def frontier_score(
    existing_score: float,
    aligned_value: float,
    lambda_align: float,
    exploration: float = 0.0,
) -> float:
    """Blended frontier-selection score (Option A / frontier_aligned_*).

        frontier_score = (1 - lambda) * existing_node_score
                       + lambda * aligned_value
                       + exploration

    ``lambda_align = 1.0`` selects purely on the frontier-aligned value (plus
    exploration); ``0.0`` recovers the existing UCT exploitation term. When the
    node has no existing score, pass ``existing_score = aligned_value`` (the spec
    allows falling back to ``aligned_value`` directly).
    """
    lam = max(0.0, min(1.0, float(lambda_align)))
    return (1.0 - lam) * float(existing_score) + lam * float(aligned_value) + float(exploration)


@dataclass
class RolloutAlignedConfig:
    """Tunables for rollout-aligned common-horizon evaluation."""

    use_rollout_aligned: bool = False

    # Dynamic parent-local horizon (the main mode): H_p = min over the parent's
    # children of their remaining horizon, supplied per-node by the caller via
    # evaluate(..., h_override=H_p). When use_dynamic_H is False (or no override
    # is given) the evaluator falls back to the fixed common_horizon_H. The fixed
    # value also serves as an upper cap on H_p (a child never aligns above it).
    use_dynamic_H: bool = True
    common_horizon_H: int = 15  # fixed suffix horizon / max cap on H_p
    cap_dynamic_h_at_fixed: bool = True  # H_p <= common_horizon_H when capping

    # Safety floor on the dynamic horizon (None disables the check).
    min_dynamic_horizon: Optional[int] = None
    fallback_if_H_too_small: str = SMALL_H_USE_ANYWAY

    redo: int = 1  # number of prefix rollouts averaged per node
    prefix_rollout_policy: str = "random"
    boundary_mode: str = BOUNDARY_OVERSHOOT

    # Optional blended-selection weight (0..1). Stored for the MCTS layer; the
    # evaluator returns V_aligned, the caller decides how to blend with Q.
    lambda_align: float = 1.0

    cache_aligned_values: bool = False  # sample estimates -> off by default

    # Safety budget (0 == unlimited). The per-search caps are honored against the
    # evaluator's per-search counters (reset via reset_search_budget()).
    max_prefix_rollouts_per_node: int = 0
    max_prefix_rollouts_per_search: int = 0
    max_prefix_rollout_time_per_search: float = 0.0

    # When the budget is exhausted:
    #   horizon_capped -> PTRPG(state, min(R, H))
    #   raw            -> PTRPG(state, R)
    fallback_mode: str = FALLBACK_HORIZON_CAPPED

    # Diagnostic only: also evaluate raw PTRPG(state, R) to compare against the
    # aligned value (doubles cost, so off by default).
    track_baseline_comparison: bool = False


@dataclass
class RolloutAlignedDiagnostics:
    aligned_evaluations: int = 0
    delta_zero_evaluations: int = 0
    dynamic_h_evaluations: int = 0
    fixed_h_evaluations: int = 0
    small_h_fallbacks: int = 0
    h_p_sum: int = 0  # sum of chosen comparison horizons (for averaging)
    prefix_rollouts: int = 0
    prefix_rollout_total_length: int = 0
    prefix_rollouts_reached_goal: int = 0
    prefix_rollouts_dead_end: int = 0
    suffix_value_sum: float = 0.0
    suffix_value_count: int = 0
    prefix_rollout_time_sec: float = 0.0
    cache_hits: int = 0
    budget_fallbacks: int = 0
    aligned_value_sum: float = 0.0
    aligned_value_count: int = 0
    baseline_value_sum: float = 0.0
    baseline_value_count: int = 0

    def as_dict(self) -> Dict[str, object]:
        n_roll = max(1, self.prefix_rollouts)
        n_align = max(1, self.aligned_value_count)
        return {
            "aligned_evaluations": self.aligned_evaluations,
            "delta_zero_evaluations": self.delta_zero_evaluations,
            "dynamic_h_evaluations": self.dynamic_h_evaluations,
            "fixed_h_evaluations": self.fixed_h_evaluations,
            "small_h_fallbacks": self.small_h_fallbacks,
            "avg_comparison_horizon": self.h_p_sum / n_align,
            "prefix_rollouts": self.prefix_rollouts,
            "avg_prefix_rollout_length": self.prefix_rollout_total_length / n_roll,
            "prefix_rollouts_reached_goal": self.prefix_rollouts_reached_goal,
            "prefix_rollouts_dead_end": self.prefix_rollouts_dead_end,
            "prefix_goal_rate": self.prefix_rollouts_reached_goal / n_roll,
            "avg_suffix_ptrpg_value": (
                self.suffix_value_sum / self.suffix_value_count
                if self.suffix_value_count
                else 0.0
            ),
            "prefix_rollout_time_sec": self.prefix_rollout_time_sec,
            "cache_hits": self.cache_hits,
            "budget_fallbacks": self.budget_fallbacks,
            "avg_aligned_value": self.aligned_value_sum / n_align,
            "avg_baseline_value": (
                self.baseline_value_sum / self.baseline_value_count
                if self.baseline_value_count
                else None
            ),
        }


class RolloutAlignedEvaluator:
    """Owns the rollout-aligned evaluation, its budget, cache and diagnostics."""

    def __init__(
        self,
        config: RolloutAlignedConfig,
        raw_eval_fn: RawEvalFn,
        prefix_rollout_fn: PrefixRolloutFn,
        state_hash_fn: Optional[StateHashFn] = None,
    ):
        self.config = config
        self.raw_eval_fn = raw_eval_fn
        self.prefix_rollout_fn = prefix_rollout_fn
        self.state_hash_fn = state_hash_fn
        self.diagnostics = RolloutAlignedDiagnostics()
        self._cache: Dict[Hashable, float] = {}
        # Per-search budget counters (reset via reset_search_budget()).
        self._search_rollouts = 0
        self._search_time = 0.0

    def reset_search_budget(self) -> None:
        """Reset the per-search rollout/time budget (call at each MCTS decision)."""
        self._search_rollouts = 0
        self._search_time = 0.0

    # -- main entry ---------------------------------------------------------

    def evaluate(
        self,
        state: State,
        remaining_horizon: int,
        h_override: Optional[int] = None,
    ) -> float:
        """Aligned value of a node at remaining horizon R.

        ``h_override`` is the parent-local comparison horizon H_p = min over the
        parent's children of their remaining horizon. When ``use_dynamic_H`` is
        on and an override is given, the node is aligned to H_p; otherwise the
        fixed ``common_horizon_H`` is used.
        """
        cfg = self.config
        R = max(0, int(remaining_horizon))
        self.diagnostics.aligned_evaluations += 1

        H, raw_fallback = self._resolve_horizon(state, R, h_override)
        if raw_fallback:  # min_dynamic_horizon + fallback_if_H_too_small == raw
            value = self.raw_eval_fn(state, R)
            self._record_aligned(state, R, H, value)
            return value
        delta = R - H

        # Δ == 0: node already at (or below) the comparison horizon -> raw PTRPG.
        if delta <= 0:
            self.diagnostics.delta_zero_evaluations += 1
            value = self.raw_eval_fn(state, H)
            self._record_aligned(state, R, H, value)
            return value

        cache_key = None
        if cfg.cache_aligned_values and self.state_hash_fn is not None:
            # H is part of the key: the value is parent-local (depends on H_p).
            cache_key = (
                self.state_hash_fn(state),
                R,
                H,
                int(cfg.redo),
                cfg.prefix_rollout_policy,
                cfg.boundary_mode,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.diagnostics.cache_hits += 1
                return cached

        affordable = self._affordable_rollouts()
        if affordable <= 0:
            self.diagnostics.budget_fallbacks += 1
            value = self._fallback(state, R, H)
            self._record_aligned(state, R, H, value)
            if cache_key is not None:
                self._cache[cache_key] = value
            return value

        values = []
        for _ in range(affordable):
            if self._search_time_exhausted():
                break
            t0 = time.perf_counter()
            state2, reached_goal, length, dead_end = self.prefix_rollout_fn(state, delta)
            elapsed = time.perf_counter() - t0
            self.diagnostics.prefix_rollout_time_sec += elapsed
            self._search_time += elapsed
            self.diagnostics.prefix_rollouts += 1
            self._search_rollouts += 1
            self.diagnostics.prefix_rollout_total_length += int(length)

            if reached_goal:
                self.diagnostics.prefix_rollouts_reached_goal += 1
                values.append(1.0)
            elif dead_end:
                self.diagnostics.prefix_rollouts_dead_end += 1
                values.append(0.0)
            else:
                suffix_value = self.raw_eval_fn(state2, H)
                self.diagnostics.suffix_value_sum += suffix_value
                self.diagnostics.suffix_value_count += 1
                values.append(suffix_value)

        if not values:
            self.diagnostics.budget_fallbacks += 1
            value = self._fallback(state, R, H)
        else:
            value = sum(values) / len(values)

        self._record_aligned(state, R, H, value)
        if cache_key is not None:
            self._cache[cache_key] = value
        return value

    def _resolve_horizon(self, state, R: int, h_override):
        """Pick the comparison horizon H. Returns (H, raw_fallback_flag)."""
        cfg = self.config
        fixed = min(int(cfg.common_horizon_H), R)
        if cfg.use_dynamic_H and h_override is not None:
            H = max(0, int(h_override))
            if cfg.cap_dynamic_h_at_fixed:
                H = min(H, int(cfg.common_horizon_H))
            H = min(H, R)
            self.diagnostics.dynamic_h_evaluations += 1
            # Safety floor on a too-small dynamic horizon.
            if cfg.min_dynamic_horizon is not None and H < int(cfg.min_dynamic_horizon):
                self.diagnostics.small_h_fallbacks += 1
                if cfg.fallback_if_H_too_small == SMALL_H_RAW:
                    return H, True
                if cfg.fallback_if_H_too_small == SMALL_H_FIXED:
                    H = fixed
                # SMALL_H_USE_ANYWAY: keep H as-is.
            return H, False
        self.diagnostics.fixed_h_evaluations += 1
        return fixed, False

    # -- budget -------------------------------------------------------------

    def _affordable_rollouts(self) -> int:
        cfg = self.config
        affordable = max(0, int(cfg.redo))
        if cfg.max_prefix_rollouts_per_node > 0:
            affordable = min(affordable, cfg.max_prefix_rollouts_per_node)
        if cfg.max_prefix_rollouts_per_search > 0:
            remaining = cfg.max_prefix_rollouts_per_search - self._search_rollouts
            affordable = min(affordable, max(0, remaining))
        if self._search_time_exhausted():
            affordable = 0
        return affordable

    def _search_time_exhausted(self) -> bool:
        cap = self.config.max_prefix_rollout_time_per_search
        return cap > 0.0 and self._search_time >= cap

    def _fallback(self, state: State, R: int, H: int) -> float:
        if self.config.fallback_mode == FALLBACK_RAW:
            return self.raw_eval_fn(state, R)
        # horizon_capped (default): evaluate raw PTRPG at the common horizon.
        return self.raw_eval_fn(state, min(R, H))

    def _record_aligned(self, state: State, R: int, H: int, value: float) -> None:
        self.diagnostics.aligned_value_sum += value
        self.diagnostics.aligned_value_count += 1
        self.diagnostics.h_p_sum += int(H)
        if self.config.track_baseline_comparison:
            baseline = self.raw_eval_fn(state, R)
            self.diagnostics.baseline_value_sum += baseline
            self.diagnostics.baseline_value_count += 1
