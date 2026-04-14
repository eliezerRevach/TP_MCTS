"""
Lightweight per-call runtime metrics for the parallel heuristic.

Usage
-----
    from unified_planning.engines.heuristic_timing import reset_metrics, get_metrics

    reset_metrics()          # start a fresh measurement run
    ... run planner ...
    m = get_metrics()
    print(m.summary())

Two measurement levels are tracked independently:
  - *wrapper*: one entry per `_heuristic_value()` call in greedy_parallel.py
  - *worker*: one entry per `heuristic_score()` / `_temporal_heuristic()` call,
              tagged with whether the result was a query-cache hit.

Cache semantics
---------------
The `TemporalProbabilisticRPGHeuristic._query_cache` is keyed by
(state_facts, depth, start_layer, strategy).  A hit means the full
propagation graph was skipped; only a dict copy was returned.  First call
for any (state, depth, time, strategy) tuple is always a miss.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HeuristicCallMetrics:
    """Accumulates per-call timing for wrapper and worker heuristic levels."""

    # --- wrapper level (_heuristic_value) ---
    wrapper_times: List[float] = field(default_factory=list)

    # --- worker level (_temporal_heuristic / heuristic_score) ---
    worker_times: List[float] = field(default_factory=list)
    worker_cache_hits: List[bool] = field(default_factory=list)

    def record_wrapper(self, elapsed: float) -> None:
        self.wrapper_times.append(elapsed)

    def record_worker(self, elapsed: float, cache_hit: bool) -> None:
        self.worker_times.append(elapsed)
        self.worker_cache_hits.append(cache_hit)

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict of aggregated metrics ready for display."""

        def _stats(times: List[float], label: str) -> dict:
            n = len(times)
            if n == 0:
                return {f"{label}_total_calls": 0}
            total = sum(times)
            first = times[0]
            avg = total / n
            return {
                f"{label}_total_calls": n,
                f"{label}_total_time_sec": round(total, 6),
                f"{label}_first_call_sec": round(first, 6),
                f"{label}_avg_call_sec": round(avg, 6),
            }

        result = {}
        result.update(_stats(self.wrapper_times, "wrapper"))
        result.update(_stats(self.worker_times, "worker"))

        # Cache-hit average (worker level only).
        hit_times = [
            t for t, h in zip(self.worker_times, self.worker_cache_hits) if h
        ]
        miss_times = [
            t for t, h in zip(self.worker_times, self.worker_cache_hits) if not h
        ]
        result["worker_cache_hits"] = len(hit_times)
        result["worker_cache_misses"] = len(miss_times)
        if hit_times:
            result["worker_cache_hit_avg_sec"] = round(sum(hit_times) / len(hit_times), 6)
        if miss_times:
            result["worker_cache_miss_avg_sec"] = round(sum(miss_times) / len(miss_times), 6)

        return result

    def report(self) -> str:
        """Human-readable report string."""
        s = self.summary()
        lines = ["=== Heuristic Per-Call Runtime Report ==="]

        def _fmt_level(prefix: str, label: str) -> None:
            n = s.get(f"{prefix}_total_calls", 0)
            if n == 0:
                lines.append(f"  [{label}] No calls recorded.")
                return
            total = s.get(f"{prefix}_total_time_sec", 0.0)
            first = s.get(f"{prefix}_first_call_sec", 0.0)
            avg = s.get(f"{prefix}_avg_call_sec", 0.0)
            lines.append(f"  [{label}]")
            lines.append(f"    total calls : {n}")
            lines.append(f"    total time  : {total:.4f} s")
            lines.append(f"    first call  : {first:.4f} s")
            lines.append(f"    avg / call  : {avg:.6f} s  ({total:.4f} / {n} = {avg:.6f})")

        _fmt_level("wrapper", "Wrapper  (_heuristic_value)")
        _fmt_level("worker", "Worker   (heuristic_score)")

        hits = s.get("worker_cache_hits", 0)
        misses = s.get("worker_cache_misses", 0)
        if hits or misses:
            lines.append(f"  [Worker cache breakdown]")
            lines.append(f"    cache hits   : {hits}")
            lines.append(f"    cache misses : {misses}")
            if "worker_cache_hit_avg_sec" in s:
                lines.append(f"    hit avg      : {s['worker_cache_hit_avg_sec']:.6f} s")
            if "worker_cache_miss_avg_sec" in s:
                lines.append(f"    miss avg     : {s['worker_cache_miss_avg_sec']:.6f} s")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_active_metrics: Optional[HeuristicCallMetrics] = None


def reset_metrics() -> HeuristicCallMetrics:
    """Start a fresh metrics run.  Returns the new collector."""
    global _active_metrics
    _active_metrics = HeuristicCallMetrics()
    return _active_metrics


def get_metrics() -> Optional[HeuristicCallMetrics]:
    """Return the current collector, or None if not started."""
    return _active_metrics


def is_active() -> bool:
    return _active_metrics is not None


# ---------------------------------------------------------------------------
# Context-manager helpers used by instrumented call sites
# ---------------------------------------------------------------------------

class _WrapperTimer:
    """Use as: `with _WrapperTimer(): ...`"""

    __slots__ = ("_t0", "_elapsed")

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self._elapsed = time.perf_counter() - self._t0
        m = _active_metrics
        if m is not None:
            m.record_wrapper(self._elapsed)


class _WorkerTimer:
    """Use as: `with _WorkerTimer() as wt: ...; wt.hit = result.cache_hit`"""

    __slots__ = ("_t0", "_elapsed", "hit")

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.hit = False
        return self

    def __exit__(self, *_):
        self._elapsed = time.perf_counter() - self._t0
        m = _active_metrics
        if m is not None:
            m.record_worker(self._elapsed, self.hit)


WrapperTimer = _WrapperTimer
WorkerTimer = _WorkerTimer
