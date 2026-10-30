"""Layers vs intervals (TP_MCTS_WILAO_VI) on real windows-PDB tables: ONE forward build per pattern, then
both backward passes on the same graph -- each timed, every cell compared, the value memory of each.

    python scripts/compare_wilao_vi.py --domain nasa_rover --object_amount 2 --deadline 25

The MDP is made as in prebuild_wilao_tables.py; the WILAO settings come from the environment (export the
notebook's TP_MCTS_WILAO_* values, without TP_MCTS_WILAO_TABLE_CACHE so the tables are really built).
Prints one [vi] line per pattern table and a [vi-total] line:
    states, layers R, ring W, forward / layers / intervals seconds, max |layers - intervals| over every
    cell, value bytes (dense array vs intervals), pieces kept, h(s0) through the interval tables.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic  # noqa: E402
from comdp_plus_no_deadline.engines.windows_lao import WindowsTable, _concat_ranges  # noqa: E402
from prebuild_wilao_tables import make_mdp  # noqa: E402

ROWS = []


def dense_of(iv) -> np.ndarray:
    """IntervalValues as the (R, n) array, vectorised (IntervalValues.dense loops per state)."""
    R, n = iv.shape
    out = np.zeros(R * n)                             # flat, state-major: i * R + k
    state = np.repeat(np.arange(n), np.diff(iv.ptr))
    ks = iv.ks.astype(np.int64)
    nxt = np.append(ks[1:], R)
    last = np.append(state[1:] != state[:-1], True)
    end = np.where(last, R, nxt)
    end = np.minimum(end, iv.first[state].astype(np.int64))
    start = np.minimum(ks, end)
    idx = _concat_ranges(state * R + start, state * R + end)
    out[idx] = np.repeat(iv.vals, end - start)
    tail = _concat_ranges(np.arange(n) * R + np.minimum(iv.first.astype(np.int64), R), np.arange(n) * R + R)
    out[tail] = 1.0
    return out.reshape(n, R).T


def peak_rss_gb():
    try:
        import resource                               # Linux (the VM); None elsewhere
    except ImportError:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


_orig_backward = WindowsTable._backward


def both(self):
    t0 = time.perf_counter()
    self.vi = "layers"
    m_layers = _orig_backward(self)
    layers_s = time.perf_counter() - t0
    dense = self.V
    t0 = time.perf_counter()
    self.vi = "intervals"
    m_int = _orig_backward(self)
    int_s = time.perf_counter() - t0
    iv = self.V
    row = dict(states=len(self.states), layers_R=int(dense.shape[0]), forward_s=None,
               layers_s=round(layers_s, 2), intervals_s=round(int_s, 2), method_layers=m_layers,
               method_intervals=m_int, dense_bytes=int(dense.nbytes), intervals_bytes=int(iv.nbytes))
    if hasattr(iv, "ks"):
        row.update(pieces=int(len(iv.vals)), max_abs_diff=float(np.abs(dense_of(iv) - dense).max()),
                   ring_W=int(self.stats.get("ring_W", 0)) or None)
    else:
        row.update(pieces=None, max_abs_diff=float(np.abs(iv - dense).max()))
    ROWS.append((self, row))
    return m_int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--object_amount", type=int, default=1)
    ap.add_argument("--garbage_amount", type=int, default=0)
    ap.add_argument("--deadline", type=int, required=True)
    args = ap.parse_args()
    if (os.environ.get("TP_MCTS_WILAO_TABLE_CACHE") or "").strip():
        raise SystemExit("unset TP_MCTS_WILAO_TABLE_CACHE: a cache hit would skip the builds being timed")
    WindowsTable._backward = both
    t0 = time.perf_counter()
    mdp = make_mdp(args.domain, args.object_amount, args.garbage_amount, args.deadline)
    pdb = WindowsILAOPDBHeuristic(mdp)
    pdb.build()
    total_s = time.perf_counter() - t0
    h0 = pdb.heuristic_score(mdp.initial_state(), mdp.problem.goals, fixed_depth=args.deadline, start_time=0.0,
                             running_remaining={})
    tag = dict(domain=args.domain, object_amount=args.object_amount, deadline=args.deadline,
               growth=os.environ.get("TP_MCTS_WILAO_PATTERN_GROWTH"), blur=os.environ.get("TP_MCTS_WILAO_BLUR"))
    for table, row in ROWS:
        row["forward_s"] = table.stats.get("forward_seconds")
        print("[vi] " + json.dumps(dict(tag, **row), default=str), flush=True)
    rows = [r for _t, r in ROWS]
    total = dict(tag, tables=len(rows), states=sum(r["states"] for r in rows),
                 forward_s=round(sum(r["forward_s"] or 0 for r in rows), 1),
                 layers_s=round(sum(r["layers_s"] for r in rows), 1),
                 intervals_s=round(sum(r["intervals_s"] for r in rows), 1),
                 dense_bytes=sum(r["dense_bytes"] for r in rows), intervals_bytes=sum(r["intervals_bytes"] for r in rows),
                 max_abs_diff=max((r["max_abs_diff"] for r in rows), default=0.0),
                 h_s0_intervals=round(h0, 6), build_total_s=round(total_s, 1),
                 peak_rss_gb=peak_rss_gb())
    print("[vi-total] " + json.dumps(total, default=str), flush=True)


if __name__ == "__main__":
    main()
