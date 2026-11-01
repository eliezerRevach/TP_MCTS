"""How big is the MDP over (facts, end queue) alone -- no windows, no c, no deadline -- next to the windows
table that is built today, for the same pattern models.

    python scripts/count_fq_states.py --domain nasa_rover --object_amount 2 --deadline 25

For every pattern table the windows PDB builds, ONE run reports:
    fq_states / fq_edges : the (F, Q) graph: from the pattern's root, every start (any gap of the queue) and
                           every first end, until no new node appears (no time, so no deadline cut and no
                           order check); goals are not expanded, as in the table
    table_states         : the windows table today (F, Q, windows), with this run's TP_MCTS_WILAO_BLUR
    table_fq             : distinct (F, Q) among those table states
The options come from the real generator (WindowsLAO._rest_options) with windows that never reject a gap.
The WILAO settings come from the environment (set TP_MCTS_WILAO_PATTERN_GROWTH, BLUR, ...; no table cache).
"""

import argparse
import json
import os
import sys
import time
from fractions import Fraction

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic  # noqa: E402
from comdp_plus_no_deadline.engines.windows_lao import WindowsTable  # noqa: E402
from prebuild_wilao_tables import make_mdp  # noqa: E402

FREE = (Fraction(0), Fraction(10 ** 9), Fraction(0))     # a window that accepts every gap
ROWS = []


def fq_graph(lao, root_facts, cap):
    """(nodes, edges, seconds, capped) of the (F, Q) graph from root_facts."""
    t0 = time.perf_counter()
    goals = lao.model.goals
    root = (frozenset(root_facts), ())
    seen, todo, edges = {root}, [root], 0
    while todo:
        facts, queue = todo.pop()
        if goals <= facts:
            continue
        for _label, _charge, _inert, q2, _w2, outs in lao._rest_options(facts, queue, (FREE,) * len(queue)):
            for f2, p in outs:
                if p <= 1e-12:
                    continue
                edges += 1
                key = (frozenset(f2), q2)
                if key not in seen:
                    seen.add(key)
                    todo.append(key)
                    if len(seen) >= cap:
                        return len(seen), edges, round(time.perf_counter() - t0, 1), True
    return len(seen), edges, round(time.perf_counter() - t0, 1), False


_orig_build = WindowsTable.build


def counted_build(self, root_facts, time_budget=None):
    row = {}
    n, e, s, capped = fq_graph(self.lao, root_facts, CAP)
    row.update(fq_states=n, fq_edges=e, fq_seconds=s, fq_capped=capped)
    t0 = time.perf_counter()
    ok = _orig_build(self, root_facts, time_budget)
    row.update(table_built=ok, table_seconds=round(time.perf_counter() - t0, 1))
    if ok:
        row.update(table_states=len(self.states), table_fq=len({(f, q) for f, q, _w in self.states}))
    ROWS.append(row)
    print("[fq] " + json.dumps(row), flush=True)
    return ok


def main():
    global CAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--object_amount", type=int, default=1)
    ap.add_argument("--garbage_amount", type=int, default=0)
    ap.add_argument("--deadline", type=int, required=True)
    ap.add_argument("--cap", type=int, default=30_000_000, help="stop the (F, Q) graph at this many nodes")
    args = ap.parse_args()
    CAP = args.cap
    if (os.environ.get("TP_MCTS_WILAO_TABLE_CACHE") or "").strip():
        raise SystemExit("unset TP_MCTS_WILAO_TABLE_CACHE: a cache hit would skip the table builds")
    WindowsTable.build = counted_build
    mdp = make_mdp(args.domain, args.object_amount, args.garbage_amount, args.deadline)
    WindowsILAOPDBHeuristic(mdp).build()
    total = dict(domain=args.domain, object_amount=args.object_amount, deadline=args.deadline,
                 growth=os.environ.get("TP_MCTS_WILAO_PATTERN_GROWTH"), blur=os.environ.get("TP_MCTS_WILAO_BLUR"),
                 tables=len(ROWS), any_capped=any(r["fq_capped"] for r in ROWS))
    for key in ("fq_states", "fq_edges", "fq_seconds", "table_states", "table_fq", "table_seconds"):
        total[key] = round(sum(r.get(key) or 0 for r in ROWS), 1)
    print("[fq-total] " + json.dumps(total), flush=True)


if __name__ == "__main__":
    main()
