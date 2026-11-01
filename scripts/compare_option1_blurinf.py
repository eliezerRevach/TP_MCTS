"""Option 1 (backward pieces over the (facts, end queue) graph) against the windows table at blur inf
((facts, queue, c) states), on the SAME pattern models.

    TP_MCTS_WILAO_BLUR=1000 python scripts/compare_option1_blurinf.py --domain prob_conc --garbage_amount 7 --deadline 10

Option 1 is ``critical_path_pdb.CriticalPathPDB`` (Temporal_PDB_Backward_Critical_Path5): nodes (F, Q); each node keeps
entries (T, one tail per running end, p), regressed backward from the goal (an end opens its tail, a start closes
it: T = max(T, d + tail)); outcome = sum, choice = union with dominance. Here it is run on the windows PDB's own
pattern models, with the two things the windows solver does for patterns: an end is legal by the pattern's rule
(outside facts free) and an event whose probability reads an outside fact gets one option per value (VI picks).

For every pattern table the windows PDB builds (with this run's blur), one [opt1] line: both sizes, both build
times, and the root's value at every r = 0..D from each (both are upper bounds on the pattern; lower is tighter).
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from comdp_plus_no_deadline.engines.critical_path_pdb import CriticalPathPDB, Entry, prune  # noqa: E402
from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic  # noqa: E402
from comdp_plus_no_deadline.engines.windows_lao import WindowsTable  # noqa: E402
from prebuild_wilao_tables import make_mdp  # noqa: E402

ROWS = []


class OptionOne(CriticalPathPDB):
    """CriticalPathPDB on a fact pattern: the pattern's end rule and one option per outcome variant."""

    def _end_legal(self, op, facts):
        check = getattr(self.model, "end_legal_op", None)
        return check(op, facts) if check is not None else super()._end_legal(op, facts)

    def _variants(self, op, which, facts):
        getter = getattr(self.model, "outcome_variants", None)
        if getter is None or isinstance(op.start_action, str):
            return (self._outcomes(op, which, facts),)
        return getter(op.start_action if which == "S" else op.end_action, facts)

    def _expand(self, state, node):
        facts, queue = state
        ops = self.model.ops()
        for name in self.model.legal_action_names(facts):
            op = ops.get(name)
            if op is None:
                continue
            variants = self._variants(op, "S", facts)
            if op.end_action is None:
                for outs in variants:
                    node.starts.append((name, None, self._link(state, outs, queue)))
                continue
            durations = tuple(ops[k].duration for k in queue)
            for gap in range(len(queue) + 1):
                if self._gap_ok(durations, op.duration, gap):
                    new_queue = queue[:gap] + (name,) + queue[gap:]
                    for outs in variants:
                        node.starts.append((name, gap, self._link(state, outs, new_queue)))
        node.noop = []                                 # one noop per end variant
        if queue:
            op = ops.get(queue[0])
            if op is not None and op.end_action is not None and self._end_legal(op, facts):
                node.noop = [self._link(state, outs, queue[1:]) for outs in self._variants(op, "E", facts)]
        node.expanded = True
        self.expanded_count += 1
        self._dirty = True

    def _backup(self, state):
        node = self.nodes[state]
        k = len(state[1])
        candidates = []
        for key, gap, branches in node.starts:
            d = self._units[key]
            candidates.extend(self._outcome_sum(k, [
                (p, [self._regress_start(e, gap, d) for e in self.entries[child]]) for p, child in branches]))
        for branches in node.noop or ():
            candidates.extend(self._outcome_sum(k, [
                (p, [Entry(e.T, (e.T,) + e.tails, e.p) for e in self.entries[child]]) for p, child in branches]))
        return prune(candidates)


_orig_build = WindowsTable.build


def compared_build(self, root_facts, time_budget=None):
    t0 = time.perf_counter()
    ok = _orig_build(self, root_facts, time_budget)
    table_s = time.perf_counter() - t0
    D = int(self.horizon)
    row = dict(table_built=ok, table_states=len(self.states) if ok else None, table_s=round(table_s, 2))
    t0 = time.perf_counter()
    opt = OptionOne(self.lao.model, self.horizon, max_states=MAX_STATES, max_updates=MAX_UPDATES)
    opt.build([(frozenset(root_facts), ())])
    opt.solve()
    opt_s = time.perf_counter() - t0
    sizes = [len(v) for v in opt.entries.values()]
    row.update(opt1_nodes=len(opt.nodes), opt1_entries=sum(sizes), opt1_max_entries=max(sizes, default=0),
               opt1_complete=opt.complete, opt1_converged=opt.converged, opt1_s=round(opt_s, 2))
    curve_t = [round(self.lookup(root_facts, r)[0], 6) if ok else None for r in range(D + 1)]
    curve_o = [round(opt.state_value((frozenset(root_facts), ()), r), 6) for r in range(D + 1)]
    row.update(root_table=curve_t, root_opt1=curve_o,
               max_abs_diff=max(abs(a - b) for a, b in zip(curve_t, curve_o)) if ok else None,
               opt1_tighter_at=sum(1 for a, b in zip(curve_t, curve_o) if ok and b < a - 1e-9),
               table_tighter_at=sum(1 for a, b in zip(curve_t, curve_o) if ok and a < b - 1e-9))
    ROWS.append(row)
    print("[opt1] " + json.dumps(row), flush=True)
    return ok


def main():
    global MAX_STATES, MAX_UPDATES
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--object_amount", type=int, default=1)
    ap.add_argument("--garbage_amount", type=int, default=0)
    ap.add_argument("--deadline", type=int, required=True)
    ap.add_argument("--max_states", type=int, default=2_000_000)
    ap.add_argument("--max_updates", type=int, default=50_000_000)
    args = ap.parse_args()
    MAX_STATES, MAX_UPDATES = args.max_states, args.max_updates
    if (os.environ.get("TP_MCTS_WILAO_TABLE_CACHE") or "").strip():
        raise SystemExit("unset TP_MCTS_WILAO_TABLE_CACHE: a cache hit would skip the table builds")
    WindowsTable.build = compared_build
    mdp = make_mdp(args.domain, args.object_amount, args.garbage_amount, args.deadline)
    WindowsILAOPDBHeuristic(mdp).build()
    total = dict(domain=args.domain, object_amount=args.object_amount, garbage=args.garbage_amount,
                 deadline=args.deadline, growth=os.environ.get("TP_MCTS_WILAO_PATTERN_GROWTH"),
                 blur=os.environ.get("TP_MCTS_WILAO_BLUR"), tables=len(ROWS))
    for key in ("table_states", "table_s", "opt1_nodes", "opt1_entries", "opt1_s"):
        total[key] = round(sum(r.get(key) or 0 for r in ROWS), 2)
    total["opt1_max_entries"] = max((r["opt1_max_entries"] for r in ROWS), default=0)
    total["all_complete"] = all(r["opt1_complete"] and r["opt1_converged"] for r in ROWS)
    total["max_abs_diff"] = max((r["max_abs_diff"] or 0 for r in ROWS), default=0)
    print("[opt1-total] " + json.dumps(total), flush=True)


if __name__ == "__main__":
    main()
