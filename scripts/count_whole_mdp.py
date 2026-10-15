"""Size of the whole MDP that LAO* searches -- every action, every outcome, no heuristic, no pruning --
to measure what the admissible heuristic saves (LAO* nodes / whole-MDP nodes).

    python scripts/count_whole_mdp.py --domain stuck_car --object_amount 1 --deadline 10 --budget 1800 --json out.json

whole_mdp_nodes    : distinct (state, STN) nodes, the same node identity LAO* uses. Exact when the
                     expansion finishes inside --budget, else a lower bound (the nodes reached).
whole_mdp_expanded : nodes whose actions were all expanded.
tree_nodes_exact   : the same MDP as a tree (one node per path, like TP-MCTS's tree); finished runs only.
Out of memory is reported like the time limit (finished = false, stopped = "out of memory").
"""
import argparse
import json
import os
import sys
import threading
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--object_amount", type=int, default=1)
    ap.add_argument("--garbage_amount", type=int, default=0)
    ap.add_argument("--deadline", type=int, required=True)
    ap.add_argument("--budget", type=float, default=1800.0, help="seconds (0 = no limit)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]                       # unified_planning parses sys.argv on import
    sys.path.insert(0, REPO)
    os.chdir(REPO)

    import importlib.util
    spec = importlib.util.spec_from_file_location("runner", os.path.join(REPO, "scripts", "run_lao_tp_mdp.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    mdp = runner.build_mdp(args.domain, args.object_amount, args.garbage_amount, args.deadline)
    from comdp_plus_no_deadline.engines.lao_tp_mdp import TPMDPLAO

    lao = TPMDPLAO(mdp, heuristic=None, time_budget=args.budget)
    t0 = time.perf_counter()

    def ticker():
        while True:
            time.sleep(30)
            print(f"[{time.perf_counter() - t0:7.0f}s] nodes={len(lao.nodes)} expanded={lao.stats['expanded']}",
                  flush=True)
    threading.Thread(target=ticker, daemon=True).start()

    stopped = ""
    try:
        finished = lao.expand_all()
        if not finished:
            stopped = f"not done after {args.budget:.0f} s"
    except MemoryError:
        finished, stopped = False, "out of memory"
    out = dict(domain=args.domain, object_amount=args.object_amount, garbage_amount=args.garbage_amount,
               deadline=args.deadline, finished=finished, stopped=stopped,
               seconds=round(time.perf_counter() - t0, 1),
               whole_mdp_nodes=len(lao.nodes), whole_mdp_expanded=lao.stats["expanded"])
    if finished:
        def children(n):
            return [k for _name, branches in n.options for _p, k in branches] if n.options else []
        tree = {}
        stack = [(lao.root.key, False)]
        while stack:                               # post-order over the graph: children first
            k, ready = stack.pop()
            if k in tree:
                continue
            ch = children(lao.nodes[k])
            if ready:
                tree[k] = 1 + sum(tree[c] for c in ch)
                continue
            stack.append((k, True))
            stack.extend((c, False) for c in ch if c not in tree)
        out["tree_nodes_exact"] = tree[lao.root.key]
    lao.nodes.clear()
    print(json.dumps(out, indent=1), flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
