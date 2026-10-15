"""Solve TP-MCTS's MDP offline with ILAO* (comdp_plus_no_deadline/engines/lao_tp_mdp.py) and report
the optimal value from the initial state, runtime and node counts.

    python scripts/run_lao_tp_mdp.py --domain prob_conc --garbage_amount 7 --deadline 10
    python scripts/run_lao_tp_mdp.py --domain nasa_rover --object_amount 2 --deadline 25
    python scripts/run_lao_tp_mdp.py --domain stuck_car --object_amount 1 --deadline 10 --heuristic none

--heuristic pdb  : the windows PDB, with the TP_MCTS_WILAO_* settings from the environment
                   (defaults below = the experiment settings: pairs, min, grouping, blur 3)
--heuristic none : h = 1 everywhere (blind control: same optimum, shows what the PDB saves)
--reject_after   : a run not done after this many seconds in total (MDP + PDB build + search) is
                   rejected and reported as such (default 600 = 10 min); the PDB has no budget of its own
"""

import argparse
import json
import os
import sys
import threading
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PDB_DEFAULTS = dict(
    TP_MCTS_WILAO_OFFLINE_SECONDS="0", TP_MCTS_WILAO_OFFLINE_EXTEND="1", TP_MCTS_WILAO_PATTERN_GOALS="1",
    TP_MCTS_WILAO_PATTERN_FACTS="10", TP_MCTS_WILAO_MISS="one", TP_MCTS_WILAO_AGG="min",
    TP_MCTS_WILAO_GROUPING="1", TP_MCTS_WILAO_PATTERN_GROWTH="pairs", TP_MCTS_WILAO_BLUR="3",
    TP_MCTS_WILAO_EXACT_STATICS="1", TP_MCTS_WILAO_NO_STN_PASS="0", TP_MCTS_WILAO_REPORT="0",
)


def build_mdp(domain, object_amount, garbage_amount, deadline):
    """The regular MDP exactly as unified_planning/run_domain.py run_regular builds it."""
    import unified_planning as up
    import unified_planning.domains  # noqa: F401
    from unified_planning.engines.convert_problem import Convert_problem
    from unified_planning.engines.mdp import MDP

    # the table of unified_planning/run_domain.py (that module runs an experiment on import);
    # a domain class this checkout does not have is simply not offered
    names = dict(machine_shop="Machine_Shop", nasa_rover="Nasa_Rover", stuck_car_1o="Stuck_Car_1o",
                 stuck_car="Stuck_Car", conc="Conc", full_conc="Full_Conc", prob_conc="Prob_Conc",
                 best_no_parallel="Best_No_Parallel", simple="Simple", hosting="Hosting",
                 prob_match_cellar="Prob_MatchCellar", robot_lab="Robot_Lab")
    domains = {k: getattr(up.domains, v) for k, v in names.items() if hasattr(up.domains, v)}
    if domain not in domains:
        raise SystemExit(f"unknown domain {domain!r}; available: {sorted(domains)}")
    model = domains[domain](kind="regular", deadline=deadline, object_amount=object_amount,
                            garbage_amount=garbage_amount)
    if domain == "nasa_rover":
        grounder = up.engines.compilers.Grounder(model.grounding_map())
    else:
        grounder = up.engines.compilers.Grounder()
    ground = grounder._compile(model.problem).problem
    return MDP(Convert_problem(ground)._converted_problem, discount_factor=1.0, reward_mode="terminal",
               step_penalty=0.0)


def _peak_rss_gb():
    """Peak resident memory of this process in GB (Linux / macOS), None elsewhere."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2), 2)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--object_amount", type=int, default=1)
    ap.add_argument("--garbage_amount", type=int, default=0)
    ap.add_argument("--deadline", type=int, required=True)
    ap.add_argument("--heuristic", choices=("pdb", "none"), default="pdb")
    ap.add_argument("--max_nodes", type=int, default=0, help="stop after this many nodes (0 = no limit)")
    ap.add_argument("--time_budget", type=float, default=0.0, help="seconds (0 = no limit)")
    ap.add_argument("--json", default="", help="also write the report to this file")
    ap.add_argument("--progress", action="store_true", help="print one line per ILAO* pass")
    ap.add_argument("--exhaustive", action="store_true",
                    help="no search: expand every reachable node (the whole MDP) and report its size")
    ap.add_argument("--reject_after", type=float, default=600.0,
                    help="reject the run if it is not done after this many seconds in total (0 = never)")
    ap.add_argument("--validate_runs", type=int, default=0,
                    help="after solving, execute the policy this many times in CoMDP+ (mdp.step, times fixed"
                         " at execution as TP-MCTS does) and check every step")
    ap.add_argument("--validate_exact", action="store_true",
                    help="after solving, execute the policy over EVERY outcome (exact probability)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diagnostics", action="store_true",
                    help="per-depth report: STN-rejected actions, ties, Q spread (gradient), h-pruned options,"
                         " tightness h0 vs V* on finished runs, lower bound on V(s0) over time")
    ap.add_argument("--algorithm", choices=("ilao", "bounded"), default="ilao",
                    help="ilao = ILAO* (upper bounds); bounded = upper + lower bounds, FRTDP-style trials")
    ap.add_argument("--gap_epsilon", type=float, default=0.0,
                    help="bounded only: stop when upper(s0) - lower(s0) <= this (0 = exact)")
    ap.add_argument("--build_only", action="store_true",
                    help="build (and cache) the PDB tables, report their size and time, and stop")
    ap.add_argument("--canonical_starts", action="store_true",
                    help="two independent starts in a row only in name order (fewer orderings; check V* on/off)")
    ap.add_argument("--diagnostics_tips", type=int, default=0,
                    help="per sample, the k tips carrying most of V(s0): facts, time left, h per pattern, value"
                         " after one expansion (0 = off)")
    ap.add_argument("--diagnostics_interval", type=float, default=600.0,
                    help="seconds between lower-bound samples (0 = only at the end)")
    args = ap.parse_args()

    for k, v in PDB_DEFAULTS.items():
        os.environ.setdefault(k, v)
    sys.argv = [sys.argv[0]]                       # unified_planning parses sys.argv on import
    sys.path.insert(0, REPO)
    os.chdir(REPO)

    from comdp_plus_no_deadline.engines.lao_tp_mdp import PolicyExecution, TPMDPLAO, windows_pdb_heuristic

    run = {"phase": "mdp", "lao": None, "start": time.perf_counter()}

    def reject():
        """Over the limit: report what was reached and stop (no partial answer is passed off as one)."""
        lao = run["lao"]
        out = {"domain": args.domain, "object_amount": args.object_amount, "garbage_amount": args.garbage_amount,
               "deadline": args.deadline, "heuristic": args.heuristic, "rejected": True,
               "reason": f"not done after {args.reject_after:.0f} s", "phase": run["phase"],
               "seconds": round(time.perf_counter() - run["start"], 1)}
        if run.get("h_root") is not None:
            out["h(s0)"] = run["h_root"]
        if run.get("pdb") is not None:
            out.update({"pdb_build_seconds": run.get("pdb_seconds"), "pdb": run["pdb"]})
        if lao is not None:
            out.update({"V(s0) so far (upper bound)": round(lao.root.value, 6),
                        "nodes_generated": lao.stats["generated"], "nodes_expanded": lao.stats["expanded"],
                        "passes": lao.stats["passes"]})
        print(json.dumps(out, indent=1, default=str), flush=True)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1, default=str)
        os._exit(3)

    if args.reject_after > 0:
        # the search stops itself at reject_after and reports in full; this timer is only a backstop
        # (PDB build stuck past the limit, or the final report taking far too long)
        grace = 900.0 if args.diagnostics else 60.0
        timer = threading.Timer(args.reject_after + grace, reject)
        timer.daemon = True
        timer.start()

    t0 = time.perf_counter()
    mdp = build_mdp(args.domain, args.object_amount, args.garbage_amount, args.deadline)
    t_mdp = time.perf_counter() - t0

    h, t_pdb, pdb_report = None, 0.0, None
    if args.heuristic == "pdb":
        h = windows_pdb_heuristic(mdp)
        run["phase"] = "pdb build"
        t1 = time.perf_counter()
        if args.progress:
            print(f"[mdp built] {t_mdp:.1f}s  actions={len(mdp.problem.actions)}  -- building the PDB ...", flush=True)
        h.pdb.build()
        t_pdb = time.perf_counter() - t1
        if args.progress:
            print(f"[pdb built] {t_pdb:.1f}s  table_states={h.pdb.offline_report.get('table_states')}", flush=True)
        rep = h.pdb.offline_report
        pdb_report = {k: rep.get(k) for k in ("patterns", "pattern_facts", "growth", "table_states", "values")}
        run["pdb_seconds"], run["pdb"] = round(t_pdb, 2), pdb_report
    if args.build_only:
        out = {"domain": args.domain, "object_amount": args.object_amount, "deadline": args.deadline,
               "build_only": True, "pdb_build_seconds": round(t_pdb, 2), "pdb": pdb_report,
               "peak_rss_gb": _peak_rss_gb()}
        print(json.dumps(out, indent=1, default=str), flush=True)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1, default=str)
        os._exit(0)

    def progress(solver, seconds):
        st = solver.stats
        print(f"[pass {st['passes']}] {seconds:8.1f}s  generated={st['generated']}  expanded={st['expanded']}"
              f"  V(s0)={solver.root.value:.6f}", flush=True)

    run["phase"] = "search"
    budget = args.time_budget
    if args.reject_after > 0:
        left = max(1.0, args.reject_after - (time.perf_counter() - run["start"]))
        budget = min(budget, left) if budget else left
    lao = TPMDPLAO(mdp, heuristic=h, max_nodes=args.max_nodes, time_budget=budget,
                   on_pass=progress if args.progress else None, diagnostics=args.diagnostics,
                   diagnostics_interval=args.diagnostics_interval, canonical_starts=args.canonical_starts,
                   diagnostics_tips=args.diagnostics_tips, algorithm=args.algorithm,
                   gap_epsilon=args.gap_epsilon)
    run["lao"] = lao

    if args.progress:                              # counts every 30 s, also inside one long pass
        def ticker():
            while True:
                time.sleep(30)
                st = lao.stats
                print(f"[{time.perf_counter() - run['start']:7.0f}s] generated={st['generated']}"
                      f"  expanded={st['expanded']}  passes={st['passes']}", flush=True)
        threading.Thread(target=ticker, daemon=True).start()
    h_root = lao.root.value
    run["h_root"] = round(h_root, 6)
    if args.exhaustive:
        lao.expand_all()
        report = {"rejected": False, "domain": args.domain, "object_amount": args.object_amount,
                  "garbage_amount": args.garbage_amount, "deadline": args.deadline, "mode": "exhaustive",
                  "reachable_nodes": len(lao.nodes), "expanded": lao.stats["expanded"],
                  "goal_nodes": sum(1 for n in lao.nodes.values() if n.goal),
                  "dead_ends": sum(1 for n in lao.nodes.values() if n.dead),
                  "seconds": round(lao.stats["seconds"], 2)}
        print(json.dumps(report, indent=1), flush=True)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=1)
        return
    try:
        value = lao.solve()
    except MemoryError:
        # Over the memory limit: report what was reached, the same way as a run over the time limit.
        counts = (lao.stats["generated"], lao.stats["expanded"], lao.stats["passes"], lao.root.value)
        lao.nodes.clear()                          # free the graph first: the report itself needs memory
        run["lao"] = None
        import gc
        gc.collect()
        try:
            diag = lao.diagnostics(walk=False) if args.diagnostics else None   # counters only
        except MemoryError:
            diag = {"error": "no memory left even for the counters"}
        out = {"domain": args.domain, "object_amount": args.object_amount, "deadline": args.deadline,
               "heuristic": args.heuristic, "rejected": True, "reason": "out of memory", "phase": "search",
               "seconds": round(time.perf_counter() - run["start"], 1), "V(s0) so far (upper bound)": counts[3],
               "pdb_build_seconds": round(t_pdb, 2), "pdb": pdb_report,
               "nodes_generated": counts[0], "nodes_expanded": counts[1], "passes": counts[2]}
        if diag is not None:
            out["diagnostics"] = diag
        print(json.dumps(out, indent=1, default=str), flush=True)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1, default=str)
        os._exit(4)
    s = lao.stats
    out_of_time = not s["complete"] and lao._deadline_at is not None and time.perf_counter() >= lao._deadline_at
    report = {
        "rejected": out_of_time,
        "domain": args.domain, "object_amount": args.object_amount, "garbage_amount": args.garbage_amount,
        "deadline": args.deadline, "heuristic": args.heuristic,
        "V(s0)": round(value, 6), "h(s0)": round(h_root, 6), "optimal": s["complete"],
        "first_action": lao.best_action(),
        "nodes_generated": s["generated"], "nodes_expanded": s["expanded"], "policy": lao.policy_counts(),
        "passes": s["passes"], "backups": s["backups"],
        "search_seconds": round(s["seconds"], 2), "pdb_build_seconds": round(t_pdb, 2),
        "mdp_build_seconds": round(t_mdp, 2),
        "heuristic_calls": s["h_calls"], "heuristic_seconds": round(s["h_seconds"], 2),
        "key_seconds": round(s["key_seconds"], 2), "expand_seconds": round(s["expand_seconds"], 2),
        "inconsistent_actions": s["inconsistent"], "folded_zero_time_loops": s["folded"],
        "dropped_noops": s["dropped_noop"], "cycles": s["cycles"],
        "canonical_starts": args.canonical_starts, "canonical_skipped": s["canonical_skipped"],
        "algorithm": args.algorithm, "gap_epsilon": args.gap_epsilon,
        "upper_bound(s0)": round(lao.root.value, 6), "lower_bound(s0)": round(getattr(lao.root, "lower", 0.0), 6),
        "trials": s["trials"], "cut_options": s["cut_options"],
        "pdb": pdb_report,
    }
    report["peak_rss_gb"] = _peak_rss_gb()
    if out_of_time:
        report["reason"] = f"not done after {args.reject_after:.0f} s"
        report["V(s0) so far (upper bound)"] = report.pop("V(s0)")
    if args.diagnostics:
        run["phase"] = "diagnostics"
        report["diagnostics"] = lao.diagnostics(walk=True)
    if (args.validate_runs or args.validate_exact) and s["complete"]:
        run["phase"] = "policy execution"
        execution = PolicyExecution(lao)
        t2 = time.perf_counter()
        if args.validate_exact:
            report["execution_exact"] = execution.exact()
        if args.validate_runs:
            report["execution_sampled"] = execution.run(args.validate_runs, seed=args.seed)
        report["execution_seconds"] = round(time.perf_counter() - t2, 2)
    print(json.dumps(report, indent=1, default=str), flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
