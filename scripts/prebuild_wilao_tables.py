"""Build windows_ilao_pdb tables into TP_MCTS_WILAO_TABLE_CACHE ahead of a Script 3 run, one scenario
after another on ONE core -- so a heavy build runs while the MCTS jobs of other scenarios use the
other cores, and that scenario's workers later only load the file (cache hit).

    python scripts/prebuild_wilao_tables.py --domain machine_shop --object_amount 2 --deadlines 17 27 40

The MDP is made the way run_domain.py's run_regular makes it (same problem -> same cache key). The
WILAO settings come from the environment, exactly as for run_domain.py: export the same
TP_MCTS_WILAO_* values the notebook config sets. Prints one line per scenario:
    [prebuild] {domain, object_amount, garbage_amount, deadline, growth, build_seconds, h_s0, report}
build_seconds is 0-ish when the file was already there (cache hit).
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import unified_planning as up  # noqa: E402
from unified_planning.shortcuts import *  # noqa: E402,F401,F403
import unified_planning.domains  # noqa: E402,F401
from unified_planning.engines.convert_problem import Convert_problem  # noqa: E402
from unified_planning.engines.mdp import MDP  # noqa: E402

from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic  # noqa: E402

DOMAINS = dict(machine_shop=up.domains.Machine_Shop, nasa_rover=up.domains.Nasa_Rover, stuck_car=up.domains.Stuck_Car,
               conc=up.domains.Conc, prob_conc=up.domains.Prob_Conc, simple=up.domains.Simple,
               hosting=up.domains.Hosting)


def make_mdp(domain, object_amount, garbage_amount, deadline):
    model = DOMAINS[domain](kind="regular", deadline=deadline, object_amount=object_amount,
                            garbage_amount=garbage_amount)
    grounder = (up.engines.compilers.Grounder(model.grounding_map()) if domain in ("nasa_rover", "robot_lab")
                else up.engines.compilers.Grounder())
    ground = grounder._compile(model.problem).problem
    return MDP(Convert_problem(ground)._converted_problem, discount_factor=1.0, reward_mode="terminal",
               step_penalty=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--object_amount", type=int, default=1)
    ap.add_argument("--garbage_amount", type=int, default=0)
    ap.add_argument("--deadlines", type=int, nargs="+", required=True)
    args = ap.parse_args()
    if not (os.environ.get("TP_MCTS_WILAO_TABLE_CACHE") or "").strip():
        raise SystemExit("TP_MCTS_WILAO_TABLE_CACHE is not set: a prebuild without a cache is lost")
    for deadline in args.deadlines:
        t0 = time.perf_counter()
        mdp = make_mdp(args.domain, args.object_amount, args.garbage_amount, deadline)
        pdb = WindowsILAOPDBHeuristic(mdp)
        pdb.build()
        seconds = round(time.perf_counter() - t0, 1)
        # h(s0) through the real lookup, as TP-MCTS asks it at the root (nothing running)
        h0 = pdb.heuristic_score(mdp.initial_state(), mdp.problem.goals, fixed_depth=deadline, start_time=0.0,
                                 running_remaining={})
        report = dict(pdb.offline_report)
        line = {"domain": args.domain, "object_amount": args.object_amount, "garbage_amount": args.garbage_amount,
                "deadline": deadline, "growth": report.get("growth"), "build_seconds": seconds,
                "h_s0": round(h0, 6), "report": report}
        print("[prebuild] " + json.dumps(line, default=str), flush=True)


if __name__ == "__main__":
    main()
