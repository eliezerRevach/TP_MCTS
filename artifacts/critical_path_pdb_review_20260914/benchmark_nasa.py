"""Bounded NASA rover measurements of the unchanged critical-path PDB."""
import _thread
from collections import Counter
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0]]
import unified_planning as up
import unified_planning.shortcuts
import unified_planning.domains
from unified_planning.engines.convert_problem import Convert_problem
from unified_planning.engines.mdp import MDP
from comdp_plus_no_deadline.engines.critical_path_pdb import CriticalPathPDB
from comdp_plus_no_deadline.engines.temporal_stn_pdb import EngineModel, goal_relevance_closure

OUT = Path(__file__).parent
CASE = os.environ.get('NASA_PDB_CASE', 'full')
LIMIT = float(os.environ.get('NASA_PDB_SECONDS', '180'))

class ProgressPDB(CriticalPathPDB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_report = time.perf_counter()

    def _expand(self, state, node):
        super()._expand(state, node)
        now = time.perf_counter()
        if now - self.last_report >= 15:
            print(json.dumps(dict(case=CASE, phase='build', states=len(self.nodes),
                                  expanded=self.expanded_count)), flush=True)
            self.last_report = now

    def _backup(self, state):
        result = super()._backup(state)
        now = time.perf_counter()
        if now - self.last_report >= 15:
            print(json.dumps(dict(case=CASE, phase='solve', updates=self.updates,
                                  entries=sum(map(len, self.entries.values())))), flush=True)
            self.last_report = now
        return result

def main():
    setup_start = time.perf_counter()
    domain = up.domains.Nasa_Rover('regular', 25, 1)
    ground = up.engines.Grounder(domain.grounding_map())._compile(domain.problem).problem
    mdp = MDP(Convert_problem(ground).converted_problem, .95)
    full = EngineModel(mdp)
    goals = full.goals
    if CASE == 'image':
        goals = {g for g in goals if str(g).startswith('communicated_image_data')}
    elif CASE == 'rock':
        goals = {g for g in goals if str(g) == 'communicated_rock_data(x0)'}
    elif CASE != 'full':
        raise ValueError(CASE)
    allowed = None if CASE == 'full' else set().union(*(goal_relevance_closure(full,g) for g in goals))
    model = EngineModel(mdp, goals=goals, allowed_ops=allowed)
    pdb = ProgressPDB(model, 25)
    root = (frozenset(mdp.initial_state().predicates), ())
    ops = model.ops()
    info = dict(case=CASE, object_amount=1, horizon=25, time_limit_s=LIMIT,
                goals=sorted(map(str,goals)), ops=len(ops),
                durative_actions=sum(o.end_action is not None for o in ops.values()),
                instantaneous_actions=sum(o.end_action is None for o in ops.values()),
                full_model_ground_boolean_atoms=len(mdp.problem.initial_values),
                initial_true_facts=sorted(map(str,root[0])),
                setup_s=time.perf_counter()-setup_start,
                max_expanded_states=pdb.max_states)
    print(json.dumps(info), flush=True)
    started = time.perf_counter()
    phase = 'build'
    build_s = None
    solve_s = None
    status = 'complete'
    value = None
    timer = threading.Timer(LIMIT, _thread.interrupt_main)
    timer.daemon = True
    timer.start()
    try:
        pdb.build([root])
        build_s = time.perf_counter()-started
        print(json.dumps(dict(case=CASE, build_s=build_s, **pdb.stats())), flush=True)
        phase = 'solve'
        solve_start = time.perf_counter()
        pdb.solve()
        solve_s = time.perf_counter()-solve_start
        value = pdb.state_value(root,25)
        if not pdb.complete:
            status = 'state_budget_cut'
        elif not pdb.converged:
            status = 'update_budget_cut'
    except KeyboardInterrupt:
        status = 'timeout'
    finally:
        timer.cancel()
    elapsed = time.perf_counter()-started
    union = set()
    intersection = None
    for facts, queue in pdb.nodes:
        union.update(facts)
        if intersection is None:
            intersection = set(facts)
        else:
            intersection.intersection_update(facts)
    varying = union - (intersection or set())
    data = dict(**info, status=status, stopped_phase=phase, elapsed_s=elapsed,
                build_s=build_s, solve_s=solve_s, value=value, **pdb.stats(),
                facts_seen=sorted(map(str,union)),
                facts_varying_in_seen_states=sorted(map(str,varying)),
                max_running_seen=max((len(s[1]) for s in pdb.nodes), default=0))
    if status == 'timeout':
        # The solver's converged flag is set on entry and is not a timeout signal.
        data['converged'] = None
    (OUT / f'nasa_{CASE}.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k not in ('facts_seen','facts_varying_in_seen_states','initial_true_facts')}), flush=True)
    print(json.dumps(dict(facts_seen=len(union), varying_facts_seen=len(varying))), flush=True)

if __name__ == '__main__':
    main()
