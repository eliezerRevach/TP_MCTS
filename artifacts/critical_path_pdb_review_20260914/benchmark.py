"""Read-only benchmark of the existing heuristic; no solver modifications."""
import cProfile
import json
import math
import os
from pathlib import Path
import pstats
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0]]  # repository imports parse argv
import unified_planning as up
import unified_planning.shortcuts
import unified_planning.domains
from unified_planning.engines.convert_problem import Convert_problem
from unified_planning.engines.mdp import MDP
from comdp_plus_no_deadline.engines.critical_path_pdb import CriticalPathPDB
from comdp_plus_no_deadline.engines.temporal_stn_pdb import EngineModel, goal_relevance_closure

OUT = Path(__file__).parent
results = []

def make(garbage=0):
    domain = up.domains.Prob_Conc('regular', 25, garbage_amount=garbage)
    ground = up.engines.Grounder()._compile(domain.problem).problem
    converted = Convert_problem(ground).converted_problem
    return domain, MDP(converted, 0.95)

def report(data):
    results.append(data)
    print(json.dumps(data, sort_keys=True), flush=True)
    (OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

def measure(horizon, profile=False):
    domain, mdp = make()
    model = EngineModel(mdp)
    pdb = CriticalPathPDB(model, horizon)
    root = (frozenset(mdp.initial_state().predicates), ())
    t = time.perf_counter()
    pdb.build([root])
    build = time.perf_counter() - t
    prof = cProfile.Profile() if profile else None
    t = time.perf_counter()
    if prof: prof.enable()
    pdb.solve()
    if prof: prof.disable()
    solve = time.perf_counter() - t
    t = time.perf_counter()
    value = pdb.value(root[0], horizon)
    lookup = time.perf_counter() - t
    data = dict(horizon=horizon, profiled=profile, build_s=build, solve_s=solve,
                lookup_s=lookup, value=value, **pdb.stats())
    data['closed_form'] = math.prod(1-(1-p)**(horizon//d) for d,p in [(8,1),(4,.7),(2,.49),(1,.3)])
    data['fact_sets'] = len({s[0] for s in pdb.nodes})
    data['facts_used'] = sorted(map(str, set().union(*(s[0] for s in pdb.nodes))))
    data['queue_lengths'] = dict(Counter(len(s[1]) for s in pdb.nodes))
    data['entries_by_queue_length'] = dict(Counter({k:sum(len(v) for s,v in pdb.entries.items() if len(s[1])==k) for k in range(5)}))
    data['durative_actions'] = {k:str(op.duration) for k,op in model.ops().items()}
    data['compiled_action_count'] = len(mdp.problem.actions)
    data['initial_true_facts'] = list(map(str, root[0]))
    data['max_facts_true_in_state'] = max(map(lambda s:len(s[0]), pdb.nodes))
    data['root_entries'] = len(pdb.entries[root])
    data['start_gap_branches'] = sum(len(n.starts) for n in pdb.nodes.values())
    data['noop_branches'] = sum(n.noop is not None for n in pdb.nodes.values())
    report(data)
    if prof:
        prof.dump_stats(str(OUT / f'profile_D{horizon}.prof'))
        with (OUT / f'profile_D{horizon}.txt').open('w', encoding='utf-8') as stream:
            pstats.Stats(prof, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(35)
    return model

if __name__ == '__main__':
    for h in (8,16,25):
        print(f'Starting unprofiled D={h}', flush=True)
        measure(h)
    for goal in sorted(make()[1].problem.goals, key=str):
        _, mdp = make()
        full = EngineModel(mdp)
        allowed = goal_relevance_closure(full, goal)
        model = EngineModel(mdp, allowed_ops=allowed, goals={goal})
        pdb = CriticalPathPDB(model, 25)
        t = time.perf_counter()
        value = pdb.value(mdp.initial_state().predicates, 25)
        report(dict(single_goal=str(goal), elapsed_s=time.perf_counter()-t, value=value,
                    actions=sorted(allowed), **pdb.stats()))
    print('Starting profiled D=16', flush=True)
    measure(16, profile=True)
