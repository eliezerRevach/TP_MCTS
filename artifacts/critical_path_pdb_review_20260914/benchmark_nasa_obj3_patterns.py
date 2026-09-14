"""NASA rover, obj=3, D=25: one independently built PDB for each goal."""
import _thread
import json
from pathlib import Path
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.argv = [sys.argv[0]]
import benchmark_nasa as base

OUT = Path(__file__).parent
RESULT = OUT / 'nasa_obj3_patterns.json'
OBJECTS = 3
HORIZON = 25
TIME_LIMIT = 180.0

def save(summary):
    RESULT.write_text(json.dumps(summary, indent=2), encoding='utf-8')

def main():
    start = time.perf_counter()
    domain = base.up.domains.Nasa_Rover('regular', HORIZON, OBJECTS)
    ground = base.up.engines.Grounder(domain.grounding_map())._compile(domain.problem).problem
    mdp = base.MDP(base.Convert_problem(ground).converted_problem, .95)
    full = base.EngineModel(mdp)
    goals = sorted(full.goals, key=str)
    initial = frozenset(mdp.initial_state().predicates)
    summary = dict(domain='nasa_rover', kind='regular', object_amount=OBJECTS,
                   horizon=HORIZON, pattern_count=len(goals),
                   pattern_definition='one goal, existing goal_relevance_closure action restriction, full facts retained',
                   full_model_ops=len(full.ops()),
                   full_model_declared_boolean_atoms=len(mdp.problem.initial_values),
                   initial_true_facts=sorted(map(str, initial)),
                   setup_s=time.perf_counter()-start,
                   per_pattern_time_limit_s=TIME_LIMIT,
                   patterns=[])
    print(json.dumps({k:v for k,v in summary.items() if k != 'initial_true_facts'}), flush=True)
    save(summary)
    retained = []
    batch_started = time.perf_counter()
    for index, goal in enumerate(goals, 1):
        base.CASE = str(goal)
        begin = time.perf_counter()
        allowed = base.goal_relevance_closure(full, goal)
        model = base.EngineModel(mdp, goals={goal}, allowed_ops=allowed)
        pdb = base.ProgressPDB(model, HORIZON)
        root = (initial, ())
        ops = model.ops()
        preparation_s = time.perf_counter()-begin
        print(json.dumps(dict(starting_pattern=index, goal=str(goal), ops=len(ops))), flush=True)
        timer = threading.Timer(TIME_LIMIT, _thread.interrupt_main)
        timer.daemon = True
        phase = 'build'
        status = 'complete'
        build_s = solve_s = value = None
        started = time.perf_counter()
        timer.start()
        try:
            pdb.build([root])
            build_s = time.perf_counter()-started
            print(json.dumps(dict(goal=str(goal), phase='built', build_s=build_s,
                                  states=len(pdb.nodes), complete=pdb.complete)), flush=True)
            phase = 'solve'
            solve_start = time.perf_counter()
            pdb.solve()
            solve_s = time.perf_counter()-solve_start
            value = pdb.state_value(root, HORIZON)
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
        common = set(initial)
        for facts, _ in pdb.nodes:
            union.update(facts)
            common.intersection_update(facts)
        varying = union-common
        flags = {f for f in varying if str(f).startswith('inExecution(')}
        result = dict(goal=str(goal), status=status, stopped_phase=phase,
                      preparation_s=preparation_s, build_s=build_s, solve_s=solve_s,
                      elapsed_s=elapsed, value=value,
                      ops=len(ops), durative_actions=sum(o.end_action is not None for o in ops.values()),
                      instantaneous_actions=sum(o.end_action is None for o in ops.values()),
                      varying_domain_facts=len(varying)-len(flags), execution_flags=len(flags),
                      constant_true_facts=len(common), facts_seen=len(union),
                      varying_fact_names=sorted(map(str,varying)),
                      allowed_ops=sorted(allowed), max_running_seen=max(len(s[1]) for s in pdb.nodes),
                      max_expanded_states=pdb.max_states, **pdb.stats())
        if status == 'timeout':
            result['converged'] = None
        summary['patterns'].append(result)
        save(summary)
        print(json.dumps({k:v for k,v in result.items() if k not in ('varying_fact_names','allowed_ops')}), flush=True)
        retained.append((goal, pdb))
    summary['sequential_batch_wall_s'] = time.perf_counter()-batch_started
    summary['sum_build_s'] = sum(r['build_s'] or 0 for r in summary['patterns'])
    summary['sum_solve_s'] = sum(r['solve_s'] or 0 for r in summary['patterns'])
    summary['sum_pattern_elapsed_s'] = sum(r['elapsed_s'] for r in summary['patterns'])
    summary['sum_preparation_s'] = sum(r['preparation_s'] for r in summary['patterns'])
    summary['total_states'] = sum(r['states'] for r in summary['patterns'])
    summary['total_entries'] = sum(r['entries'] for r in summary['patterns'])
    summary['all_complete'] = all(r['status']=='complete' for r in summary['patterns'])
    if summary['all_complete']:
        timings = []
        for _ in range(31):
            lookup_start = time.perf_counter()
            values = [pdb.value(initial,HORIZON) for _,pdb in retained]
            timings.append(time.perf_counter()-lookup_start)
        summary['cached_all_patterns_lookup_median_s'] = statistics.median(timings)
        summary['cached_lookup_values_match'] = all(abs(v-r['value']) < 1e-12 for v,r in zip(values,summary['patterns']))
    save(summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('initial_true_facts','patterns')}), flush=True)

if __name__ == '__main__':
    main()
