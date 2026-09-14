"""Count repeated backup work without changing the heuristic's decisions."""
import json
import time
from collections import Counter, deque
from benchmark import make, OUT, EngineModel, CriticalPathPDB, goal_relevance_closure
import comdp_plus_no_deadline.engines.critical_path_pdb as cp

class CountedPDB(CriticalPathPDB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.work_counts = Counter()
        self.seen_children = {}
        self.branch_times = Counter()

    def _backup(self, state):
        self.work_counts['backups'] += 1
        node = self.nodes[state]
        branches = [b for _, _, b in node.starts]
        if node.noop is not None:
            branches.append(node.noop)
        unchanged_flags = []
        for i, branch in enumerate(branches):
            self.work_counts['branch_backups'] += 1
            child_lists = tuple(self.entries[c] for _, c in branch)
            previous = self.seen_children.get((state, i))
            unchanged = previous is not None and all(a is b for a,b in zip(previous, child_lists))
            unchanged_flags.append(unchanged)
            if unchanged:
                self.work_counts['branch_backups_with_unchanged_children'] += 1
            self.seen_children[state, i] = child_lists
        self._branch_unchanged = deque(unchanged_flags)
        return super()._backup(state)

    def _outcome_sum(self, k, per_outcome):
        unchanged = self._branch_unchanged.popleft()
        start = time.perf_counter()
        result = super()._outcome_sum(k, per_outcome)
        elapsed = time.perf_counter() - start
        self.branch_times['all_branch_sum_s'] += elapsed
        if unchanged:
            self.branch_times['unchanged_branch_sum_s'] += elapsed
        return result

if __name__ == '__main__':
    counts = Counter()
    original = cp.prune
    def counted_prune(entries):
        entries = list(entries)
        counts['prune_calls'] += 1
        counts['candidate_entries_passed_to_prune'] += len(entries)
        counts['largest_single_prune_input'] = max(counts['largest_single_prune_input'], len(entries))
        return original(entries)
    cp.prune = counted_prune
    _, mdp = make()
    pdb = CountedPDB(EngineModel(mdp), 16)
    start = time.perf_counter()
    value = pdb.value(mdp.initial_state().predicates, 16)
    data = dict(horizon=16, instrumented=True, value=value,
                elapsed_s=time.perf_counter()-start, **counts, **pdb.work_counts,
                **pdb.branch_times)
    print(json.dumps(data, indent=2), flush=True)
    extras = []
    prior = OUT / 'work_counts.json'
    if prior.exists():
        extras = json.loads(prior.read_text(encoding='utf-8')).get('garbage_structure_checks', [])
    for garbage in (() if extras else (0, 1, 2)):
        _, mdp = make(garbage)
        model = EngineModel(mdp)
        pdb = CriticalPathPDB(model, 25)
        start = time.perf_counter()
        pdb.build([(frozenset(mdp.initial_state().predicates), ())])
        allowed = set().union(*(goal_relevance_closure(model, g) for g in model.goals))
        extras.append(dict(garbage_actions=garbage, total_durative_actions=len(model.ops()),
                           build_only_s=time.perf_counter()-start, states=len(pdb.nodes),
                           facts_used=len(set().union(*(s[0] for s in pdb.nodes))),
                           actions_after_explicit_goal_closure=sorted(allowed)))
    data['garbage_structure_checks'] = extras
    (OUT / 'work_counts.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps(extras, indent=2), flush=True)
