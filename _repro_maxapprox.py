import os, sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unified_planning as up
from unified_planning.shortcuts import *

# Defensive up.args so module-level CLI reads don't crash.
up.args = SimpleNamespace(
    discount_factor=0.95, reward_mode="deadline", step_penalty=-0.01, seed=0,
    deadline=35, temporal_heuristic_depth=None,
    resolution_alpha=2.0, resolution_forced_minimum=False, resolution_reference_t=None,
    object_amount=2, garbage_amount=0,
)

import unified_planning.domains
from unified_planning.engines.convert_problem_combination import Convert_problem_combination
from unified_planning.engines.mdp import combinationMDP
from unified_planning.engines.utils import create_init_stn
from unified_planning.engines.solvers.max_approximation_selector import (
    MaxApproximationConfig, build_heuristic_adapter, select_max_approximation_action_set,
)

DEADLINE, OBJ, GARB = 35, 2, 0

model = up.domains.Nasa_Rover(kind='combination', deadline=DEADLINE, object_amount=OBJ, garbage_amount=GARB)
grounder = up.engines.compilers.Grounder(model.grounding_map())
ground_problem = grounder._compile(model.problem).problem
ccp = Convert_problem_combination(model, ground_problem)
converted = ccp._converted_problem
model.remove_actions(converted)
converted.set_deadline(up.model.timing.Timing(delay=DEADLINE,
    timepoint=up.model.timing.Timepoint(up.model.timing.TimepointKind.START)))

mdp = combinationMDP(converted, discount_factor=0.95, reward_mode="deadline", step_penalty=-0.01)
state = mdp.initial_state()
stn = create_init_stn(mdp)
legal = mdp.legal_actions(state)
print("deadline:", mdp.deadline(), "| #legal actions at root:", len(legal))
print("goals:", sorted(str(g) for g in mdp.problem.goals)[:8], "...")

res_kwargs = dict(resolution_alpha=2.0, resolution_forced_minimum=False, resolution_reference_t=None)
adapter = build_heuristic_adapter(
    mdp, heuristic_name="temporal_probabilistic_rpg",
    temporal_heuristic_strategy="atom_backtrack_exact_resolution",
    temporal_heuristic_depth=35, resolution_kwargs=res_kwargs,
)

ct = float(stn.get_current_end_time())
scores = adapter.action_scores(state, legal, remaining_deadline=float(DEADLINE), current_time=ct)
nz = {k: v for k, v in scores.items() if v > 0.0}
print(f"\n=== action_scores at ROOT: {len(scores)} actions, {len(nz)} with score>0 ===")
for k, v in sorted(nz.items(), key=lambda kv: -kv[1])[:15]:
    print(f"   {v:.4f}  {k}")

cfg = MaxApproximationConfig(alpha=1.5, num_samples=32, seed=0, debug=False)
action_set, dbg = select_max_approximation_action_set(
    mdp, state, stn, None, legal, adapter, float(DEADLINE), cfg)
nonempty = [s for s in dbg.sampled_sets if s]
print(f"\n=== selector result ===")
print("best_set:", dbg.best_set, "| best_value:", dbg.best_value)
print(f"sampled sets: {len(dbg.sampled_sets)} total, {len(nonempty)} non-empty")
print("=> RETURNS EMPTY SET" if not action_set else f"=> returns {len(action_set)} actions")
