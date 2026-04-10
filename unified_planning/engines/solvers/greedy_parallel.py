import math
from typing import Dict

import unified_planning as up
from unified_planning.engines.solvers.mcts import _temporal_heuristic
from unified_planning.engines.utils import create_init_stn, update_stn


def _heuristic_value(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    current_time: float,
    heuristic_name: str,
    temporal_heuristic_depth: int,
    temporal_heuristic_strategy: str,
    temporal_cache_table=None,
    return_cache_table: bool = False,
):
    if heuristic_name == "temporal_probabilistic_rpg":
        return _temporal_heuristic(
            mdp,
            state,
            current_time,
            temporal_heuristic_depth,
            temporal_heuristic_strategy,
            cached_table=temporal_cache_table,
            return_cache_table=return_cache_table,
        )
    heuristic = up.engines.heuristics.TRPG(mdp, state, current_time)
    value = heuristic.get_heuristic()
    if return_cache_table:
        return value, None
    return value


def _score_action(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    action: "up.engines.Action",
    heuristic_name: str,
    temporal_heuristic_depth: int,
    temporal_heuristic_strategy: str,
    heuristic_weight: float,
    temporal_cache_table=None,
):
    candidate_stn = stn.clone()
    candidate_prev = previous_action_node
    try:
        candidate_prev = update_stn(
            candidate_stn,
            action,
            candidate_prev,
            type="SetTime",
        )
    except Exception:
        return -math.inf, None, None, None, False, {}

    if not candidate_stn.is_consistent():
        return -math.inf, None, None, None, False, {}
    if candidate_stn.get_current_end_time() > mdp.deadline():
        return -math.inf, None, None, None, False, {}

    current_time = candidate_stn.get_current_end_time()
    # Expected-value scoring over probabilistic outcomes.
    transitions = mdp.transition_function(state, action)
    if not transitions:
        return -math.inf, None, None, None, False, {}

    step_penalty = -0.01
    expected_reward = 0.0
    expected_h = 0.0
    any_terminal = False
    transition_cache_by_state: Dict["up.engines.State", object] = {}
    for next_state, prob in transitions:
        terminal = mdp.is_terminal(next_state)
        any_terminal = any_terminal or terminal
        expected_reward += prob * (mdp.terminal_reward(terminal, next_state) + step_penalty)
        if (
            heuristic_name == "temporal_probabilistic_rpg"
            and temporal_heuristic_strategy == "baseline_cached"
        ):
            h_value, candidate_cache = _heuristic_value(
                mdp=mdp,
                state=next_state,
                current_time=current_time,
                heuristic_name=heuristic_name,
                temporal_heuristic_depth=temporal_heuristic_depth,
                temporal_heuristic_strategy=temporal_heuristic_strategy,
                temporal_cache_table=temporal_cache_table,
                return_cache_table=True,
            )
            transition_cache_by_state[next_state] = candidate_cache
            expected_h += prob * h_value
        else:
            expected_h += prob * _heuristic_value(
                mdp=mdp,
                state=next_state,
                current_time=current_time,
                heuristic_name=heuristic_name,
                temporal_heuristic_depth=temporal_heuristic_depth,
                temporal_heuristic_strategy=temporal_heuristic_strategy,
            )

    score = expected_reward + heuristic_weight * expected_h - 0.001 * current_time
    # Do not return a sampled next_state here; sampling must only happen for the chosen action.
    return score, None, candidate_stn, candidate_prev, any_terminal, transition_cache_by_state


def plan(
    mdp: "up.engines.MDP",
    steps: int,
    search_time: int,
    search_depth: int,
    exploration_constant: float,
    selection_type: str = "avg",
    k: int = 10,
    heuristic_name: str = "trpg",
    temporal_heuristic_depth: int = 25,
    temporal_heuristic_strategy: str = "baseline",
):
    """
    Heuristic-only greedy dispatcher (no MCTS tree).

    At each decision step, greedily dispatches a feasible set of actions while
    staying STN-consistent. Action scoring is immediate reward + heuristic value.
    """
    del search_time, search_depth, exploration_constant, selection_type, k

    stn = create_init_stn(mdp)
    root_state = mdp.initial_state()
    previous_action_node = None
    max_parallel_set_size = 32
    heuristic_weight = 0.2
    step = 0
    temporal_cache_table = None

    while stn.get_current_end_time() <= mdp.deadline() and step < steps:
        print(f"started step {step}")
        decision_time = stn.get_current_end_time()
        chosen_in_set = 0

        while chosen_in_set < max_parallel_set_size:
            legal_actions = mdp.legal_actions(root_state)
            if not legal_actions:
                break

            best = None
            best_score = -math.inf
            for action in legal_actions:
                candidate = _score_action(
                    mdp=mdp,
                    state=root_state,
                    stn=stn,
                    previous_action_node=previous_action_node,
                    action=action,
                    heuristic_name=heuristic_name,
                    temporal_heuristic_depth=temporal_heuristic_depth,
                    temporal_heuristic_strategy=temporal_heuristic_strategy,
                    heuristic_weight=heuristic_weight,
                    temporal_cache_table=temporal_cache_table,
                )
                if candidate[0] > best_score:
                    best_score = candidate[0]
                    best = (action, candidate)

            if best is None or not math.isfinite(best_score):
                break

            action, (_, _, next_stn, next_prev, _, transition_cache_by_state) = best
            print(f"Current state is {root_state}")
            print(f"The chosen action is {action.name}")

            terminal, next_state, _ = mdp.step(root_state, action)
            if (
                heuristic_name == "temporal_probabilistic_rpg"
                and temporal_heuristic_strategy == "baseline_cached"
            ):
                temporal_cache_table = transition_cache_by_state.get(next_state)
            root_state = next_state
            stn = next_stn
            previous_action_node = next_prev
            chosen_in_set += 1

            print(f"The time of the plan so far: {stn.get_current_end_time()}")
            if terminal:
                print(f"Current state is {root_state}")
                print(f"The amount of time the plan took: {stn.get_current_end_time()}")
                return 1, stn.get_current_end_time()

            # Keep each dispatch step as one parallel time slice.
            if stn.get_current_end_time() > decision_time:
                break

        if chosen_in_set == 0:
            print("A valid plan is not found")
            return 0, -math.inf
        step += 1

    print("A valid plan is not found")
    return 0, -math.inf
