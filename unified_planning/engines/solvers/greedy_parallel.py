import math
from typing import Dict, List, Optional, Tuple

import unified_planning as up
from unified_planning.engines.solvers.mcts import _temporal_heuristic, _uses_tprpg_family
from unified_planning.engines.utils import create_init_stn, update_stn
from unified_planning.engines.heuristic_timing import WrapperTimer, is_active

GREEDY_HEURISTIC_WEIGHT = 0.2
GREEDY_MAX_PARALLEL_SET_SIZE = 32
GREEDY_DEFAULT_TIME_SLICES = 90


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
    with WrapperTimer() if is_active() else _null_ctx():
        if _uses_tprpg_family(heuristic_name):
            return _temporal_heuristic(
                mdp,
                state,
                current_time,
                temporal_heuristic_depth,
                temporal_heuristic_strategy,
                cached_table=temporal_cache_table,
                return_cache_table=return_cache_table,
                leaf_heuristic_name=heuristic_name,
            )
        heuristic = up.engines.heuristics.TRPG(mdp, state, current_time)
        value = heuristic.get_heuristic()
        if return_cache_table:
            return value, None
        return value


class _null_ctx:
    """No-op context manager used when metrics are disabled."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


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

    step_penalty = getattr(mdp, "step_penalty", -0.01)
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


def _action_name_key(action: "up.engines.Action") -> str:
    name = getattr(action, "name", None)
    return name if name is not None else str(action)


ActionScoreEntry = Tuple[
    float,
    "up.engines.Action",
    "up.plans.stn.STNPlan",
    "up.plans.stn.STNPlanNode",
    bool,
    Dict,
]


def rank_actions_by_score(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    legal_actions: List["up.engines.Action"],
    heuristic_name: str,
    temporal_heuristic_depth: int,
    temporal_heuristic_strategy: str,
    heuristic_weight: float,
    temporal_cache_table=None,
    tie_break: str = "stable",
) -> List[ActionScoreEntry]:
    ranked: List[ActionScoreEntry] = []
    for action in legal_actions:
        score, _, candidate_stn, candidate_prev, any_terminal, transition_cache = _score_action(
            mdp=mdp,
            state=state,
            stn=stn,
            previous_action_node=previous_action_node,
            action=action,
            heuristic_name=heuristic_name,
            temporal_heuristic_depth=temporal_heuristic_depth,
            temporal_heuristic_strategy=temporal_heuristic_strategy,
            heuristic_weight=heuristic_weight,
            temporal_cache_table=temporal_cache_table,
        )
        if math.isfinite(score):
            ranked.append(
                (score, action, candidate_stn, candidate_prev, any_terminal, transition_cache)
            )
    if tie_break == "stable":
        ranked.sort(key=lambda item: (-item[0], _action_name_key(item[1])))
    return ranked


def pick_best_action(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    heuristic_name: str,
    temporal_heuristic_depth: int,
    temporal_heuristic_strategy: str,
    heuristic_weight: float = GREEDY_HEURISTIC_WEIGHT,
    temporal_cache_table=None,
    tie_break: str = "stable",
    legal_actions: Optional[List["up.engines.Action"]] = None,
) -> Optional[ActionScoreEntry]:
    if legal_actions is None:
        legal_actions = mdp.legal_actions(state)
    if not legal_actions:
        return None
    if tie_break == "stable":
        ranked = rank_actions_by_score(
            mdp=mdp,
            state=state,
            stn=stn,
            previous_action_node=previous_action_node,
            legal_actions=legal_actions,
            heuristic_name=heuristic_name,
            temporal_heuristic_depth=temporal_heuristic_depth,
            temporal_heuristic_strategy=temporal_heuristic_strategy,
            heuristic_weight=heuristic_weight,
            temporal_cache_table=temporal_cache_table,
            tie_break="stable",
        )
        return ranked[0] if ranked else None

    best: Optional[ActionScoreEntry] = None
    best_score = -math.inf
    for action in legal_actions:
        candidate = _score_action(
            mdp=mdp,
            state=state,
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
            best = (
                candidate[0],
                action,
                candidate[2],
                candidate[3],
                candidate[4],
                candidate[5],
            )
    return best


def greedy_matched_value_target(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    action: "up.engines.Action",
    current_time: float,
    heuristic_name: str,
    temporal_heuristic_depth: int,
    temporal_heuristic_strategy: str,
    heuristic_weight: float = GREEDY_HEURISTIC_WEIGHT,
):
    """
    Expected one-step greedy score for a fixed (state, action), reusing the
    same shaping form as the greedy wrapper.

    Terminal successors contribute terminal reward only (no step penalty and no
    heuristic bonus) for TP-MCTS greedy_matched backups.
    """
    transitions = mdp.transition_function(state, action)
    if not transitions:
        return -math.inf

    step_penalty = getattr(mdp, "step_penalty", -0.01)
    expected_reward = 0.0
    expected_h = 0.0
    for next_state, prob in transitions:
        terminal = mdp.is_terminal(next_state)
        if terminal:
            expected_reward += prob * mdp.terminal_reward(True, next_state)
            continue

        expected_reward += prob * (mdp.terminal_reward(False, next_state) + step_penalty)
        expected_h += prob * _heuristic_value(
            mdp=mdp,
            state=next_state,
            current_time=current_time,
            heuristic_name=heuristic_name,
            temporal_heuristic_depth=temporal_heuristic_depth,
            temporal_heuristic_strategy=temporal_heuristic_strategy,
        )

    return expected_reward + heuristic_weight * expected_h - 0.001 * current_time


def terminal_success_value(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
) -> float:
    if not mdp.is_terminal(state):
        return 0.0
    if stn.get_current_end_time() > mdp.deadline():
        return 0.0
    return 1.0


def simulate_greedy_mdp_until_terminal(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    heuristic_name: str,
    temporal_heuristic_depth: int,
    temporal_heuristic_strategy: str,
    *,
    max_time_slices: int = GREEDY_DEFAULT_TIME_SLICES,
    max_parallel_set_size: int = GREEDY_MAX_PARALLEL_SET_SIZE,
    max_mdp_steps: Optional[int] = None,
    tie_break: str = "legacy",
    heuristic_weight: float = GREEDY_HEURISTIC_WEIGHT,
    temporal_cache_table=None,
    on_mdp_step=None,
) -> Tuple[float, "up.plans.stn.STNPlan"]:
    """Greedy MDP dispatcher (same as ``plan()``) until goal, dead end, or deadline.

    Returns ``(terminal_success_value, final_stn)``. Optional ``on_mdp_step`` callback:
    ``(mdp_step_index, state, stn, action, terminal)``.
    """
    if max_mdp_steps is None:
        max_mdp_steps = max(1, max_parallel_set_size * max(1, max_time_slices))

    current_state = state
    current_stn = stn
    current_prev = previous_action_node
    cache = temporal_cache_table
    mdp_step = 0
    time_slice = 0

    while current_stn.get_current_end_time() <= mdp.deadline() and time_slice < max_time_slices:
        decision_time = current_stn.get_current_end_time()
        chosen_in_set = 0

        while chosen_in_set < max_parallel_set_size:
            if mdp_step >= max_mdp_steps:
                return (
                    terminal_success_value(mdp, current_state, current_stn),
                    current_stn,
                )

            if mdp.is_terminal(current_state):
                return (
                    terminal_success_value(mdp, current_state, current_stn),
                    current_stn,
                )

            legal_actions = mdp.legal_actions(current_state)
            if not legal_actions:
                break

            picked = pick_best_action(
                mdp=mdp,
                state=current_state,
                stn=current_stn,
                previous_action_node=current_prev,
                heuristic_name=heuristic_name,
                temporal_heuristic_depth=temporal_heuristic_depth,
                temporal_heuristic_strategy=temporal_heuristic_strategy,
                heuristic_weight=heuristic_weight,
                temporal_cache_table=cache,
                tie_break=tie_break,
                legal_actions=legal_actions,
            )
            if picked is None:
                break

            best_score, action, next_stn, next_prev, _, transition_cache_by_state = picked
            if not math.isfinite(best_score):
                break

            terminal, next_state, _reward = mdp.step(current_state, action)
            if (
                heuristic_name == "temporal_probabilistic_rpg"
                and temporal_heuristic_strategy == "baseline_cached"
            ):
                cache_update = transition_cache_by_state.get(next_state)
                if isinstance(cache_update, tuple):
                    if cache is not None:
                        temp_dict, new_state_facts = cache_update
                        cache.apply_temp(temp_dict, new_state_facts)
                else:
                    cache = cache_update

            if on_mdp_step is not None:
                on_mdp_step(mdp_step, current_state, current_stn, action, terminal)

            current_state = next_state
            current_stn = next_stn
            current_prev = next_prev
            chosen_in_set += 1
            mdp_step += 1

            if terminal:
                return (
                    terminal_success_value(mdp, current_state, current_stn),
                    current_stn,
                )

            if current_stn.get_current_end_time() > decision_time:
                break

        if chosen_in_set == 0:
            return (
                terminal_success_value(mdp, current_state, current_stn),
                current_stn,
            )
        time_slice += 1

    return (
        terminal_success_value(mdp, current_state, current_stn),
        current_stn,
    )


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

    def _log_step(_idx, _state, _stn, action, terminal):
        print(f"Current state is {_state}")
        print(f"The chosen action is {action.name}")
        print(f"The time of the plan so far: {_stn.get_current_end_time()}")
        if terminal:
            print(f"Current state is {_state}")
            print(f"The amount of time the plan took: {_stn.get_current_end_time()}")

    value, stn = simulate_greedy_mdp_until_terminal(
        mdp=mdp,
        state=root_state,
        stn=stn,
        previous_action_node=None,
        heuristic_name=heuristic_name,
        temporal_heuristic_depth=temporal_heuristic_depth,
        temporal_heuristic_strategy=temporal_heuristic_strategy,
        max_time_slices=steps,
        max_parallel_set_size=GREEDY_MAX_PARALLEL_SET_SIZE,
        tie_break="legacy",
        heuristic_weight=GREEDY_HEURISTIC_WEIGHT,
        on_mdp_step=_log_step,
    )
    if value >= 1.0:
        return 1, stn.get_current_end_time()
    print("A valid plan is not found")
    return 0, -math.inf
