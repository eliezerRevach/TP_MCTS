from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import csv
import random
from typing import Dict, List, Optional, Sequence, Tuple
from collections import deque

import numpy as np
import unified_planning as up
from unified_planning.shortcuts import Timing, Timepoint, TimepointKind
from unified_planning.engines.compilers import Grounder
from unified_planning.engines.convert_problem import Convert_problem
from unified_planning.engines.mdp import MDP
from unified_planning.engines.solvers import mcts as mcts_solver
from unified_planning.engines.utils import create_init_stn, update_stn


# ----------------------------
# Global experiment settings
# ----------------------------
M = 2  # number of keys/doors; edit here
DEADLINE = 30  # None => no deadline; or set an int
SEARCH_TIME = 5
SEARCH_DEPTH = 200
EXPLORATION_CONSTANT = 10
SELECTION_TYPE = "rootInterval"
K_RANDOM_ACTIONS = 10
SEED = 123
TRACE_STEPS = True
RUN_BFS_SOLVER = True
RUN_TP_MCTS = False
TP_MCTS_REWARD_MODE = "deadline_step"


class ActionType(str, Enum):
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    TAKE_KEY = "take_key"
    DROP_KEY = "drop_key"
    OPEN_DOOR = "open_door"
    PRESS_BUTTON = "press_button"


@dataclass(frozen=True)
class State:
    position: int
    doors_open_mask: int
    held_key: Optional[int]
    key_positions: Tuple[Optional[int], ...]
    button_pressed: bool


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    index: Optional[int] = None

    def __str__(self) -> str:
        if self.index is None:
            return f"{self.action_type.value}"
        return f"{self.action_type.value}({self.index})"


class KeyDoorCorridor:
    def __init__(
        self,
        key_positions: Sequence[int],
        door_positions: Sequence[int],
        button_position: Optional[int] = None,
        requires_button: bool = False,
    ) -> None:
        if len(key_positions) != len(door_positions):
            raise ValueError("Number of keys and doors must match.")
        if sorted(door_positions) != list(door_positions):
            raise ValueError("Door positions must be nondecreasing.")
        if requires_button and button_position is None:
            raise ValueError("Button position required when button is enabled.")
        self.key_positions = list(key_positions)
        self.door_positions = list(door_positions)
        self.num_keys = len(key_positions)
        self.button_position = button_position
        self.requires_button = requires_button
        max_positions = [0] + list(key_positions) + list(door_positions)
        if button_position is not None:
            max_positions.append(button_position)
        self.length = max(max_positions)

    def initial_state(self) -> State:
        return State(
            position=0,
            doors_open_mask=0,
            held_key=None,
            key_positions=tuple(self.key_positions),
            button_pressed=False,
        )

    def is_terminal(self, state: State) -> bool:
        all_open_mask = (1 << self.num_keys) - 1
        if state.doors_open_mask != all_open_mask:
            return False
        if self.requires_button:
            return state.button_pressed
        return True

    def _is_door_open(self, state: State, door_index: int) -> bool:
        return bool(state.doors_open_mask & (1 << door_index))

    def available_actions(self, state: State) -> List[Action]:
        actions: List[Action] = []
        if state.position > 0:
            actions.append(Action(ActionType.MOVE_LEFT))
        if state.position < self.length:
            actions.append(Action(ActionType.MOVE_RIGHT))
        if state.held_key is not None:
            actions.append(Action(ActionType.DROP_KEY))

        for i, pos in enumerate(state.key_positions):
            if pos is not None and pos == state.position and state.held_key is None:
                actions.append(Action(ActionType.TAKE_KEY, i))

        for i, door_pos in enumerate(self.door_positions):
            if (
                door_pos == state.position
                and state.held_key == i
                and not self._is_door_open(state, i)
            ):
                actions.append(Action(ActionType.OPEN_DOOR, i))

        if (
            self.requires_button
            and self.button_position == state.position
            and not state.button_pressed
            and state.held_key is None
        ):
            actions.append(Action(ActionType.PRESS_BUTTON))
        return actions

    def step(self, state: State, action: Action) -> State:
        position = state.position
        doors_open_mask = state.doors_open_mask
        held_key = state.held_key
        key_positions = list(state.key_positions)
        button_pressed = state.button_pressed

        if action.action_type == ActionType.MOVE_LEFT:
            if position == 0:
                raise ValueError("Cannot move left at corridor start.")
            position -= 1
        elif action.action_type == ActionType.MOVE_RIGHT:
            if position == self.length:
                raise ValueError("Cannot move right at corridor end.")
            position += 1
        elif action.action_type == ActionType.TAKE_KEY:
            if action.index is None:
                raise ValueError("Missing key index.")
            key_index = action.index
            if held_key is not None:
                raise ValueError("Cannot take key while holding another key.")
            if key_positions[key_index] != position:
                raise ValueError("Key not available at current position.")
            held_key = key_index
            key_positions[key_index] = None
        elif action.action_type == ActionType.DROP_KEY:
            if held_key is None:
                raise ValueError("No key to drop.")
            key_positions[held_key] = position
            held_key = None
        elif action.action_type == ActionType.OPEN_DOOR:
            if action.index is None:
                raise ValueError("Missing door index.")
            door_index = action.index
            if held_key != door_index:
                raise ValueError("Must hold matching key to open door.")
            if self._is_door_open(state, door_index):
                raise ValueError("Door already open.")
            if self.door_positions[door_index] != position:
                raise ValueError("Not at door position.")
            doors_open_mask |= 1 << door_index
            held_key = None
        elif action.action_type == ActionType.PRESS_BUTTON:
            if not self.requires_button:
                raise ValueError("No button available in this environment.")
            if self.button_position != position:
                raise ValueError("Not at button position.")
            if held_key is not None:
                raise ValueError("Must have empty hand to press the button.")
            if button_pressed:
                raise ValueError("Button already pressed.")
            button_pressed = True
        else:
            raise ValueError(f"Unknown action: {action.action_type}")

        return State(
            position=position,
            doors_open_mask=doors_open_mask,
            held_key=held_key,
            key_positions=tuple(key_positions),
            button_pressed=button_pressed,
        )


def _move_along(start: int, end: int) -> List[Action]:
    actions: List[Action] = []
    if end > start:
        actions.extend([Action(ActionType.MOVE_RIGHT)] * (end - start))
    elif end < start:
        actions.extend([Action(ActionType.MOVE_LEFT)] * (start - end))
    return actions


def build_corridor_layout(m: int) -> Tuple[List[int], List[int]]:
    key_positions = list(range(1, m + 1))
    door_positions = [m + 2 * (i + 1) for i in range(m)]
    return key_positions, door_positions


def build_up_key_door_problem(m: int, deadline: Optional[int]) -> up.model.Problem:
    key_positions, door_positions = build_corridor_layout(m)
    corridor_length = door_positions[-1] if door_positions else 0

    problem = up.model.Problem(f"key_door_corridor_m{m}")

    Pos = up.shortcuts.UserType("Pos")
    Key = up.shortcuts.UserType("Key")
    Door = up.shortcuts.UserType("Door")

    positions = [up.model.Object(f"p{i}", Pos) for i in range(corridor_length + 1)]
    keys = [up.model.Object(f"k{i}", Key) for i in range(m)]
    doors = [up.model.Object(f"d{i}", Door) for i in range(m)]
    problem.add_objects(positions + keys + doors)

    at = up.model.Fluent("at", up.shortcuts.BoolType(), p=Pos)
    connected = up.model.Fluent("connected", up.shortcuts.BoolType(), p_from=Pos, p_to=Pos)
    key_at = up.model.Fluent("key_at", up.shortcuts.BoolType(), k=Key, p=Pos)
    holding = up.model.Fluent("holding", up.shortcuts.BoolType(), k=Key)
    hand_empty = up.model.Fluent("hand_empty", up.shortcuts.BoolType())
    door_open = up.model.Fluent("door_open", up.shortcuts.BoolType(), d=Door)
    busy = up.model.Fluent("busy", up.shortcuts.BoolType())

    problem.add_fluent(at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)
    problem.add_fluent(key_at, default_initial_value=False)
    problem.add_fluent(holding, default_initial_value=False)
    problem.add_fluent(hand_empty, default_initial_value=False)
    problem.add_fluent(door_open, default_initial_value=False)
    problem.add_fluent(busy, default_initial_value=False)

    # Initial state
    problem.set_initial_value(at(positions[0]), True)
    problem.set_initial_value(hand_empty, True)
    problem.set_initial_value(busy, False)
    for i, key_pos in enumerate(key_positions):
        problem.set_initial_value(key_at(keys[i], positions[key_pos]), True)
    for i in range(corridor_length):
        problem.set_initial_value(connected(positions[i], positions[i + 1]), True)
        problem.set_initial_value(connected(positions[i + 1], positions[i]), True)

    def add_busy_mutex(action: up.model.DurativeAction) -> None:
        # Enforce strictly sequential execution (no overlap) so that
        # time matches walking steps in the deterministic corridor.
        action.add_precondition(up.model.timing.OverallPreconditionTiming(), busy, False)
        action.add_start_effect(busy, True)
        action.add_effect(busy, False)

    # Move action
    move = up.model.DurativeAction("move", p_from=Pos, p_to=Pos)
    p_from = move.parameter("p_from")
    p_to = move.parameter("p_to")
    move.set_fixed_duration(1)
    move.add_precondition(up.model.timing.StartPreconditionTiming(), at(p_from), True)
    move.add_precondition(up.model.timing.StartPreconditionTiming(), connected(p_from, p_to), True)
    move.add_effect(at(p_from), False)
    move.add_effect(at(p_to), True)
    add_busy_mutex(move)
    problem.add_action(move)

    # Take / drop actions
    take = up.model.DurativeAction("take_key", k=Key, p=Pos)
    k = take.parameter("k")
    p = take.parameter("p")
    take.set_fixed_duration(1)
    take.add_precondition(up.model.timing.StartPreconditionTiming(), at(p), True)
    take.add_precondition(up.model.timing.StartPreconditionTiming(), key_at(k, p), True)
    take.add_precondition(up.model.timing.StartPreconditionTiming(), hand_empty, True)
    take.add_effect(holding(k), True)
    take.add_effect(key_at(k, p), False)
    take.add_effect(hand_empty, False)
    add_busy_mutex(take)
    problem.add_action(take)

    drop = up.model.DurativeAction("drop_key", k=Key, p=Pos)
    k = drop.parameter("k")
    p = drop.parameter("p")
    drop.set_fixed_duration(1)
    drop.add_precondition(up.model.timing.StartPreconditionTiming(), at(p), True)
    drop.add_precondition(up.model.timing.StartPreconditionTiming(), holding(k), True)
    drop.add_effect(holding(k), False)
    drop.add_effect(key_at(k, p), True)
    drop.add_effect(hand_empty, True)
    add_busy_mutex(drop)
    problem.add_action(drop)

    # Open door actions (one per door/key; enforces ordering)
    for i in range(m):
        open_action = up.model.DurativeAction(f"open_door_{i}")
        open_action.set_fixed_duration(1)
        open_action.add_precondition(
            up.model.timing.StartPreconditionTiming(), at(positions[door_positions[i]]), True
        )
        open_action.add_precondition(
            up.model.timing.StartPreconditionTiming(), holding(keys[i]), True
        )
        open_action.add_precondition(
            up.model.timing.StartPreconditionTiming(), door_open(doors[i]), False
        )
        if i > 0:
            open_action.add_precondition(
                up.model.timing.StartPreconditionTiming(), door_open(doors[i - 1]), True
            )
        open_action.add_effect(door_open(doors[i]), True)
        open_action.add_effect(holding(keys[i]), False)
        open_action.add_effect(hand_empty, True)
        add_busy_mutex(open_action)
        problem.add_action(open_action)

    for i in range(m):
        problem.add_goal(door_open(doors[i]))

    if deadline is not None:
        deadline_timing = Timing(delay=deadline, timepoint=Timepoint(TimepointKind.START))
        problem.set_deadline(deadline_timing)

    return problem


def run_tp_mcts_on_key_door(m: int, deadline: Optional[int]) -> Tuple[int, float]:
    problem = build_up_key_door_problem(m, deadline)
    grounder = Grounder()
    grounding_result = grounder._compile(problem)
    ground_problem = grounding_result.problem
    converted_problem = Convert_problem(ground_problem).converted_problem

    mdp = MDP(converted_problem, discount_factor=0.95, reward_mode=TP_MCTS_REWARD_MODE)
    success, total_time = mcts_solver.plan(
        mdp,
        steps=90,
        search_time=SEARCH_TIME,
        search_depth=SEARCH_DEPTH,
        exploration_constant=EXPLORATION_CONSTANT,
        selection_type=SELECTION_TYPE,
        k=K_RANDOM_ACTIONS,
    )
    return success, total_time


def _state_summary(state: up.engines.State) -> Dict[str, object]:
    at_pos = None
    holding_key = None
    doors_open = []
    key_positions = {}
    for pred in state.predicates:
        if not pred.is_fluent_exp():
            continue
        name = pred.fluent().name
        args = [a.object().name for a in pred.args]
        if name == "at":
            at_pos = args[0]
        elif name == "holding":
            holding_key = args[0]
        elif name == "door_open":
            doors_open.append(args[0])
        elif name == "key_at":
            key_positions[args[0]] = args[1]
    doors_open.sort()
    return {
        "at": at_pos,
        "holding": holding_key,
        "doors_open": doors_open,
        "key_positions": key_positions,
    }


def bfs_optimal_plan(env: KeyDoorCorridor, max_nodes: int = 200000) -> Tuple[int, List[Action]]:
    """
    Exponential BFS over the explicit state graph.
    Finds the shortest plan (by action count) and avoids loops via visited set.
    """
    start = env.initial_state()
    if env.is_terminal(start):
        return 0, []

    queue = deque([start])
    parent: Dict[State, Tuple[Optional[State], Optional[Action]]] = {start: (None, None)}
    expanded = 0

    while queue:
        state = queue.popleft()
        expanded += 1
        if expanded > max_nodes:
            break

        for action in env.available_actions(state):
            next_state = env.step(state, action)
            if next_state in parent:
                continue
            parent[next_state] = (state, action)
            if env.is_terminal(next_state):
                # Reconstruct
                plan: List[Action] = []
                cur = next_state
                while True:
                    prev, act = parent[cur]
                    if prev is None or act is None:
                        break
                    plan.append(act)
                    cur = prev
                plan.reverse()
                return len(plan), plan
            queue.append(next_state)

    return float("inf"), []


def run_tp_mcts_with_trace(m: int, deadline: Optional[int]) -> Tuple[int, float]:
    problem = build_up_key_door_problem(m, deadline)
    grounder = Grounder()
    ground_problem = grounder._compile(problem).problem
    converted_problem = Convert_problem(ground_problem).converted_problem

    mdp = MDP(converted_problem, discount_factor=0.95, reward_mode=TP_MCTS_REWARD_MODE)
    stn = create_init_stn(mdp)
    root_state = mdp.initial_state()
    root_node = None
    previous_action_node = None
    step = 0

    while stn.get_current_end_time() <= mdp.deadline():
        mcts = mcts_solver.C_MCTS(
            mdp,
            root_node,
            root_state,
            SEARCH_DEPTH,
            EXPLORATION_CONSTANT,
            stn,
            SELECTION_TYPE,
            K_RANDOM_ACTIONS,
            previous_action_node,
        )
        action = mcts.search(SEARCH_TIME, SELECTION_TYPE)
        if action == -1:
            print("TP-MCTS: no valid action found")
            return 0, float("-inf")

        summary = _state_summary(root_state)
        print(
            f"step={step:02d} t={stn.get_current_end_time():>2} "
            f"at={summary['at']} hold={summary['holding']} "
            f"doors={summary['doors_open']} action={action.name}"
        )

        terminal, root_state, reward = mcts.mdp.step(root_state, action)
        action_node = mcts.root_node.children[action] if SELECTION_TYPE == "rootInterval" else None
        previous_action_node = update_stn(
            stn, action, previous_action_node, type="SetTime", action_node=action_node
        )

        if terminal:
            return 1, stn.get_current_end_time()
        step += 1

    return 0, float("-inf")


def optimal_one_hand_plan(env: KeyDoorCorridor) -> Tuple[int, List[Action]]:
    state = env.initial_state()
    actions: List[Action] = []
    for i in range(env.num_keys):
        if state.held_key is not None and state.held_key != i:
            actions.append(Action(ActionType.DROP_KEY))
            state = env.step(state, actions[-1])

        if state.held_key is None:
            key_pos = state.key_positions[i]
            if key_pos is None:
                raise ValueError("Key already consumed unexpectedly.")
            for move in _move_along(state.position, key_pos):
                actions.append(move)
                state = env.step(state, move)
            take = Action(ActionType.TAKE_KEY, i)
            actions.append(take)
            state = env.step(state, take)

        door_pos = env.door_positions[i]
        for move in _move_along(state.position, door_pos):
            actions.append(move)
            state = env.step(state, move)
        open_action = Action(ActionType.OPEN_DOOR, i)
        actions.append(open_action)
        state = env.step(state, open_action)

    if env.requires_button:
        if env.button_position is None:
            raise ValueError("Button position missing in environment.")
        for move in _move_along(state.position, env.button_position):
            actions.append(move)
            state = env.step(state, move)
        press = Action(ActionType.PRESS_BUTTON)
        actions.append(press)
        state = env.step(state, press)

    if not env.is_terminal(state):
        raise ValueError("Plan did not reach terminal state.")
    return len(actions), actions


def relaxed_multi_key_plan(env: KeyDoorCorridor) -> Tuple[int, List[Action]]:
    # Relaxed planner: ignore the one-hand mutex and carry all keys at once,
    # so it can sweep right and never backtrack. It also ignores any
    # extra postconditions like a checkpoint button.
    actions: List[Action] = []
    position = 0

    for i, key_pos in enumerate(env.key_positions):
        for move in _move_along(position, key_pos):
            actions.append(move)
            position += 1 if move.action_type == ActionType.MOVE_RIGHT else -1
        actions.append(Action(ActionType.TAKE_KEY, i))

    for i, door_pos in enumerate(env.door_positions):
        for move in _move_along(position, door_pos):
            actions.append(move)
            position += 1 if move.action_type == ActionType.MOVE_RIGHT else -1
        actions.append(Action(ActionType.OPEN_DOOR, i))

    return len(actions), actions


def build_branch_a() -> KeyDoorCorridor:
    # Branch A: original M=8 corridor, plus a mandatory button press
    # after the final door. The button adds 1 true step that the relaxed
    # planner ignores, yielding a strict deadline miss.
    m = 8
    key_positions = list(range(1, m + 1))
    door_positions = [m + 2 * (i + 1) for i in range(m)]
    button_position = door_positions[-1]
    return KeyDoorCorridor(
        key_positions,
        door_positions,
        button_position=button_position,
        requires_button=True,
    )


def build_branch_b() -> KeyDoorCorridor:
    # Branch B: fewer keys and closer doors -> lower true time,
    # but still large relaxed estimate due to a longer sweep.
    key_positions = [1, 2, 3, 4, 5]
    door_positions = [16, 20, 24, 28, 32]
    return KeyDoorCorridor(key_positions, door_positions)


def choose_branch_by_relaxation(
    env_a: KeyDoorCorridor, env_b: KeyDoorCorridor, deadline: int
) -> Dict[str, object]:
    t_relaxed_a, _ = relaxed_multi_key_plan(env_a)
    t_relaxed_b, _ = relaxed_multi_key_plan(env_b)
    chosen = "A" if t_relaxed_a <= t_relaxed_b else "B"

    if chosen == "A":
        t_true_chosen, _ = optimal_one_hand_plan(env_a)
    else:
        t_true_chosen, _ = optimal_one_hand_plan(env_b)

    return {
        "chosen": chosen,
        "t_relaxed_a": t_relaxed_a,
        "t_relaxed_b": t_relaxed_b,
        "t_true_chosen": t_true_chosen,
        "success": t_true_chosen <= deadline,
    }


def run_experiment(
    max_m: int = 8,
    save_csv: bool = False,
    csv_path: str = "key_door_corridor_results.csv",
) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    for m in range(1, max_m + 1):
        key_positions = list(range(1, m + 1))
        door_positions = [m + 2 * (i + 1) for i in range(m)]
        env = KeyDoorCorridor(key_positions, door_positions)

        t_true, _ = optimal_one_hand_plan(env)
        t_relaxed, _ = relaxed_multi_key_plan(env)
        bias = t_true - t_relaxed

        rows.append(
            {
                "M": m,
                "T_true": t_true,
                "T_relaxed": t_relaxed,
                "Bias": bias,
            }
        )

    if save_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["M", "T_true", "T_relaxed", "Bias"])
            writer.writeheader()
            writer.writerows(rows)

    return rows


def _assertions(rows: List[Dict[str, int]]) -> None:
    if rows:
        if rows[0]["M"] == 1 and rows[0]["Bias"] != 0:
            raise AssertionError("Expected bias 0 for M=1.")
    for i in range(1, len(rows)):
        if rows[i]["Bias"] <= rows[i - 1]["Bias"]:
            raise AssertionError("Expected bias to increase with M.")


def run_deadline_demo() -> None:
    deadline = 194
    env_a = build_branch_a()
    env_b = build_branch_b()

    t_true_a, _ = optimal_one_hand_plan(env_a)
    t_relaxed_a, _ = relaxed_multi_key_plan(env_a)
    t_true_b, _ = optimal_one_hand_plan(env_b)
    t_relaxed_b, _ = relaxed_multi_key_plan(env_b)

    choice = choose_branch_by_relaxation(env_a, env_b, deadline)

    # Sanity checks for the counterexample
    if t_true_b > deadline:
        raise AssertionError("Branch B must meet the deadline.")
    if t_relaxed_b <= t_relaxed_a:
        raise AssertionError("Relaxation must prefer branch A.")
    if choice["chosen"] != "A":
        raise AssertionError("Relaxation should choose branch A.")
    if t_true_a <= deadline:
        raise AssertionError("Branch A should miss the deadline.")
    if t_true_b > deadline:
        raise AssertionError("Branch B should succeed under the deadline.")

    print("Deadline counterexample (deterministic, two-branch):")
    print(
        f"D={deadline} | "
        f"T_true_A={t_true_a}, T_relaxed_A={t_relaxed_a} | "
        f"T_true_B={t_true_b}, T_relaxed_B={t_relaxed_b}"
    )
    print(f"Relaxation chooses branch {choice['chosen']}.")
    print(f"Chosen branch meets deadline? {choice['success']}")
    print(
        "Branch B would meet the deadline, but relaxation prefers A due to the "
        "optimistic estimate. The extra button press in A is required in the true "
        "model but ignored by the relaxation."
    )


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    key_positions, door_positions = build_corridor_layout(M)
    env = KeyDoorCorridor(key_positions, door_positions)
    t_true, _ = optimal_one_hand_plan(env)
    t_relaxed, _ = relaxed_multi_key_plan(env)

    deadline = DEADLINE if DEADLINE is not None else t_true + 5
    if RUN_TP_MCTS:
        if TRACE_STEPS:
            success, total_time = run_tp_mcts_with_trace(M, deadline)
        else:
            success, total_time = run_tp_mcts_on_key_door(M, deadline)

        print("KeyDoorCorridor with TP-MCTS (PTRPG heuristic)")
        print(f"M = {M}")
        print(f"Deadline = {deadline}")
        print(f"Optimal (one-hand) time = {t_true}")
        print(f"Relaxed (multi-key) time = {t_relaxed}")
        print(f"TP-MCTS success = {success}, time = {total_time}")

    if RUN_BFS_SOLVER:
        bfs_time, bfs_plan = bfs_optimal_plan(env)
        print(f"BFS optimal time = {bfs_time}, plan length = {len(bfs_plan)}")


if __name__ == "__main__":
    main()
