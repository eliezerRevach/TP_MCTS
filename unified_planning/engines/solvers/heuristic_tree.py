"""
Heuristic Lookahead Tree Solver
================================
A bounded-depth lookahead tree that uses the atom_backtrack_cached heuristic
(or any other temporal PTRPG strategy) to score leaf nodes, with branch-and-bound
pruning enabled by the admissibility of the heuristic.

Architecture (three layers):
  Online loop  ->  Tree search  ->  Heuristic evaluation
  - Online loop owns the real MDP state, real STN, and calls _tree_search at
    each decision point.
  - _tree_search builds a simulated lookahead tree.  At each tree node it
    expands every feasible action, takes the *most probable* successor state
    (deterministic-style), and recurses.  Leaves are evaluated by the heuristic.
  - The heuristic (TemporalProbabilisticRPGHeuristic) is attached once to the
    MDP and its internal _query_cache (keyed by state/depth/time) deduplicates
    calls automatically across the whole tree and across online steps.
  - After the tree returns the best action the online loop calls mdp.step
    (real stochastic sampling), updates the STN, and repeats.

Pruning:
  Because atom_backtrack_cached is admissible (optimistic upper bound), at any
  node whose heuristic value is <= the best value found so far (alpha) no child
  can improve on alpha, so the whole subtree is pruned.
"""

import math

import unified_planning as up
from unified_planning.engines.solvers.greedy_parallel import _heuristic_value
from unified_planning.engines.utils import create_init_stn, update_stn


def _most_likely_outcome(transitions):
    """Return the successor state with the highest probability."""
    best_state, best_prob = None, -1.0
    for state, prob in transitions:
        if prob > best_prob:
            best_prob = prob
            best_state = state
    return best_state, best_prob


def _tree_search(
    mdp,
    state,
    stn,
    previous_action_node,
    depth,
    max_depth,
    alpha,
    heuristic_name,
    temporal_heuristic_depth,
    temporal_heuristic_strategy,
    heuristic_weight,
):
    """
    Recursive bounded-depth lookahead tree with branch-and-bound pruning.

    At each node:
    1. If depth limit reached or state is terminal -> evaluate with heuristic.
    2. Otherwise compute the admissible heuristic of the current state.
       If it is <= alpha, prune (no action here can beat the best already found).
    3. For each legal action:
       a. Clone STN and check temporal feasibility.
       b. Take the most probable successor state.
       c. Recurse on that child with updated alpha.
       d. Score = immediate_reward + heuristic_weight * child_value - 0.001 * time.
    4. Return (best_score, best_root_action).

    Parameters
    ----------
    alpha : float
        Best value found so far at this level; used for pruning.

    Returns
    -------
    (best_value, best_action)
        best_action is None at leaves.
    """
    current_time = stn.get_current_end_time()

    # --- Base cases ---
    if mdp.is_terminal(state):
        terminal_reward = mdp.terminal_reward(True, state)
        return terminal_reward, None

    if depth == max_depth:
        leaf_h = _heuristic_value(
            mdp=mdp,
            state=state,
            current_time=current_time,
            heuristic_name=heuristic_name,
            temporal_heuristic_depth=temporal_heuristic_depth,
            temporal_heuristic_strategy=temporal_heuristic_strategy,
        )
        return leaf_h, None

    # --- Admissibility-based pruning at internal nodes ---
    # The heuristic is an optimistic upper bound on value from this state.
    # If even the best possible value from here cannot beat alpha, prune.
    node_h = _heuristic_value(
        mdp=mdp,
        state=state,
        current_time=current_time,
        heuristic_name=heuristic_name,
        temporal_heuristic_depth=temporal_heuristic_depth,
        temporal_heuristic_strategy=temporal_heuristic_strategy,
    )
    # Upper bound on any score reachable from this state
    node_upper = 1.0 + heuristic_weight * node_h  # reward <= 1
    if node_upper <= alpha:
        return node_h, None  # pruned; return heuristic as pessimistic estimate

    legal_actions = mdp.legal_actions(state)
    if not legal_actions:
        return node_h, None

    best_score = -math.inf
    best_action = None

    for action in legal_actions:
        # --- STN feasibility check ---
        candidate_stn = stn.clone()
        try:
            candidate_prev = update_stn(candidate_stn, action, previous_action_node, type="SetTime")
        except Exception:
            continue

        if not candidate_stn.is_consistent():
            continue
        if candidate_stn.get_current_end_time() > mdp.deadline():
            continue

        action_time = candidate_stn.get_current_end_time()

        # --- Transition: pick most probable outcome ---
        transitions = mdp.transition_function(state, action)
        if not transitions:
            continue

        next_state, _ = _most_likely_outcome(transitions)
        terminal = mdp.is_terminal(next_state)

        step_penalty = getattr(mdp, "step_penalty", -0.01)
        immediate_reward = mdp.terminal_reward(terminal, next_state) + step_penalty

        if terminal:
            child_value = mdp.terminal_reward(True, next_state)
        else:
            child_value, _ = _tree_search(
                mdp=mdp,
                state=next_state,
                stn=candidate_stn,
                previous_action_node=candidate_prev,
                depth=depth + 1,
                max_depth=max_depth,
                alpha=best_score,
                heuristic_name=heuristic_name,
                temporal_heuristic_depth=temporal_heuristic_depth,
                temporal_heuristic_strategy=temporal_heuristic_strategy,
                heuristic_weight=heuristic_weight,
            )

        score = immediate_reward + heuristic_weight * child_value - 0.001 * action_time
        if score > best_score:
            best_score = score
            best_action = action

    if best_action is None:
        return node_h, None

    return best_score, best_action


def plan(
    mdp: "up.engines.MDP",
    steps: int,
    search_time: int,
    search_depth: int,
    exploration_constant: float,
    selection_type: str = "avg",
    k: int = 10,
    heuristic_name: str = "temporal_probabilistic_rpg",
    temporal_heuristic_depth: int = 20,
    temporal_heuristic_strategy: str = "atom_backtrack_cached",
    tree_depth: int = 3,
):
    """
    Heuristic lookahead tree dispatcher.

    At each online decision step, builds a bounded-depth lookahead tree using
    the atom_backtrack_cached heuristic (or any PTRPG strategy) to evaluate
    leaves.  Only the most probable successor state is expanded at each tree
    node, keeping the tree compact.  Branch-and-bound pruning uses the
    admissible heuristic at internal nodes.

    The heuristic cache (TemporalProbabilisticRPGHeuristic._query_cache) is
    shared across the whole tree and across online steps via the MDP object,
    so repeated state evaluations at the same depth cost nothing.

    Parameters
    ----------
    tree_depth : int
        Lookahead depth of the internal tree (default 3).  Separate from
        temporal_heuristic_depth which controls the heuristic's own planning
        graph depth.
    """
    # Unused MCTS-specific arguments
    del search_time, search_depth, exploration_constant, selection_type, k

    stn = create_init_stn(mdp)
    root_state = mdp.initial_state()
    previous_action_node = None
    max_parallel_set_size = 32
    heuristic_weight = 0.2
    step = 0

    while stn.get_current_end_time() <= mdp.deadline() and step < steps:
        print(f"started step {step}")
        decision_time = stn.get_current_end_time()
        chosen_in_set = 0

        while chosen_in_set < max_parallel_set_size:
            legal_actions = mdp.legal_actions(root_state)
            if not legal_actions:
                break

            # Run lookahead tree from current state to select best action
            _, best_action = _tree_search(
                mdp=mdp,
                state=root_state,
                stn=stn,
                previous_action_node=previous_action_node,
                depth=0,
                max_depth=tree_depth,
                alpha=-math.inf,
                heuristic_name=heuristic_name,
                temporal_heuristic_depth=temporal_heuristic_depth,
                temporal_heuristic_strategy=temporal_heuristic_strategy,
                heuristic_weight=heuristic_weight,
            )

            if best_action is None:
                break

            # Recompute the winning STN for the chosen action (tree used clones)
            next_stn = stn.clone()
            try:
                next_prev = update_stn(next_stn, best_action, previous_action_node, type="SetTime")
            except Exception:
                break
            if not next_stn.is_consistent() or next_stn.get_current_end_time() > mdp.deadline():
                break

            print(f"Current state is {root_state}")
            print(f"The chosen action is {best_action.name}")

            # Real environment step (stochastic sampling)
            terminal, next_state, _ = mdp.step(root_state, best_action)

            root_state = next_state
            stn = next_stn
            previous_action_node = next_prev
            chosen_in_set += 1

            print(f"The time of the plan so far: {stn.get_current_end_time()}")

            if terminal:
                print(f"Current state is {root_state}")
                print(f"The amount of time the plan took: {stn.get_current_end_time()}")
                return 1, stn.get_current_end_time()

            # Stay in the same parallel time slice
            if stn.get_current_end_time() > decision_time:
                break

        if chosen_in_set == 0:
            print("A valid plan is not found")
            return 0, -math.inf
        step += 1

    print("A valid plan is not found")
    return 0, -math.inf
