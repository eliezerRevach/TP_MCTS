import numpy as np
from unified_planning.shortcuts import *
import unified_planning as up
import math
import time
import random
from unified_planning.engines.utils import (
    create_init_stn,
    update_stn,
)
from unified_planning.engines.linked_list import LinkedListNode


class Base_MCTS:
    def __init__(self, mdp: "up.engines.MDP", search_depth: int,
                 exploration_constant: float, k: int, heuristic_samples: int = 50,
                 tree_seed: int = None, heuristic_mode: str = "crn_per_tree"):
        self._mdp = mdp
        self._search_depth = search_depth
        self._exploration_constant = exploration_constant
        self._root_node = None
        self._k = k
        self._heuristic_samples = max(1, int(heuristic_samples))
        self._heuristic_mode = heuristic_mode
        valid_modes = {"legacy", "crn_fixed", "crn_per_tree"}
        if self._heuristic_mode not in valid_modes:
            raise ValueError(f"Unknown heuristic_mode: {self._heuristic_mode}. Expected one of {valid_modes}.")

        self._tree_seed = random.getrandbits(32) if tree_seed is None else int(tree_seed)
        self._scenario_seeds = []
        if self._heuristic_mode == "crn_fixed":
            # Legacy CRN experiment: same scenario family for every tree.
            self._scenario_seeds = [1009 * (i + 1) + 17 for i in range(self._heuristic_samples)]
        elif self._heuristic_mode == "crn_per_tree":
            # Improved CRN: shared within tree, refreshed across trees.
            local_rng = random.Random(self._tree_seed)
            self._scenario_seeds = [local_rng.getrandbits(32) for _ in range(self._heuristic_samples)]
        self._heuristic_avg_cache = {}
        self._heuristic_scenario_cache = {}

    @property
    def mdp(self):
        return self._mdp

    @property
    def root_node(self):
        return self._root_node

    @property
    def k(self):
        return self._k

    @property
    def heuristic_samples(self):
        return self._heuristic_samples

    @property
    def heuristic_mode(self):
        return self._heuristic_mode

    def root_state(self):
        return self.root_node.state

    @property
    def search_depth(self):
        return self._search_depth

    @property
    def exploration_constant(self):
        return self._exploration_constant

    def set_root_node(self, root_node):
        self._root_node = root_node

    def _seeded_trpg_value(self, trpg_factory, seed: int):
        """
        Evaluate TRPG with a deterministic scenario seed (CRN).
        """
        np_state = np.random.get_state()
        py_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)
        try:
            return trpg_factory()
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

    @staticmethod
    def _lb_signature(lower_bounds):
        if not lower_bounds:
            return ()
        return tuple(sorted((action.name, value) for action, value in lower_bounds.items()))

    @staticmethod
    def _state_signature(state, current_time):
        predicates = tuple(sorted(str(pred) for pred in state.predicates))
        return predicates, current_time

    def trpg_value(self, mdp, state, current_time=0, lower_bounds=None):
        """
        Heuristic mode switch:
        - legacy: original behavior (single stochastic PTRPG call, no CRN averaging).
        - crn_fixed: average PTRPG over fixed global CRN scenarios.
        - crn_per_tree: average PTRPG over CRN scenarios refreshed per tree.
        """
        if self.heuristic_mode == "legacy":
            h = up.engines.heuristics.TRPG(mdp, state, current_time)
            return h.get_heuristic(lower_bounds)

        state_sig = self._state_signature(state, current_time)
        lb_sig = self._lb_signature(lower_bounds)
        avg_key = (state_sig, lb_sig)
        cached_avg = self._heuristic_avg_cache.get(avg_key)
        if cached_avg is not None:
            return cached_avg

        total = 0.0
        for scenario_id, seed in enumerate(self._scenario_seeds):
            scenario_key = (scenario_id, state_sig, lb_sig)
            value = self._heuristic_scenario_cache.get(scenario_key)
            if value is None:
                def run():
                    h = up.engines.heuristics.TRPG(mdp, state, current_time)
                    return h.get_heuristic(lower_bounds)
                value = self._seeded_trpg_value(run, seed)
                self._heuristic_scenario_cache[scenario_key] = value
            total += value

        avg_value = total / self.heuristic_samples
        self._heuristic_avg_cache[avg_key] = avg_value
        return avg_value

    def default_policy(self, state: "up.engines.State"):
        """ Choose a random action. Heustics can be used here to improve simulations. """
        return random.choice(self.mdp.legal_actions(state))

    def uct(self, snode: "up.engines.Snode", explore_constant: float):
        anodes = snode.children
        best_ub = -float('inf')
        best_action = -1
        possible_actions = snode.possible_actions
        for action in possible_actions:
            if anodes[action].count == 0:
                return action

            ub = (anodes[action].value / anodes[action].count) + (
                    explore_constant * math.sqrt(math.log(snode.count) / anodes[action].count))
            # ub = anodes[action].value + (
            #         explore_constant * math.sqrt(math.log(snode.count + 1) / anodes[action].count))
            if ub > best_ub:
                best_ub = ub
                best_action = action

        assert best_action != -1
        return best_action

    def best_action(self, root_node: "up.engines.SNode"):
        """

        :param root_node: the root node of the MCTS tree
        :return: returns the best action for the `root_node`
        """
        anodes = root_node.children
        aStart_value = float("-inf")
        aStar = -1

        for action in root_node.possible_actions:
            if anodes[action].count > 0 and anodes[action].value > aStart_value:
                aStart_value = anodes[action].value
                aStar = action

        if aStar == -1:
            print(4)

        return aStar

    def search(self, timeout=1, selection_type='avg'):
        """
        Execute the MCTS algorithm from the initial state given, with timeout in seconds
        """
        start_time = time.time()
        current_time = time.time()
        i = 0
        selection = self.selection if selection_type == 'avg' else (self.selection_root_interval if selection_type == 'rootInterval' else self.selection_max)
        while current_time < start_time + timeout:
            selection(self.root_node)
            current_time = time.time()
            i += 1
        # print(f'i = {i}')
        return self.best_action(self.root_node)

    def selection(self, snode: "up.engines.Snode"):
        raise NotImplementedError

    def selection_max(self, snode: "up.engines.Snode"):
        raise NotImplementedError

    def selection_root_interval(self, snode: "up.engines.Snode"):
        raise NotImplementedError
    def selection_root_interval_max(self, snode: "up.engines.Snode"):
        raise NotImplementedError

    def simulate(self, state, depth):
        raise NotImplementedError


class MCTS(Base_MCTS):
    """
    Original MCTS solver implementation.
    """
    def __init__(self, mdp: "up.engines.MDP", split_mdp: "up.engines.MDP", root_node: "up.engines.SNode",
                 root_state: "up.engines.state.State", search_depth: int,
                 exploration_constant: float, selection_type, k: int, heuristic_samples: int = 50,
                 tree_seed: int = None, heuristic_mode: str = "crn_per_tree"):
        super().__init__(mdp, search_depth, exploration_constant, k, heuristic_samples, tree_seed, heuristic_mode)
        self.split_mdp = split_mdp
        create_snode = self.create_Snode_max if selection_type == 'max' else self.create_Snode
        snode, _ = create_snode(root_state, 0)
        self.set_root_node(root_node if root_node is not None else snode)

    def create_Snode(self, state: "up.engines.State", depth: int,
                     parent: "up.engines.ANode" = None):
        """ Create a new Snode for the state `state` with parent `parent`"""
        return up.engines.SNode(state, depth, self.mdp.legal_actions(state), parent), None

    def create_Snode_max(self, state: "up.engines.State", depth: int,
                         parent: "up.engines.C_ANode" = None):
        """
        Create a new Snode for the state `state` with parent `parent`
        In this approach k children of snode are evaluated and the initiate value of snode is set to maximum value.

        """
        snode = up.engines.SNode(state, depth, self.mdp.legal_actions(state), parent)
        best = -math.inf

        actions_idx = list(range(len(snode.children)))
        if self.k < len(snode.children):
            # samples k children
            actions_idx = random.sample(range(0, len(snode.children)), self.k)

        for action_idx in actions_idx:
            # perform each action and evaluate the next state with the heuristic function
            action = list(snode.children.keys())[action_idx]
            terminal, next_state, reward = self.mdp.step(snode.state, action)
            reward += self.mdp.discount_factor * self.heuristic(next_state)
            snode.children[action].update(reward)
            if reward > best:
                best = reward
        if best == -math.inf:
            best = self.heuristic(state)

        snode.update(best)
        return snode, best

    def heuristic(self, state: "up.engines.State"):
        current_time = 0
        if isinstance(state, up.engines.CombinationState):
            current_time = state.current_time
        return self.trpg_value(self.split_mdp, state, current_time)

    def selection(self, snode: "up.engines.Snode"):
        """
        Traverse the tree until reaching a leaf node.
        """
        if len(snode.possible_actions) == 0 or snode.state.current_time > self.mdp.deadline():
            # Stop when there are no possible actions to take so the plan remains consistent
            return -100

        if snode.depth > self.search_depth:
            return self.heuristic(snode.state)

        explore_constant = self.exploration_constant

        # Choose a consistent action
        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        anode = snode.children[action]
        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                reward += self.mdp.discount_factor * self.selection(snodes[next_state])

            else: # leaf
                next_snode, _ = self.create_Snode(next_state, snode.depth + 1, anode)
                reward += self.mdp.discount_factor * self.heuristic(next_state)
                anode.add_child(next_snode)

        snode.update(reward)
        anode.update(reward)

        return reward

    def selection_max(self, snode: "up.engines.Snode"):
        """
        Traverse the tree until reaching a leaf node.
        Selection with max logic -
        average between states and maximum between possible actions
        """
        if len(snode.possible_actions) == 0 or snode.state.current_time > self.mdp.deadline():
            # Stop when there are no possible actions to take so the plan remains consistent
            return -100

        if snode.depth > self.search_depth:
            # Stop if the search depth is reached
            return self.heuristic(snode.state)
        explore_constant = self.exploration_constant

        # Choose a consistent action
        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        anode = snode.children[action]
        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                reward += self.mdp.discount_factor * self.selection_max(snodes[next_state])

            else: # leaf
                next_snode, snode_reward = self.create_Snode_max(next_state, snode.depth + 1, anode)
                reward += snode_reward
                anode.add_child(next_snode)

        anode.update(reward)
        max_v = snode.max_update()

        return max_v

    def simulate(self, state, depth):
        """ Simulate until a terminal state """
        cumulative_reward = 0.0
        terminal = False
        deadline = self.mdp.deadline()
        while not terminal and depth < self.search_depth and len(self.mdp.legal_actions(state)) > 0:
            # Choose an action to execute
            action = self.default_policy(state)

            # Execute the action
            (terminal, next_state, reward) = self.mdp.step(state, action)

            # Discount the reward
            cumulative_reward += pow(self.mdp.discount_factor, depth) * reward
            depth += 1

            state = next_state

        return cumulative_reward


class C_MCTS(Base_MCTS):
    """
    TP MCTS solver implementation.
    Contains STNs in each node
    """
    def __init__(self, mdp, root_node: "up.engines.C_SNode", root_state: "up.engines.state.State", search_depth: int,
                 exploration_constant: float, stn: "up.plans.stn.STNPlan", selection_type, k: int,
                 previous_chosen_action_node: "up.plans.stn.STNPlanNode" = None, heuristic_samples: int = 50,
                 tree_seed: int = None, heuristic_mode: str = "crn_per_tree"):
        super().__init__(mdp, search_depth, exploration_constant, k, heuristic_samples, tree_seed, heuristic_mode)
        self._previous_chosen_action_node = previous_chosen_action_node

        create_snode = self.create_Snode_max if selection_type == 'max' else (self.create_Snode_root_interval if selection_type == 'rootInterval' else self.create_Snode)
        snode, _ = create_snode(root_state, 0, stn,
                                previous_chosen_action_node=previous_chosen_action_node)
        self.set_root_node(root_node if root_node is not None else snode)
        self._stn = stn

    @property
    def previous_chosen_action_node(self):
        return self._previous_chosen_action_node

    @property
    def stn(self):
        return self._stn

    def create_Snode(self, state: "up.engines.State", depth: int, stn: "up.plans.stn.STNPlan",
                     parent: "up.engines.C_ANode" = None,
                     previous_chosen_action_node: "up.plans.stn.STNPlanNode" = None, isInterval=False):
        """ Create a new Snode for the state `state` with parent `parent`"""
        return up.engines.C_SNode(state, depth, self.mdp.legal_actions(state), stn, parent,
                                  previous_chosen_action_node, isInterval), None

    def create_Snode_root_interval(self, state: "up.engines.State", depth: int, stn: "up.plans.stn.STNPlan",
                     parent: "up.engines.C_ANode" = None,
                     previous_chosen_action_node: "up.plans.stn.STNPlanNode" = None, isInterval=True):
        """ Create a new Snode for the state `state` with parent `parent`
        RootInterval approach """
        return up.engines.C_SNode(state, depth, self.mdp.legal_actions(state), stn, parent,
                                  previous_chosen_action_node, isInterval), None

    def create_Snode_max(self, state: "up.engines.State", depth: int, stn: "up.plans.stn.STNPlan",
                         parent: "up.engines.C_ANode" = None,
                         previous_chosen_action_node: "up.plans.stn.STNPlanNode" = None):
        """ Create a new Snode for the state `state` with parent `parent`
         In this approach k children of snode are evaluated and the initiate value of snode is set to maximum value."""
        snode = up.engines.C_SNode(state, depth, self.mdp.legal_actions(state), stn, parent,
                                   previous_chosen_action_node)
        best = -math.inf

        actions_idx = list(range(len(snode.children)))
        if self.k < len(snode.children):
            actions_idx = random.sample(range(0, len(snode.children)), self.k)

        for action_idx in actions_idx:
            action = list(snode.children.keys())[action_idx]
            terminal, next_state, reward = self.mdp.step(snode.state, action)
            reward += self.mdp.discount_factor * self.heuristic_init(next_state, snode.children[action].stn)
            snode.children[action].update(reward)
            if reward > best:
                best = reward
        if best == -math.inf:
            best = self.heuristic(snode)

        snode.update(best)
        return snode, best


    def selection(self, snode: "up.engines.C_Snode"):
        """
                Traverse the tree until reaching a leaf node.
         """
        if len(snode.possible_actions) == 0:
            # Stop when there are no possible actions to take so the plan remains consistent
            return -100

        if snode.depth > self.search_depth:
            # Stop if the search depth is reached
            # return 0
            return self.heuristic(snode)

        explore_constant = self.exploration_constant

        # Choose a consistent action
        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        anode = snode.children[action]
        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                reward += self.mdp.discount_factor * self.selection(snodes[next_state])

            else: # leaf
                next_snode, _ = self.create_Snode(next_state, snode.depth + 1, anode.stn, anode)
                reward += self.mdp.discount_factor * self.heuristic(next_snode)
                anode.add_child(next_snode)
                next_snode.update(reward)

        snode.update(reward)
        anode.update(reward)

        return reward

    def selection_max(self, snode: "up.engines.C_Snode"):
        """
        Traverse the tree until reaching a leaf node.
        Selection with max logic -
        average between states and maximum between possible actions
        """
        if len(snode.possible_actions) == 0:
            # Stop when there are no possible actions to take so the plan remains consistent
            return -100

        if snode.depth > self.search_depth:
            # Stop if the search depth is reached
            return self.heuristic(snode)
        explore_constant = self.exploration_constant

        # Choose a consistent action
        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        anode = snode.children[action]
        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                reward += self.mdp.discount_factor * self.selection_max(snodes[next_state])

            else: #leaf
                next_snode, snode_reward = self.create_Snode_max(next_state, snode.depth + 1, anode.stn, anode)
                reward += snode_reward
                anode.add_child(next_snode)

        anode.update(reward)
        max_v = snode.max_update()

        return max_v

    def selection_root_interval(self, snode: "up.engines.C_Snode", root_STNnode: "up.plans.stn.STNPlanNode" = None):
        """
        Traverse the tree until reaching a leaf node.
        Selection with root interval logic -
        set the value per root action legal interval.
        The value is propagated and updated according the legal interval
        """
        if len(snode.possible_actions) == 0:
            # Stop when there are no possible actions to take so the plan remains consistent
            if root_STNnode is None:
                return 0
            return 0, *snode.parent.stn.get_legal_interval(root_STNnode)

        if snode.depth > self.search_depth:
            # Stop if the search depth is reached
            return self.heuristic(snode), *snode.parent.stn.get_legal_interval(root_STNnode)

        explore_constant = self.exploration_constant
        # Choose a consistent action
        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)

        anode = snode.children[action]
        if root_STNnode is None:
            root_STNnode = anode.STNNode

        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                next_reward, lower, upper = self.selection_root_interval(snodes[next_state], root_STNnode)
                reward += next_reward * self.mdp.discount_factor

            else: # leaf
                next_snode, _ = self.create_Snode_root_interval(next_state, snode.depth + 1, anode.stn, anode)
                # legal interval of the root action node in the leaf node
                lower, upper = anode.stn.get_legal_interval(root_STNnode)
                reward += self.heuristic(next_snode) * self.mdp.discount_factor
                anode.add_child(next_snode)
                next_snode.update(reward, lower, upper)

        else:
            # when the state is terminal set the lower and upper according to anode root
            lower, upper = anode.stn.get_legal_interval(root_STNnode)

        anode.update(reward, lower, upper)
        snode.update(reward, lower, upper)
        return reward, lower, upper


    def selection_root_interval_max(self, snode: "up.engines.C_Snode", root_STNnode: "up.plans.stn.STNPlanNode" = None):
        if len(snode.possible_actions) == 0:
            if root_STNnode is None:
                return -100
            # Stop when there are no possible actions to take so the plan remains consistent
            return LinkedListNode(*snode.parent.stn.get_legal_interval(root_STNnode), - 100)

        if snode.depth > self.search_depth:
            # Stop if the search depth is reached
            return LinkedListNode(*snode.parent.stn.get_legal_interval(root_STNnode), self.heuristic(snode))
        explore_constant = self.exploration_constant
        # Choose a consistent action
        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        anode = snode.children[action]
        if root_STNnode is None:
            root_STNnode = anode.STNNode

        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                backup_node = self.selection_root_interval_max(snodes[next_state], root_STNnode)
                backup_node.update_df_reward(self.mdp.discount_factor, reward)

            else:
                next_snode, backup_node = self.create_Snode_root_interval_max(next_state, snode.depth + 1, anode.stn, anode, root_STNnode=root_STNnode)
                backup_node.update_df_reward(self.mdp.discount_factor, reward) #TODO: should it be with discount reward
                anode.add_child(next_snode)

        else:
            backup_node = LinkedListNode(*anode.stn.get_legal_interval(root_STNnode), reward)

        anode.update(None, backup_node)
        backup_node = snode.max_update(backup_node)
        return backup_node

    def heuristic(self, snode: "up.engines.C_SNode"):
        current_time = 0
        lower_bounds = None
        if snode.parent:
            current_time = snode.parent.stn.get_current_end_time()
            lower_bounds = snode.parent.stn.get_lower_bound_potential_end_action()
        return self.trpg_value(self.mdp, snode.state, current_time, lower_bounds)

    def heuristic_init(self, state, stn):
        current_time = stn.get_current_end_time()
        return self.trpg_value(self.mdp, state, current_time)


def plan(mdp: "up.engines.MDP", steps: int, search_time: int, search_depth: int, exploration_constant: float,
         selection_type='avg', k=10, heuristic_samples: int = 50, heuristic_mode: str = "crn_per_tree"):
    stn = create_init_stn(mdp)
    root_state = mdp.initial_state()

    reuse = False
    history = []
    previous_action_node = None
    step = 0
    root_node = None

    while stn.get_current_end_time() <= mdp.deadline():
        print(f"started step {step}")
        tree_seed = random.getrandbits(32) if heuristic_mode == "crn_per_tree" else None
        mcts = C_MCTS(mdp, root_node, root_state, search_depth, exploration_constant, stn, selection_type, k,
                      previous_action_node, heuristic_samples=heuristic_samples, tree_seed=tree_seed,
                      heuristic_mode=heuristic_mode)
        action = mcts.search(search_time, selection_type)

        if action == -1:
            print("A valid plan is not found")
            return 0, -math.inf

        print(f"Current state is {root_state}")
        print(f"The chosen action is {action.name}")

        terminal, root_state, reward = mcts.mdp.step(root_state, action)

        if reuse and root_state in mcts.root_node.children[action].children:
            root_node = mcts.root_node.children[action].children[root_state]
            root_node.set_depth(0)

        # update STN to include the action
        action_node = mcts.root_node.children[action] if selection_type == 'rootInterval' else None

        previous_action_node = update_stn(stn, action, previous_action_node, type='SetTime', action_node=action_node)

        assert stn.is_consistent()

        print(f"The time of the plan so far: {stn.get_current_end_time()}")
        history.append(previous_action_node)

        if terminal:
            print(f"Current state is {root_state}")
            print(f"The amount of time the plan took: {stn.get_current_end_time()}")
            return 1, stn.get_current_end_time()

        step += 1

    print("A valid plan is not found")
    return 0, -math.inf


def combination_plan(mdp: "up.engines.MDP", split_mdp: "up.engines.MDP", steps: int, search_time: int,
                     search_depth: int, exploration_constant: float,
                     selection_type='avg', k=10, heuristic_samples: int = 50, heuristic_mode: str = "crn_per_tree"):
    root_state = mdp.initial_state()
    history = []
    step = 0
    root_node = None

    while root_state.current_time < mdp.deadline():
        print(f"started step {step}")

        tree_seed = random.getrandbits(32) if heuristic_mode == "crn_per_tree" else None
        mcts = MCTS(mdp, split_mdp, root_node, root_state, search_depth, exploration_constant, selection_type, k,
                    heuristic_samples=heuristic_samples, tree_seed=tree_seed, heuristic_mode=heuristic_mode)
        action = mcts.search(search_time, selection_type)

        print(f"Current state is {root_state}")
        print(f"The chosen action is {action.name}")

        terminal, root_state, reward = mcts.mdp.step(root_state, action)

        history.append(action)
        print(f'current time = {root_state.current_time}')

        if terminal and root_state.current_time <= mdp.deadline():
            print(f"Current state is {root_state}")
            print(f"The amount of time the plan took: {root_state.current_time}")
            return 1, root_state.current_time

        step += 1

    return 0, -math.inf
