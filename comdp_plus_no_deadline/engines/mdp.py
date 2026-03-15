import unified_planning as up
from unified_planning.engines.mdp import MDP as BaseMDP
from unified_planning.engines.mdp import combinationMDP as BaseCombinationMDP


class MDP(BaseMDP):
    """No-deadline-aware MDP wrapper."""

    def __init__(self, problem, discount_factor: float, reward_mode: str = "terminal"):
        super().__init__(problem, discount_factor, reward_mode=reward_mode)

    def terminal_reward(self, terminal: bool, state):
        if not terminal:
            return 0
        if self.reward_mode == "terminal":
            return 1
        if self.reward_mode == "deadline":
            if self.deadline() is None:
                return 1
            if hasattr(state, "current_time"):
                return 1 if state.current_time <= self.deadline() else 0
            return 1
        raise ValueError(f"Unknown reward_mode: {self.reward_mode}")


class combinationMDP(BaseCombinationMDP):
    """No-deadline-aware combination MDP wrapper."""

    def __init__(self, problem, discount_factor: float, reward_mode: str = "terminal"):
        super().__init__(problem, discount_factor)
        self._reward_mode = reward_mode

    def terminal_reward(self, terminal: bool, state):
        if not terminal:
            return 0
        if self.reward_mode == "terminal":
            return 1
        if self.reward_mode == "deadline":
            if self.deadline() is None:
                return 1
            if hasattr(state, "current_time"):
                return 1 if state.current_time <= self.deadline() else 0
            return 1
        raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

