from importlib import import_module

__all__ = [
    "MDP",
    "combinationMDP",
    "regular_greedy_plan",
    "combination_greedy_plan",
    "ProbabilisticOptimisticRPGHeuristic",
    "TemporalProbabilisticRPGHeuristic",
]


def __getattr__(name):
    if name in {"MDP", "combinationMDP"}:
        module = import_module(".mdp", __name__)
    elif name in {"regular_greedy_plan", "combination_greedy_plan"}:
        module = import_module(".greedy_solver", __name__)
    elif name == "ProbabilisticOptimisticRPGHeuristic":
        module = import_module(".probabilistic_rpg", __name__)
    elif name == "TemporalProbabilisticRPGHeuristic":
        module = import_module(".temporal_probabilistic_rpg", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(module, name)

