import argparse
import random
import sys

import numpy as np
_CLI_ARGS = sys.argv[1:]
# unified_planning parses argv at import time; keep only script name.
sys.argv = [sys.argv[0]]
import unified_planning as up

from comdp_plus_no_deadline.domains import DOMAIN_FACTORIES
from comdp_plus_no_deadline.engines import combinationMDP, MDP
from comdp_plus_no_deadline.engines.evaluate import evaluation_loop
from comdp_plus_no_deadline.engines.greedy_solver import (
    combination_greedy_plan,
    regular_greedy_plan,
)
from comdp_plus_no_deadline.scenarios import PRESETS


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def _ground_problem(model, domain_name):
    if domain_name == "nasa_rover":
        grounder = up.engines.compilers.Grounder(model.grounding_map())
    else:
        grounder = up.engines.compilers.Grounder()
    grounding_result = grounder._compile(model.problem)
    return grounding_result.problem


def build_regular_problem(domain_name, object_amount, garbage_amount):
    model = DOMAIN_FACTORIES[domain_name](
        kind="regular",
        deadline=None,
        object_amount=object_amount,
        garbage_amount=garbage_amount,
    )
    ground_problem = _ground_problem(model, domain_name)
    return up.engines.Convert_problem(ground_problem)._converted_problem


def build_combination_problem(domain_name, object_amount, garbage_amount):
    model = DOMAIN_FACTORIES[domain_name](
        kind="combination",
        deadline=None,
        object_amount=object_amount,
        garbage_amount=garbage_amount,
    )
    ground_problem = _ground_problem(model, domain_name)
    convert = up.engines.Convert_problem_combination(model, ground_problem)
    converted_problem = convert._converted_problem
    model.remove_actions(converted_problem)
    return converted_problem


def parse_args():
    parser = argparse.ArgumentParser(
        description="No-deadline CoMDP+ baseline runner (greedy)."
    )
    parser.add_argument("--domain", choices=DOMAIN_FACTORIES.keys(), default="nasa_rover")
    parser.add_argument(
        "--domain_type",
        choices=["regular", "combination"],
        default="combination",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--object_amount", type=int, default=1)
    parser.add_argument("--garbage_amount", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--heuristic_weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario", choices=PRESETS.keys(), default=None)
    return parser.parse_args(_CLI_ARGS)


def main():
    args = parse_args()
    if args.scenario:
        preset = PRESETS[args.scenario]
        args.domain = preset["domain"]
        args.object_amount = preset["object_amount"]

    set_seed(args.seed)

    if args.domain_type == "regular":
        converted_problem = build_regular_problem(
            args.domain, args.object_amount, args.garbage_amount
        )
        mdp = MDP(converted_problem, discount_factor=0.95, reward_mode="terminal")
        params = (mdp, args.max_steps, args.heuristic_weight)
        result = evaluation_loop(args.runs, regular_greedy_plan, params)
    else:
        converted_problem = build_combination_problem(
            args.domain, args.object_amount, args.garbage_amount
        )
        mdp = combinationMDP(converted_problem, discount_factor=0.95, reward_mode="terminal")
        params = (mdp, args.max_steps, args.heuristic_weight)
        result = evaluation_loop(args.runs, combination_greedy_plan, params)

    print("=== No-Deadline CoMDP+ Greedy Baseline ===")
    print(f"domain={args.domain} domain_type={args.domain_type}")
    print(f"runs={result['runs']} success_rate={result['success_rate']:.3f}")
    print(f"avg_makespan={result['avg_makespan']:.3f}")
    print(f"avg_plan_length={result['avg_plan_length']:.3f}")
    print(f"avg_cumulative_reward={result['avg_cumulative_reward']:.3f}")


if __name__ == "__main__":
    main()

