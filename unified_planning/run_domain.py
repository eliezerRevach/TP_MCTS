import os
import time

import dill
import sys
import random

"""For the bash script"""
# Get the current directory (where the script is located)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)  # Add the path to your 'unified_planning' directory

import unified_planning as up
from unified_planning.shortcuts import *
import unified_planning.domains
import unified_planning.engines.solvers.greedy_parallel as greedy_parallel_solver
import unified_planning.engines.solvers.heuristic_tree as heuristic_tree_solver
import numpy as np


# Map each domain name to its class
domains = dict(machine_shop=up.domains.Machine_Shop, nasa_rover=up.domains.Nasa_Rover, stuck_car_1o=up.domains.Stuck_Car_1o,
               stuck_car=up.domains.Stuck_Car, conc=up.domains.Conc, full_conc=up.domains.Full_Conc,
               prob_conc=up.domains.Prob_Conc, best_no_parallel=up.domains.Best_No_Parallel, simple=up.domains.Simple, hosting=up.domains.Hosting, prob_match_cellar=up.domains.Prob_MatchCellar)
# Map each domain name to its pickle file name
domains_files = dict(machine_shop="machine_shop_domain_comb", nasa_rover="nasa_rover_domain_comb",
                     stuck_car_1o="stuck_car_1o_domain_comb", stuck_car="stuck_car_domain_comb", conc="conc_domain_comb",
                     full_conc="full_conc_domain_comb", prob_conc="prob_conc_domain_comb",
                     simple="simple_domain_comb", hosting="hosting_domain_comb", prob_match_cellar="prob_match_cellar_comb")


def _resolved_temporal_heuristic_depth() -> int:
    """CLI depth if set; otherwise match problem deadline; last resort 25."""
    d = up.args.temporal_heuristic_depth
    if d is not None:
        return int(d)
    if up.args.deadline is not None:
        return int(up.args.deadline)
    return 25


def print_stats():
    """
    Prints parameters values
    """
    print(f'Model = {up.args.domain}')
    print(f'Solver = {up.args.solver}')
    print(f'Selection Type = {up.args.selection_type}')
    print(f'Exploration Constant = {up.args.exploration_constant}')
    print(f'Search time = {up.args.search_time}')
    print(f'Search depth = {up.args.search_depth}')
    print(f'Deadline = {up.args.deadline}')
    print(f'Domain Type = {up.args.domain_type}')
    print(f'Object Amount = {up.args.object_amount}')
    print(f'Garbage Action Amount = {up.args.garbage_amount}')
    print(f'K Random Actions = {up.args.k}')
    print(f'Reward Mode = {up.args.reward_mode}')
    print(f'Discount Factor (gamma) = {up.args.discount_factor}')
    print(f'Step Penalty = {up.args.step_penalty}')
    print(f'Seed = {up.args.seed}')
    print(f'Heuristic = {up.args.heuristic_name}')
    _eff_td = _resolved_temporal_heuristic_depth()
    print(
        f'Temporal Heuristic Depth = {_eff_td}'
        + (
            ''
            if up.args.temporal_heuristic_depth is not None
            else ' (default: same as deadline)'
        )
    )
    print(f'Temporal Heuristic Strategy = {up.args.temporal_heuristic_strategy}')
    if up.args.temporal_heuristic_strategy == "atom_backtrack_exact_resolution":
        print(f'Resolution alpha = {getattr(up.args, "resolution_alpha", 2.0)}')
        print(f'Resolution forced_minimum = {getattr(up.args, "resolution_forced_minimum", False)}')
        print(f'Resolution reference_t = {getattr(up.args, "resolution_reference_t", None)}')
    if up.args.temporal_heuristic_strategy == "baseline_pdb":
        print(f'PDB num_patterns = {getattr(up.args, "pdb_num_patterns", 4)}')
        print(f'PDB max_facts_per_pattern = {getattr(up.args, "pdb_max_facts_per_pattern", 4)}')
        print(f'PDB expansion_policy = {getattr(up.args, "pdb_expansion_policy", "max_prob")}')
    print(f'Value Mode = {up.args.value_mode}')
    if up.args.selection_type == 'max_approximation':
        print(f'Max approx alpha = {getattr(up.args, "max_approx_alpha", 1.5)}')
        print(f'Max approx samples = {getattr(up.args, "max_approx_num_samples", 32)}')
        _mas = getattr(up.args, 'max_approx_seed', None)
        print(f'Max approx seed = {_mas if _mas is not None else up.args.seed}')
        print(f'Max approx debug = {getattr(up.args, "max_approx_debug", False)}')


def _greedy_plan_tail_params():
    """Extra kwargs tuple for greedy_parallel.plan (max_approximation selector)."""
    max_approx_seed = getattr(up.args, 'max_approx_seed', None)
    if max_approx_seed is None:
        max_approx_seed = up.args.seed
    return (
        float(getattr(up.args, 'max_approx_alpha', 1.5)),
        int(getattr(up.args, 'max_approx_num_samples', 32)),
        max_approx_seed,
        bool(getattr(up.args, 'max_approx_debug', False)),
    )


def set_seed():
    if up.args.seed is None:
        return
    random.seed(up.args.seed)
    np.random.seed(up.args.seed)


def _apply_pdb_config():
    """Push the baseline_pdb CLI knobs onto the heuristic's class-level config.

    The TemporalProbabilisticRPGHeuristic is constructed deep inside the solvers
    (mcts / greedy_parallel) via .from_problem(...); it reads these class attrs
    when it lazily auto-builds the PDB for the baseline_pdb strategy, so setting
    them here once per process configures every instance without touching the
    solver construction sites."""
    if up.args.temporal_heuristic_strategy != "baseline_pdb":
        return
    from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
        TemporalProbabilisticRPGHeuristic,
    )

    TemporalProbabilisticRPGHeuristic._PDB_AUTOBUILD = True
    TemporalProbabilisticRPGHeuristic._PDB_NUM_PATTERNS = int(
        getattr(up.args, "pdb_num_patterns", 4)
    )
    TemporalProbabilisticRPGHeuristic._PDB_MAX_FACTS_PER_PATTERN = int(
        getattr(up.args, "pdb_max_facts_per_pattern", 4)
    )
    TemporalProbabilisticRPGHeuristic._PDB_EXPANSION_POLICY = str(
        getattr(up.args, "pdb_expansion_policy", "max_prob")
    )
    TemporalProbabilisticRPGHeuristic._PDB_SEED_PER_GOAL = bool(
        getattr(up.args, "pdb_seed_per_goal", True)
    )
    TemporalProbabilisticRPGHeuristic._PDB_GROW_UNTIL_COVERS = bool(
        getattr(up.args, "pdb_grow_until_covers", False)
    )
    TemporalProbabilisticRPGHeuristic._PDB_COVER_HARD_CAP = getattr(
        up.args, "pdb_cover_hard_cap", None
    )
    # Make pattern generation reproducible across runs of the same config.
    TemporalProbabilisticRPGHeuristic._PDB_SEED = up.args.seed
    # Reset the last-correction handle so post-run stats reflect this run.
    TemporalProbabilisticRPGHeuristic._LAST_PDB_CORRECTION = None


def _report_pdb_stats():
    """Print PDB pattern/usage stats after a baseline_pdb run."""
    if up.args.temporal_heuristic_strategy != "baseline_pdb":
        return
    from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
        TemporalProbabilisticRPGHeuristic,
    )

    correction = TemporalProbabilisticRPGHeuristic._LAST_PDB_CORRECTION
    if correction is None:
        print("[PDB] no correction was built (no goal facts / autobuild off).")
        return
    correction.log_summary()


def run_regular(domain, runs, domain_type, deadline, search_time, search_depth, exploration_constant, object_amount, garbage_amount,
                selection_type='avg', k=10, heuristic_name='trpg', temporal_heuristic_depth=25,
                temporal_heuristic_strategy='baseline', tree_depth=3, value_mode='tp_mcts'):
    """
    Run split action to start and end actions logic - TP-MCTS approach
    """
    assert domain in domains
    print_stats()
    set_seed()
    _apply_pdb_config()
    start_time = time.time()

    model = domains[domain](kind=domain_type, deadline=deadline, object_amount=object_amount, garbage_amount=garbage_amount)

    # ground the actions
    if domain == 'nasa_rover':
        grounder = up.engines.compilers.Grounder(model.grounding_map())
    else:
        grounder = up.engines.compilers.Grounder()

    grounding_result = grounder._compile(model.problem)
    ground_problem = grounding_result.problem

    # Transform each duration action to start and end
    convert_problem = Convert_problem(ground_problem)
    converted_problem = convert_problem._converted_problem

    end_time = time.time()

    elapsed_time = end_time - start_time

    # Print the result and elapsed time
    print(f"Compilation Time {domain} object={object_amount}, garbage={garbage_amount}: {elapsed_time} seconds")
    print(f"Action amount= {len(ground_problem.actions)}, Proposition amount= {len(ground_problem.explicit_initial_values)}")


    mdp = MDP(
        converted_problem,
        discount_factor=up.args.discount_factor,
        reward_mode=up.args.reward_mode,
        step_penalty=up.args.step_penalty,
    )

    params = (
        mdp,
        90,
        search_time,
        search_depth,
        exploration_constant,
        selection_type,
        k,
        heuristic_name,
        temporal_heuristic_depth,
        temporal_heuristic_strategy,
    )
    if up.args.solver == 'greedy_parallel':
        greedy_params = params + _greedy_plan_tail_params()
        up.engines.solvers.evaluate.evaluation_loop(runs, greedy_parallel_solver.plan, greedy_params)
    elif up.args.solver == 'heuristic_tree':
        tree_params = params + (tree_depth,)
        up.engines.solvers.evaluate.evaluation_loop(runs, heuristic_tree_solver.plan, tree_params)
    else:
        mcts_params = params + (value_mode, up.args.final_selection)
        up.engines.solvers.evaluate.evaluation_loop(runs, up.engines.solvers.mcts.plan, mcts_params)

    _report_pdb_stats()


def create_combination_domain(domain, deadline, object_amount, garbage_amount):
    """
        Create combination of domain - creates combination actions
    """
    model = domains[domain](kind='combination', deadline=deadline, object_amount=object_amount, garbage_amount=garbage_amount)

    # ground the actions
    if domain == 'nasa_rover':
        grounder = up.engines.compilers.Grounder(model.grounding_map())
    else:
        grounder = up.engines.compilers.Grounder()
    grounding_result = grounder._compile(model.problem)
    ground_problem = grounding_result.problem

    convert_combination_problem = Convert_problem_combination(model, ground_problem)
    converted_problem = convert_combination_problem._converted_problem
    model.remove_actions(converted_problem)

    return convert_combination_problem


def run_combination(domain, runs, solver, deadline, search_time, search_depth, exploration_constant, object_amount, garbage_amount,
                    selection_type='avg', k=10, heuristic_name='trpg', temporal_heuristic_depth=25,
                    temporal_heuristic_strategy='baseline', value_mode='tp_mcts'):
    """
    Run the combination logic - Mausem and Weld approach
    """
    assert domain in domains
    print_stats()
    set_seed()
    _apply_pdb_config()

    # create the pickle file name associated with the domain
    file_name = './pickle_domains/' + domains_files[domain]
    if domain == 'prob_conc' or domain == 'simple':
        file_name += "_" + str(garbage_amount)
    if domain == 'nasa_rover' or domain == 'stuck_car':
        file_name += "_" + str(object_amount)
    if domain == 'machine_shop':
        file_name += "_" + str(object_amount)

    file_name += '.pkl'
    try:
    # Try to load the saved object

        with open(file_name, "rb") as file:
            convert_combination_problem = dill.load(file)
            converted_problem = convert_combination_problem._converted_problem
            split_problem = convert_combination_problem._split_problem

        deadline_timing = Timing(delay=deadline, timepoint=Timepoint(TimepointKind.START))
        converted_problem.set_deadline(deadline_timing)
        split_problem.set_deadline(deadline_timing)

    except FileNotFoundError:
        # If the file doesn't exist, create a new instance from scratch
        convert_combination_problem = create_combination_domain(domain, deadline, object_amount, garbage_amount)
        converted_problem = convert_combination_problem._converted_problem
        split_problem = convert_combination_problem._split_problem

    mdp = combinationMDP(
        converted_problem,
        discount_factor=up.args.discount_factor,
        reward_mode=up.args.reward_mode,
        step_penalty=up.args.step_penalty,
    )
    split_mdp = MDP(
        split_problem,
        discount_factor=up.args.discount_factor,
        reward_mode=up.args.reward_mode,
        step_penalty=up.args.step_penalty,
    )

    if solver == 'rtdp':
        params = (
            mdp,
            split_mdp,
            90,
            search_time,
            search_depth,
            heuristic_name,
            temporal_heuristic_depth,
            temporal_heuristic_strategy,
        )
        up.engines.solvers.evaluate.evaluation_loop(runs, up.engines.solvers.rtdp.plan, params)
    elif solver == 'greedy_parallel':
        greedy_params = (
            mdp,
            90,
            search_time,
            search_depth,
            exploration_constant,
            selection_type,
            k,
            heuristic_name,
            temporal_heuristic_depth,
            temporal_heuristic_strategy,
        ) + _greedy_plan_tail_params()
        up.engines.solvers.evaluate.evaluation_loop(runs, greedy_parallel_solver.plan, greedy_params)
    else:
        params = (
            mdp,
            split_mdp,
            90,
            search_time,
            search_depth,
            exploration_constant,
            selection_type,
            k,
            heuristic_name,
            temporal_heuristic_depth,
            temporal_heuristic_strategy,
        )
        mcts_params = params + (value_mode,)
        up.engines.solvers.evaluate.evaluation_loop(runs, up.engines.solvers.mcts.combination_plan, mcts_params)

    _report_pdb_stats()




if up.args.domain_type == 'combination':
    run_combination(domain=up.args.domain, runs=up.args.runs, solver=up.args.solver, deadline=up.args.deadline,
                    search_time=up.args.search_time,
                    search_depth=up.args.search_depth, exploration_constant=up.args.exploration_constant,
                    selection_type=up.args.selection_type, object_amount=up.args.object_amount,
                    garbage_amount=up.args.garbage_amount, k=up.args.k,
                    heuristic_name=up.args.heuristic_name,
                    temporal_heuristic_depth=_resolved_temporal_heuristic_depth(),
                    temporal_heuristic_strategy=up.args.temporal_heuristic_strategy,
                    value_mode=up.args.value_mode)
else:
    run_regular(domain=up.args.domain, domain_type=up.args.domain_type, runs=up.args.runs, deadline=up.args.deadline,
                search_time=up.args.search_time,
                search_depth=up.args.search_depth, exploration_constant=up.args.exploration_constant,
                selection_type=up.args.selection_type, object_amount=up.args.object_amount,
                garbage_amount=up.args.garbage_amount, k=up.args.k,
                heuristic_name=up.args.heuristic_name,
                temporal_heuristic_depth=_resolved_temporal_heuristic_depth(),
                temporal_heuristic_strategy=up.args.temporal_heuristic_strategy,
                tree_depth=up.args.tree_depth,
                value_mode=up.args.value_mode)
