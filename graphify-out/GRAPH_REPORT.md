# Graph Report - TP_MCTS  (2026-09-14)

## Corpus Check
- 237 files · ~796,870 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4766 nodes · 10286 edges · 199 communities (159 shown, 40 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 894 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4b9b6ea5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 86
- Community 87
- Community 88
- Community 89
- Community 90
- Community 91
- Community 92
- Community 93
- Community 94
- Community 95
- Community 96
- Community 97
- Community 98
- Community 99
- Community 100
- Community 101
- Community 102
- Community 103
- Community 104
- Community 105
- Community 106
- Community 107
- Community 108
- Community 109
- Community 110
- Community 111
- Expression
- Community 113
- Community 114
- Community 115
- Community 116
- Community 117
- Community 118
- Action
- Community 120
- main
- Community 122
- Community 123
- Community 124
- Community 125
- Community 126
- Community 127
- Community 128
- shortcuts.py
- Community 130
- Community 131
- Community 132
- Community 133
- Community 134
- Community 135
- Community 136
- Community 137
- Community 138
- Community 139
- Community 140
- Community 141
- Community 143
- Community 144
- Community 145
- Community 146
- Community 147
- Path
- plan.py
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native AGENTS.md integration
- graphify reference: incremental update and cluster-only
- DurativeAction
- Thesis Ideation Prompts (Optional)
- TP-MCTS (Temporal Planning Monte Carlo Tree Search)
- .heuristic_expected_time
- .is_int_constant
- FreeVarsExtractor
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- extraction-spec.md
- collect_global_frontier
- .kind
- .is_global
- presets.py
- .check_stn
- .copy_stn
- .create_Snode
- .And
- Pattern
- __init__.py
- TestTableStrategyEngine
- TestChainedFootprints
- AnyChecker
- compare_paths_table_vs_admissible.py
- .is_int_constant
- sweep_paths_table_gap.py
- .create_Snode_root_interval
- ._ensure_admissible_lp_bound
- Critical path PDB runtime investigation
- ._frontier_aligned_value
- Action
- .copy_stn
- Fact
- test_greedy_parallel.py
- .interval
- TestSpecSection3Placements
- cumulative_merge_truncate
- TestTableStrategyEngine
- TestChainedFootprints
- .current_time
- .name
- nasa_obj3_patterns.md
- nasa_review.md

## God Nodes (most connected - your core abstractions)
1. `FNode` - 326 edges
2. `TemporalProbabilisticRPGHeuristic` - 260 edges
3. `C_MCTS` - 113 edges
4. `Environment` - 81 edges
5. `get_environment()` - 60 edges
6. `StrictDBM` - 57 edges
7. `MDP` - 57 edges
8. `Row` - 55 edges
9. `Segment` - 54 edges
10. `UPProblemDefinitionError` - 53 edges

## Surprising Connections (you probably didn't know these)
- `ProgressPDB` --uses--> `CriticalPathPDB`  [INFERRED]
  artifacts/critical_path_pdb_review_20260914/benchmark_nasa.py → comdp_plus_no_deadline/engines/critical_path_pdb.py
- `Base_MCTS` --uses--> `TemporalSTNPDBHeuristic`  [INFERRED]
  unified_planning/engines/solvers/mcts.py → comdp_plus_no_deadline/engines/temporal_stn_pdb.py
- `C_MCTS` --uses--> `TemporalSTNPDBHeuristic`  [INFERRED]
  unified_planning/engines/solvers/mcts.py → comdp_plus_no_deadline/engines/temporal_stn_pdb.py
- `MCTS` --uses--> `TemporalSTNPDBHeuristic`  [INFERRED]
  unified_planning/engines/solvers/mcts.py → comdp_plus_no_deadline/engines/temporal_stn_pdb.py
- `Base_MCTS` --uses--> `WindowsILAOPDBHeuristic`  [INFERRED]
  unified_planning/engines/solvers/mcts.py → comdp_plus_no_deadline/engines/windows_ilao_pdb.py

## Import Cycles
- None detected.

## Communities (199 total, 40 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.17
Nodes (10): OutcomeDetail, _feasible_actions(), _fit_action_stn(), FixedTailExpectimaxEvaluator, Expectimax prefix evaluation for fixed-tail MCTS.  V(s) = max_a Q(s,a) over ST, Stop expanding expectimax when time budget or step depth is reached., V(s) using only STN-feasible actions (MCTS children), not all MDP-legal actions., STN-feasible legal actions (same filter as greedy_parallel / MCTS children). (+2 more)

### Community 1 - "Community 1"
Cohesion: 0.02
Nodes (25): FNodeContent, FNode, object, Returns the `id` of this expression., Returns the `Environment` in which this expression exists., Returns the `Type` of this expression., Returns all the names contained in this expression., Returns the simplified version of this expression.          The simplification (+17 more)

### Community 2 - "Community 2"
Cohesion: 0.10
Nodes (35): build_patterns(), compute_earliest_times(), Delete-relaxed earliest achievement layer per fact.      The only thing the patt, Forward sweep of the joint distribution over the pattern's facts.      Returns,, One pattern per goal fact (capped), each grown independently., solve_survivor_pattern(), SurvivorPattern, _achiever() (+27 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (33): build_interest_points(), v18 section 1: the timestamps at which an effect can land, ``T``.      ``T`` is, _durative_escape_house(), _heuristic_for(), _load_aggregation_for_strategy(), Tests for the phase-augmented ("time loop") survivor-PDB (survivor_pdb_loop).  T, The escape house with SLOW searches — where the phase actually bites., Same patterns, same gates — only the sweep changes, and only downward. (+25 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (43): CachedPTRPGTable, _clamp_probability(), _extract_state_facts(), Fact, Non-negative per-action scores from forward-layer precondition support         a, Lower-bound estimate on P(all goals by deadline) using correlation-aware DP., Upper-bound estimate on P(all goals by deadline) using correlation-aware DP., Debug snapshot for one temporal layer. (+35 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (37): _collapse_phases(), Fact, Forward sweep of the joint distribution, with per-achiever phases.      Drop-in, Merge phase-augmented states sharing a mask, forgetting every counter.      The, solve_survivor_loop_pattern(), _achiever(), _escape_house_pattern(), _random_pattern() (+29 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (42): ExpressionManager, BoolExpression, Expression, Fraction, object, Creates the unified_planning expressions if it hasn't been created yet in the en, Returns a conjunction of terms.         This function has polymorphic n-argumen, Returns an disjunction of terms.         This function has polymorphic n-argume (+34 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (23): advance_phase(), anchor_residue(), attempts_in_step(), _ceil_div(), completions_in_step(), decode_phase(), encode_phase(), Phase-augmented ("time loop") survivor-PDB sweep.  Strategy ``survivor_pdb_loop` (+15 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (47): BackwardPlanPDB, Plan, Fact, Backward plan PDB: regression from the goal, no temporal constraints.  A delib, Regress from the goal. Each STATE is expanded once; plans are paths., Enumerate event sequences from ``state`` to the goal state (DFS)., One backward step: un-do an END (action becomes running) or a START., Does this plan end the running actions in the order the clock says?          T (+39 more)

### Community 9 - "Community 9"
Cohesion: 0.06
Nodes (35): AndGammaCalibrator, AndGammaConfig, build_candidate_pairs(), build_components(), build_structural_context(), _clamp01(), classify_component(), ComponentInfo (+27 more)

### Community 10 - "Community 10"
Cohesion: 0.08
Nodes (18): DP-relevant add facts of an action, keyed by action name.          Returns the f, Cache telemetry for ``survivor_pdb_loop`` (hits, misses, sweeps)., Cache telemetry for ``survivor_pdb_lazy`` (hits, misses, sweeps)., Return (and optionally print) the headline mutex-survival metric.          Accum, Duration-aware optimistic relaxed heuristic with fixed temporal depth.      Comp, Return (and optionally reset) the path-mutex survival / AND-feasibility, Product of component gammas for an action's preconditions (≥ 0)., Memoized front-end for :meth:`_compute_kmutex_actions_are_mutex`.          The s (+10 more)

### Community 11 - "Community 11"
Cohesion: 0.12
Nodes (8): Max over k sampled actions; each child gets fixed-tail leaf eval (K rollouts ins, Create a new Snode for the state `state` with parent `parent`, Create a new Snode for the state `state` with parent `parent`          In this, Traverse the tree until reaching a leaf node., Traverse the tree until reaching a leaf node.         Selection with max logic, Traverse the tree until reaching a leaf node.         Selection with root inter, Traverse the tree until reaching a leaf node., Traverse the tree until reaching a leaf node.         Selection with max logic

### Community 12 - "Community 12"
Cohesion: 0.08
Nodes (9): Base_MCTS, combination_plan(), plan(), Global frontier Option A: argmax aligned_value, expand selected node only., Evaluate the temporal_probabilistic_rpg heuristic, threading the baseline_cached, :param root_node: the root node of the MCTS tree         :return: returns the b, Return the most-visited child (robust child / argmax-N)., Execute the MCTS algorithm from the initial state given, with timeout in seconds (+1 more)

### Community 13 - "Community 13"
Cohesion: 0.06
Nodes (29): Achiever, _clamp01(), _fact_sort_key(), _has_shared_fact(), marginal_consistent_or_hazard(), MarginalConsistentORBound, _PreparedFormula, Fact (+21 more)

### Community 14 - "Community 14"
Cohesion: 0.12
Nodes (31): Action, alts_and(), alts_or(), _cap_groups(), _coalesce_union(), cut_emit_rows(), cut_fallback_row(), cut_guaranteed_rep() (+23 more)

### Community 15 - "Community 15"
Cohesion: 0.04
Nodes (40): get_all_fluent_exp(), get_ith_fluent_exp(), Returns the ith ground fluent expression., _BoolType, domain_item(), domain_size(), _IntType, is_compatible_type() (+32 more)

### Community 16 - "Community 16"
Cohesion: 0.13
Nodes (26): achievers_of_facts(), best_joint_add_distribution(), build_pattern(), _clamp01(), _collapse_ages(), compute_gate(), conditional_hazards(), first_positive_layer() (+18 more)

### Community 17 - "Community 17"
Cohesion: 0.04
Nodes (17): CombinationAction, Returns this `Action` `Environment`., Environment, IO, Returns the environment's `TypeChecker`., Returns the environment's `Factory`., Returns the environment's `Simplifier`., Returns the environment's `Substituter`. (+9 more)

### Community 18 - "Community 18"
Cohesion: 0.06
Nodes (17): _apply_function_to_effect(), check_conflicting_effects(), check_conflicting_probabilistic_effects(), Effect, ProbabilisticEffect, This module defines the `Effect` class. A basic `Effect` has a `fluent` and an, Returns this `Effect's Environment`., Returns the `Fluents` that is modified by this `Effect`. (+9 more)

### Community 19 - "Community 19"
Cohesion: 0.18
Nodes (15): pick_best_action(), Greedy MDP dispatcher (same as ``plan()``) until goal, dead end, or deadline., simulate_greedy_mdp_until_terminal(), _log_rollout_step(), pick_greedy_rollout_action(), ptrpg_guided_terminal_rollout(), PTRPG-guided terminal rollout for MCTS leaf evaluation.  Uses the same greedy, remaining_deadline() (+7 more)

### Community 20 - "Community 20"
Cohesion: 0.03
Nodes (44): Fraction, Return the given subexpression at the given position.          :param idx: The, Returns `True` if the expression is a constant, `False` otherwise., Returns the constant value stored in this expression., Return constant `boolean` value stored in this expression., Return constant `integer` value stored in this expression., Return constant `real` value stored in this expression., Return the `Fluent` stored in this expression. (+36 more)

### Community 21 - "Community 21"
Cohesion: 0.09
Nodes (27): combine_precondition_footprints(), PathMutexInstrumentation, prune_expired(), True iff the half-open windows ``[start, end)`` intersect.      Touching endpoin, Per-layer OR-hazard ``H_t(f)`` via the K-bounded :func:`insert_or_absorb`     ta, Drop registered segments that can no longer overlap anything new.      A segment, Best-effort UNION of one representative recent footprint per precondition     —, Accumulates the per-layer OR-hazard table HIT metrics. (+19 more)

### Community 22 - "Community 22"
Cohesion: 0.08
Nodes (28): BaseCombinationMDP, BaseMDP, evaluation_loop(), combination_greedy_plan(), _effective_temporal_depth(), _get_probabilistic_rpg_heuristic(), _get_temporal_probabilistic_rpg_heuristic(), PlanResult (+20 more)

### Community 23 - "Community 23"
Cohesion: 0.12
Nodes (14): _TprpgHeuristicAdapter, _aggregation_for_strategy(), _effective_temporal_depth(), _get_rollout_aligned_evaluator(), Pick the goal-aggregation for heuristic_score based on the strategy.      `bas, Build a RolloutAlignedConfig from unified_planning.parser CLI args., Build (once per MDP+suffix) a RolloutAlignedEvaluator bound to this MDP., Remaining execution time per in-flight action type, read off the search's STN. (+6 more)

### Community 24 - "Community 24"
Cohesion: 0.05
Nodes (7): Fluent, Returns the `Fluent` `Type`., Returns the `Fluent` `signature`.         The `signature` is the `List` of `Par, Returns the `Fluent` arity.          IMPORTANT NOTE: this property does some c, Returns the `Fluent` `Environment`., Returns a fluent expression with the given parameters.          :param args: T, Returns the `Fluent` `name`.

### Community 25 - "Community 25"
Cohesion: 0.09
Nodes (14): Convert_problem, convert instantaneous actions from `model` actions to be `engines` actions, Finding mutex actions and adding a precondition that they can't be executed in p, Check if two actions are mutex          :param action: The checked action, Check if two actions are soft mutex          :param action: The checked action, returns all the negative end assignments of durative `action` to fluents in, returns all the negative start assignments of `action` to fluents in         if, returns all the positive start assignments of `action` to fluents         if du (+6 more)

### Community 26 - "Community 26"
Cohesion: 0.04
Nodes (35): TypeError, Action, DurativeAction, implAction, InstantaneousAction, InstantaneousEndAction, InstantaneousStartAction, NoOpAction (+27 more)

### Community 27 - "Community 27"
Cohesion: 0.16
Nodes (12): build_mdp(), main(), Single-call comparison: baseline_admissible_paths_table vs baseline_admissible., build_mdp(), main(), Single-call comparison: baseline_admissible_survivor_pdb vs baseline_admissible., build_converted_problem(), main() (+4 more)

### Community 28 - "Community 28"
Cohesion: 0.07
Nodes (24): Grounder, GrounderHelper, Action, Returns an `Iterator` over all the possible grounded `Actions` of the `Problem`, Takes in input an `Action` and returns the iterator over all the possible parame, Grounder class: the `Grounder` takes a :class:`~unified_planning.model.Problem`, Takes an instance of a :class:`~unified_planning.model.Problem` and the `GROUNDI, This class gives the capability of grounding a :class:`~unified_planning.model.P (+16 more)

### Community 29 - "Community 29"
Cohesion: 0.04
Nodes (34): DurativeAction, InstantaneousAction, check_and_simplify_conditions(), check_and_simplify_preconditions(), create_action_with_given_subs(), create_effect_with_given_subs(), create_precondition_with_given_subs(), create_probabilistic_effect_with_given_subs() (+26 more)

### Community 30 - "Community 30"
Cohesion: 0.07
Nodes (26): make(), measure(), Read-only benchmark of the existing heuristic; no solver modifications., report(), CountedPDB, Count repeated backup work without changing the heuristic's decisions., Branches, CriticalPathPDB (+18 more)

### Community 31 - "Community 31"
Cohesion: 0.12
Nodes (34): is_option_a_strategy(), option_a_ptrpg_suffix(), apply_heuristic_alias_overrides(), best_action_name(), build_mcts(), build_mdp(), configure_fixed_tail_cli(), configure_max_approx_cli() (+26 more)

### Community 32 - "Community 32"
Cohesion: 0.08
Nodes (34): aligned_value_for_node(), build_option_a_evaluator(), collect_global_frontier(), compute_H_frontier(), format_option_a_debug_row(), node_elapsed(), option_a_config_from_cli(), OptionAConfig (+26 more)

### Community 33 - "Community 33"
Cohesion: 0.10
Nodes (24): and_components(), and_cumulative_bound(), and_emit_rows(), and_has_mutex(), and_support_kernelized(), AndKernelResult, cut_or_hazard(), _fact_max() (+16 more)

### Community 34 - "Community 34"
Cohesion: 0.06
Nodes (5): Parameter, Represents an :func:`action parameter <unified_planning.model.Action.parameters>, Returns the `Parameter` `name`., Returns the `Parameter` `type`., Return the `Parameter` `Environment`

### Community 35 - "Community 35"
Cohesion: 0.06
Nodes (5): Represents a variable; a `Variable` has a name and a type., Returns the `Variable` name., Returns the `Variable` `Type`., Return the `Variable` `Environment`., Variable

### Community 36 - "Community 36"
Cohesion: 0.09
Nodes (22): admissible_and_support(), _clamp_probability(), cumulative_retry_update(), propagate_admissible_temporal_rpg(), Fact, Probabilistic Temporal RPG Heuristic — admissible upper-bound version.  Heuristi, Clamp a probability into ``[0, 1]`` (doc Section 9 domain guard)., AND layer (doc 5.1): admissible upper bound on the joint probability that all (+14 more)

### Community 37 - "Community 37"
Cohesion: 0.14
Nodes (20): _accumulate_alternatives(), common_footprint(), exact_component_value(), _facts_mutex(), guaranteed_mutex(), insert_or_absorb(), _merge_alternatives(), One row of the K-bounded OR fact table: an abstract policy alternative.      ``p (+12 more)

### Community 38 - "Community 38"
Cohesion: 0.11
Nodes (12): DagWalker, Returns True, independently from the children's value., Returns False, independently from the children's value., Returns None, independently from the children's value., Returns expression, independently from the childrens's value., Returns True if any of the children returned True., Returns True if all the children returned True., DagWalker treats the expression as a DAG and performs memoization of the     in (+4 more)

### Community 39 - "Community 39"
Cohesion: 0.11
Nodes (11): _non_singleton_component(), Tests for the baseline_survival_and_gamma temporal heuristic strategy and the A, atom_backtrack_exact_resolution_and_gamma: resolution backtrack + gamma., Minimal action-like object compatible with the heuristic's model builder., SyntheticAction, TestCalibrationStatistics, TestCaseA_NoDependency, TestCaseBCD_ComponentFactors (+3 more)

### Community 40 - "Community 40"
Cohesion: 0.13
Nodes (14): T, DeltaNeighbors, DeltaSimpleTemporalNetwork, Any, Adds the constraint `x - y <= b`. This gives an upper bound to the time, Checks the consistency of this STN., Returns the assignment to the given event in the minimal-makespan consistent sol, Check if there is a harder constraint from x to y (+6 more)

### Community 41 - "Community 41"
Cohesion: 0.10
Nodes (36): _execution_conflict_table(), Optimal probability of achieving the pattern goal within ``remaining``.      Pas, Static overlap mutexes encoded by converted ``inExecution`` facts.      ``conver, solve_pattern(), _exhaustive_deterministic(), _pattern(), Tests for the exact pattern-MDP heuristic (``exact_pattern_mdp``).  The anchors, Redundant starts disappear and useful groups jump to their boundary. (+28 more)

### Community 42 - "Community 42"
Cohesion: 0.09
Nodes (22): _clamp01(), _fold_free_support(), kmutex_or_hazard(), KMutexInstrumentation, KMutexORResult, _merge_closest_pair(), Mutex-aware K-bounded OR-layer tightening for the admissible PTRPG.  Heuristic n, Fold a free (non-mutex) support into an existing row, clearing its footprint. (+14 more)

### Community 43 - "Community 43"
Cohesion: 0.14
Nodes (7): Represents a `STNPlan`. A Simple Temporal Network plan is a generalization of, Returns all the constraints given by this `STNPlan`. Subsumed constraints, Returns a new `STNPlan` where every `ActionInstance` of the current plan is repl, This function takes a `PlanKind` and returns the representation of `self`, Returns True if exists a time assignment for each STNPlanNode that         does, Returns the earliest tine node can be executed according to the STN constraints, STNPlan

### Community 44 - "Community 44"
Cohesion: 0.12
Nodes (8): Nasa_Rover, sample_rock_good Action, turn_on_dropping Action, turn_on_good_hand Action, communicate_rock_data Action, communicate_image_data Action, ObjectExp(), Returns an expression for the given object.      :param obj: The `Object` that

### Community 45 - "Community 45"
Cohesion: 0.06
Nodes (25): main(), main(), NASA rover, obj=3, D=25: one independently built PDB for each goal., save(), ProgressPDB, Bounded NASA rover measurements of the unchanged critical-path PDB., _action_adds(), _bump_serial() (+17 more)

### Community 46 - "Community 46"
Cohesion: 0.10
Nodes (11): Convert_problem_combination, checks if one of the actions already in the combination is in mutex with the can, adds as a combination action to the problem the `combination`          :param, convert actions from `model` actions to be `engines` actions         This is fo, Finding mutex actions and adding a precondition that they can't be executed in p, Check if two actions are mutex          :param action: The checked action, Adding to the `conflicting_action`, and `action` a precondition that they would, The function adds as an action all combinations of durative actions that can run (+3 more)

### Community 47 - "Community 47"
Cohesion: 0.17
Nodes (4): Machine_Shop, Immersionpaint Action, OverallPreconditionTiming(), Returns the overall timing of an :class:`~unified_planning.model.Action`.

### Community 48 - "Community 48"
Cohesion: 0.10
Nodes (16): LogLevel, LogMessage, PlanGenerationResult, PlanGenerationResultStatus, Enum, This class is composed by a message and the Enum LogLevel indicating     this m, This class represents the base class for results given by the engines to the use, This predicate should state if the Result is definitive or if it can be improved (+8 more)

### Community 49 - "Community 49"
Cohesion: 0.19
Nodes (21): _and_n_facts(), _and_pairwise(), build_action_specs(), _clamp01(), compute_correlation_preplanning(), CorrActionSpec, _extract_effect_delay_steps(), joint_add_distribution_from_action() (+13 more)

### Community 51 - "Community 51"
Cohesion: 0.19
Nodes (17): _apply_pdb_config(), create_combination_domain(), _greedy_plan_tail_params(), print_stats(), Push the baseline_pdb CLI knobs onto the heuristic's class-level config.      Th, Print PDB pattern/usage stats after a baseline_pdb run., Run split action to start and end actions logic - TP-MCTS approach, Create combination of domain - creates combination actions (+9 more)

### Community 52 - "Community 52"
Cohesion: 0.14
Nodes (15): _build_achiever_index(), generate_patterns(), grow_pattern(), pattern_covers_any_action(), PDBAction, Fact, Durative probabilistic action with explicit joint outcomes., Probability this action sets ``fact`` true in one application. (+7 more)

### Community 53 - "Community 53"
Cohesion: 0.15
Nodes (4): SyntheticAction, TestPathMutexStrategy, test_direct_strategy_names_are_accepted(), test_strategy_name_is_accepted()

### Community 54 - "Community 54"
Cohesion: 0.13
Nodes (10): FixedTailExpectimaxGuards, _MockAction, _MockState, _MockSTN, _MockSTNNode, Unit tests for fixed-tail expectimax prefix evaluation., TestFixedTailExpectimax, TestFixedTailExpectimaxMCTSSeed (+2 more)

### Community 55 - "Community 55"
Cohesion: 0.08
Nodes (13): combinationMDP, MDP, Apply the action to this state to produce the next state., Gets the add and delete effect of the prob_outcome index effect, :param action: draw the outcome of the probabilistic effects         :return: th, Enumerates next-state distribution for the given (state, action).          This, :return: the initial state of the problem, Checks if all the goal predicates hold in the `state`         and there are no a (+5 more)

### Community 56 - "Community 56"
Cohesion: 0.07
Nodes (8): ActionQueue, CombinationState, QueueNode, holds action and it's duration left, Compares two nodes based on the duration left, Actions currently in execution and the remaining duration left for their executi, Get the actions that have the smallest duration left.         There can be seve, Extract delta from each of the actions in data: duration_left = duration_left -

### Community 57 - "Community 57"
Cohesion: 0.06
Nodes (47): _first_two_starts(), Tests for Part I of the symbolic-STN temporal PDB (``temporal_stn_pdb``).  Anc, Section 9's overall-condition shape, in the form this codebase HAS one.      `, One retryable action: V = 1 - (1-p)^floor(D/d).      ``inExecution`` forbids a, Section 2: "Do not impose E_a <= D ... a start effect may achieve the     goal, Section 5: "Budget limits computation, not the domain: UNKNOWN is not     value, Whatever the budget, [lo, hi] must contain the fully expanded value., Section 3 enumerated at START time must give the same value as letting     the (+39 more)

### Community 58 - "Community 58"
Cohesion: 0.09
Nodes (33): _exec_facts(), _fold(), _is_exec_fact(), lookup(), Fact, Fraction, Survivor sweep: delete-relaxed occupancy over (facts, phases) -- ``rpg_exact_sta, ``P(goal <= t)`` for every timestamp ``t <= horizon`` from one query.      ``run (+25 more)

### Community 59 - "Community 59"
Cohesion: 0.05
Nodes (35): ClosedTimeInterval(), EndTiming(), GlobalEndTiming(), GlobalStartTiming(), LeftOpenTimeInterval(), OpenTimeInterval(), Fraction, Class used to define the point in the time from which a :class:`~unified_plannin (+27 more)

### Community 60 - "Community 60"
Cohesion: 0.11
Nodes (4): Conc, Full_Conc, Returns the start timing of an :class:`~unified_planning.model.Action`.      F, StartPreconditionTiming()

### Community 61 - "Community 61"
Cohesion: 0.09
Nodes (14): Engine, EngineMeta, OperationMode, Enum, type, Sets the flag deciding if a fail on the problem's :func:`kind <unified_planning., Manages entering a Context (i.e., with statement), Manages exiting from Context (i.e., with statement) (+6 more)

### Community 62 - "Community 62"
Cohesion: 0.20
Nodes (6): Tests for rollout-aligned common-horizon PTRPG:    * the MDP-agnostic wrapper (R, Spec sanity (#10): align frontier nodes to the deepest elapsed.      Frontier A:, SyntheticAction, TestBaselineSurvivalResolution, TestFrontierStrategyRecognition, TestOptionASanity

### Community 63 - "Community 63"
Cohesion: 0.06
Nodes (15): Object, Represents an `Object` of the `unified_planning` library.      An `Object` con, Returns the `Object` `name`., Returns the `Object` `Type`., Return the `Object` `Environment`, ExpressionQuantifiersRemover, This walker is used to remove all the quantifiers from an expression by substitu, This method takes in input an expression that might contain quantifiers and a `p (+7 more)

### Community 64 - "Community 64"
Cohesion: 0.05
Nodes (16): Action, InstantaneousAction, Represents an instantaneous action., This is the `Action` interface., Returns the `list` of the `Action` `preconditions`., Removes all the `Action preconditions`, Returns the `list` of the `Action effects`., Returns the `list` of the `Action effects`. (+8 more)

### Community 65 - "Community 65"
Cohesion: 0.12
Nodes (11): MCTS, _normalize_max_approximation_selection(), Simulate until a terminal state, ``selection_type='max_approximation'`` is the single switch (matching the     g, Choose a random action. Heustics can be used here to improve simulations., Original MCTS solver implementation., Create a new Snode for the state `state` with parent `parent`, Create a new Snode for the state `state` with parent `parent`         In this a (+3 more)

### Community 66 - "Community 66"
Cohesion: 0.19
Nodes (6): Place a rock under the car Action, Search a rock Action             the robot can find a one of the rocks, Push Gas Pedal Action         The probability of getting the car out is lower t, Push Car Action             The probability of getting the car out is higher th, Init things that can be pushed, Stuck_Car

### Community 67 - "Community 67"
Cohesion: 0.32
Nodes (5): Tunables for rollout-aligned common-horizon evaluation., RolloutAlignedConfig, FakeState, _make_evaluator(), TestRolloutAlignedWrapper

### Community 68 - "Community 68"
Cohesion: 0.08
Nodes (19): _achievers_share_fact(), build_resolution_delta_schedule(), _env_int(), _env_str(), _grid_ceil(), Read an int from the environment, falling back to ``default`` on error., True iff a precondition fact appears in two distinct achiever records.      Reco, Read a (lower-cased, stripped) string from the environment. (+11 more)

### Community 69 - "Community 69"
Cohesion: 0.09
Nodes (22): DurativeOp, Instance, Node, Apply one processing order, expanding the outcome cross-product., Section 4's condition check, at the event that is actually firing., One startable action type, with its two snap actions and duration.      ``key`, One running or completed action instance.      ``serial`` keeps repeated insta, The spec's ``X``: an annotated STN plus its replayed semantic state. (+14 more)

### Community 70 - "Community 70"
Cohesion: 0.11
Nodes (12): MachineShopNoDeadline, NasaRoverNoDeadline, Stuck Car (1 object) variant with no deadline., Machine Shop variant with same goals and no deadline., Nasa Rover variant with identical goals and no deadline constraint., StuckCar1oNoDeadline, Place a rock under the car Action, Search a rock Action             the robot can find a one of the rocks (+4 more)

### Community 72 - "Community 72"
Cohesion: 0.11
Nodes (22): _as_fraction(), end_windows(), normalise(), Fraction, State, The first end fires; returns (windows of the rest, time charged).      Everyone, ``model`` is an ``EngineModel`` or ``ToyModel`` from ``temporal_stn_pdb``., Q ordered by remaining time, windows [rem, rem, rem].          ``running`` is `` (+14 more)

### Community 73 - "Community 73"
Cohesion: 0.17
Nodes (20): _add_mask(), best_joint_outcomes(), build_pattern(), _clamp01(), _conflict_table(), _env_int(), joint_outcomes(), _mask_of() (+12 more)

### Community 74 - "Community 74"
Cohesion: 0.13
Nodes (4): C_ANode, Adds to the SNode the possible actions as children.         If a specific child, Action node with consistency STN check, add constraints to the STN according to this `self` action         If this pare

### Community 75 - "Community 75"
Cohesion: 0.14
Nodes (7): AbstractProblem, This is an abstract class that represents a generic `planning problem`.      T, Returns the `Problem` `Environment`., Returns the `Problem` `name`., Sets the `Problem` `name`., Returns `True` the given `name` is already used inside this `Problem`,, Normalizes the given `Plan`, that is potentially the result of another

### Community 76 - "Community 76"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native AGENTS.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 77 - "Community 77"
Cohesion: 0.20
Nodes (8): PatternSolver, Memoised backward induction over boundary states of one pattern.      ``memo`` I, Canonicalise remaining-time multisets inside symmetry classes., Return true when starting ``op`` can only occupy its running slot., Next deadline-aligned duration-GCD boundary in remaining time., Optimise one same-time dispatch group without adding PDB rows., Advance to the next completion or required decision-grid boundary., Fold the joint outcome distribution of simultaneously ending ops.          Disti

### Community 78 - "Community 78"
Cohesion: 0.06
Nodes (16): DurativeAction, Fraction, Represents a durative action., Returns the `list` of the `Action` `preconditions`., Removes all the `Action preconditions`, Returns the `list` of the `Action effects`., Returns the `list` of the `Action effects`., Returns the `list` of the `Action effects`. (+8 more)

### Community 79 - "Community 79"
Cohesion: 0.19
Nodes (7): _extract_state_facts(), Any, Fact, Fact-level optimistic abstraction of an action., Propagate relaxed fact probabilities from ``state``.          If the fact-acti, Score a state with either product or minimum goal aggregation.          ``prod, RelaxedActionModel

### Community 80 - "Community 80"
Cohesion: 0.13
Nodes (13): _build_split_problem(), _chain_heuristic(), _ChainStubAdapter, _name(), Marginal table model for unit tests: each named add-fact lifts the value.     va, 2-step chain: A --a_to_b--> B --b_to_g--> G; goal G., The key regression: a chain-prefix action (a_to_b adds B, not the goal), SyntheticAction (+5 more)

### Community 81 - "Community 81"
Cohesion: 0.18
Nodes (4): LinkedList, updates the value of the list according to the lower_bound and upper_bound, Works only  if the rewards are not negative         update the max_value accord, Returns the value of the intervals between the lower and upper bound

### Community 82 - "Community 82"
Cohesion: 0.12
Nodes (7): Precondition, This class represent an precondition. It has a :class:`~unified_planning.model.F, check if the effect and the precondition are the same, Returns the `Fluent` of this `precondition`., Returns the `value` of the `Fluent` needed for the action execution., Sets the `value` needed to the `Precondition` of the `Fluent`.          :param, Returns this `Precondition's Environment`.

### Community 83 - "Community 83"
Cohesion: 0.33
Nodes (4): Attach (or clear) a pre-built PDB correction for the ``baseline_pdb``         st, Generate goal-directed patterns, build their PDBs, and attach the         result, Return the attached PDB correction, lazily auto-building one for the         ``b, PDBCorrection

### Community 84 - "Community 84"
Cohesion: 0.12
Nodes (9): DemoAction, LayerTrace, PropagationResult, Debug snapshot for one propagation layer., Output bundle returned by ``heuristic_propagate``., Build an engine ``State`` from problem initial values for PE evaluation., _reference_state_from_problem(), State (+1 more)

### Community 85 - "Community 85"
Cohesion: 0.05
Nodes (12): Any, Returns true iff the two initial states are equivalent., ObjectsSetMixin, Returns the object instance if the `problem` has `Object` with the given `name`,, Returns `True` if the `object` with the given `name` is in the `problem`,, Returns the `objects` compatible with the given `Type`: this includes the given, Returns the `list` containing all the `objects` in the `problem`., This class is a mixin that contains a `set` of `objects` with some related metho (+4 more)

### Community 86 - "Community 86"
Cohesion: 0.09
Nodes (31): gap_is_consistent(), Section 3's interval-node STN, restricted to what ``(F, Q)`` knows.      ``now <, pdb_for(), Tests for the first (upper-bound) version of ``critical_path_pdb``.  Anchors are, Section 11: a1 before a2 -> 1.0, a2 before a1 -> 0.0., a1 [0,3], a2 [2,4], a3 [3,4]: needs D >= 4., a3 needs both ends: T = max(d1 + d3, d2 + d3)., Section 8: a running action that must still finish adds r_a + tail_a. (+23 more)

### Community 87 - "Community 87"
Cohesion: 0.12
Nodes (26): in_execution(), The toy stand-in for the compilation's ``inExecution(start-<a>)`` fact., Explicit little model for the section-9 checks and unit tests.      Same inter, ToyModel, test_probability_mass_is_conserved_across_outcomes(), Tests for ILAO* over the time-left windows MDP (``windows_lao``)., a then b, D = 8: exact 0.84 (a succeeds by 2 or 4, then b gets 2 or 1 tries)., An unrelated c cannot help, so the true value stays 0.84. Charging lo per     en (+18 more)

### Community 88 - "Community 88"
Cohesion: 0.11
Nodes (23): ActionScoreEntry, is_active(), _action_name_key(), greedy_matched_value_target(), _heuristic_value(), _null_ctx, plan(), rank_actions_by_score() (+15 more)

### Community 91 - "Community 91"
Cohesion: 0.05
Nodes (34): ClosedDurationInterval(), Duration, DurationInterval, EndPreconditionTiming(), FixedDuration(), Interval, LeftOpenDurationInterval(), OpenDurationInterval() (+26 more)

### Community 92 - "Community 92"
Cohesion: 0.12
Nodes (7): This class represents a node of the `STNPlan`.      :param kind: The `Timepoin, Returns the end time according to the STN when the actions are performed in the, Legal interval for this node in the current plan., Returns the latest tine node can be executed according to the STN constraints, Adds the end action as a chosen action          - The end action must be before, add a deadline to the STN: end plan - start plan <= deadline         :param dea, STNPlanNode

### Community 93 - "Community 93"
Cohesion: 0.14
Nodes (8): Relaxed action mutex for parallel set construction (not admissible)., Memoized front-end for :meth:`_compute_path_actions_mutex` (incl. self)., SOUND action-mutex for the path-mutex layer, INCLUDING self-mutex.          Deli, Names of actions participating in >= 1 certified mutex pair (incl.         self-, Per action: the mutex PARTNERS (actions it certifiably conflicts with,         i, Build the structural context, per-action components and calibrator once., Build name->model and fact->deleters indices once per heuristic object., Action-mutex extended beyond pure Graphplan delete-interference.          Two ac

### Community 94 - "Community 94"
Cohesion: 0.16
Nodes (4): baseline_admissible_resolution: goal-directed backward pass over the     2^(k/2), baseline_admissible_resolution_forward: forward anchor-jump with a     per-block, TestAdmissibleResolutionForwardStrategy, TestAdmissibleResolutionStrategy

### Community 95 - "Community 95"
Cohesion: 0.21
Nodes (4): Focused tests for the atom_backtrack_exact_unbiased temporal heuristic strategy., Minimal action-like object compatible with the heuristic's action model builder., SyntheticAction, TestAtomBacktrackExactUnbiased

### Community 96 - "Community 96"
Cohesion: 0.22
Nodes (3): Init all actions into the new actions list         ensures the end actions can, Calculates the heuristic based on the current state and time, TRPG

### Community 98 - "Community 98"
Cohesion: 0.07
Nodes (27): Returns the `OperatorKind` that defines the semantic of this expression., OperatorKind, Enum, This module defines all the operators used by the unified_planning library., Enum representing the type of an :class:`~unified_planning.model.FNode`. The :fu, AnyChecker, This expression walker checks if any subexpression matches a given predicate., Checks if any of the subexpression matches the predicate.          :param expr (+19 more)

### Community 99 - "Community 99"
Cohesion: 0.19
Nodes (17): main(), Quick check: fixed-tail prefix-frac bootstrap and MCTS backups are sensible., _aggregation_for_strategy(), _clock_time(), fixed_tail_bootstrap_value(), fixed_tail_config_from_args(), fixed_tail_dead_end_value(), _goal_reached() (+9 more)

### Community 101 - "Community 101"
Cohesion: 0.10
Nodes (23): Exception, SyntaxError, ANMLSyntaxError, Base class for all custom exceptions of the unified_planning (UP) library., UPConflictingPreconditionException, UPException, UPExpressionDefinitionError, UPNoRequestedEngineAvailableException (+15 more)

### Community 103 - "Community 103"
Cohesion: 0.15
Nodes (12): _action_name(), _action_positive_duration(), FixedTailRandomRolloutConfig, FixedTailRandomRolloutEvaluator, pick_rollout_action(), Random, random_rollout_config_from_args(), Ephemeral fixed-tail prefix rollouts for MCTS leaf evaluation (Option A).  Fro (+4 more)

### Community 104 - "Community 104"
Cohesion: 0.13
Nodes (23): cut_and_bound(), cut_components(), cut_fact_keys(), cut_retry_fold(), cut_truncate(), CutRow, cutrow_mutex(), cutrow_or_merge_into() (+15 more)

### Community 105 - "Community 105"
Cohesion: 0.18
Nodes (7): UPUnreachableCodeError, Dnf, Nnf, Class used to transform a logic expression into the equivalent     Disjunctive, Function used to transform a logic expression into the equivalent         Disju, Class used to transform a logic expression into the equivalent     Negation Nor, Function used to transform a logic expression into the equivalent         Negat

### Community 106 - "Community 106"
Cohesion: 0.18
Nodes (7): build_pdb_actions(), Tests for the horizon-indexed Pattern Database (PDB) correction prototype.  Incl, Test 3: stochastic robot (0.8 advance, 0.2 stay)., SyntheticAction, SyntheticProbabilisticEffect, TestAdapterJointOutcomes, TestStochasticRobot

### Community 107 - "Community 107"
Cohesion: 0.29
Nodes (10): advance_to_elapsed(), build_mdp(), goal_product_by_layer(), main(), Standalone probe: inspect the PTRPG (baseline_survival) layer-by-layer propagati, G_t = prod_g P_t(g) for every layer t., Step (random legal) until current_time advances to >= target_elapsed     (commit, t* = first layer where G_t crosses theta*g_inf (horizon-invariant signal). (+2 more)

### Community 109 - "Community 109"
Cohesion: 0.08
Nodes (14): returned_action_name(), C_MCTS, TP MCTS solver implementation.     Contains STNs in each node, Per-action goal-backtrack marginal lift from this node's state, cached, Global open/expandable leaf nodes across the tree (spanning depths)., aligned_value(n): prefix-roll delta then PTRPG at the GLOBAL H_frontier, Standard backprop of a freshly expanded value up to the root., Expand ONE child of the selected ORIGINAL node, standard backprop.         The (+6 more)

### Community 110 - "Community 110"
Cohesion: 0.08
Nodes (22): main(), Batch greedy_parallel runtime: baseline vs resolution backward/forward (alpha=2), build_mdp(), parse_args(), Any, Namespace, Heuristic Per-Call Runtime Benchmark ======================================  Mea, Build and compile an MDP from scratch (fresh caches on each call). (+14 more)

### Community 111 - "Community 111"
Cohesion: 0.16
Nodes (10): PatternDatabase, Horizon-indexed PDB for one pattern.      The DP uses ``max`` over *all* project, Test 4: door-chain pattern growth and per-pattern V values., move_i: pre {at_i (, battery_high)}, add at_{i+1}, duration 1, p=1., Test 1: deterministic robot., Test 2: ignored precondition (optimistic projection)., _robot_chain(), TestDeterministicRobot (+2 more)

### Community 112 - "Expression"
Cohesion: 0.25
Nodes (6): compute_precondition_support(), ProbabilisticOptimisticRPGHeuristic, Return only the relaxed support probability ``R_t(a)`` for a precondition., Optimistic probabilistic RPG-style heuristic with monotone retry updates., SyntheticAction, TestProbabilisticOptimisticRPG

### Community 113 - "Community 113"
Cohesion: 0.12
Nodes (15): Cost (trial $300), Files, Machine types, One-time: create the VM, Option A — Cursor / VS Code Remote SSH (recommended), Option B — Jupyter in browser via SSH tunnel, Option C — Headless (no notebook), Prerequisites (+7 more)

### Community 114 - "Community 114"
Cohesion: 0.06
Nodes (13): The flag ``invalidate_memoization`` can be used to clear the cache         afte, LinearChecker, Checks if the given expression is linear or not and returns the `set` of the `fl, Returns the tuple containing a flag saying if the expression is linear or not,, QuantifierSimplifier, Same to the :class:`~unified_planning.model.walkers.Simplifier`, but does not ex, Simplifies the expression and the quantifiers in it.         The quantifiers ar, Add children to the stack. (+5 more)

### Community 115 - "Community 115"
Cohesion: 0.14
Nodes (20): Protocol, _action_name(), _apply_action_set_sampled(), _build_action_set(), HeuristicAdapter, MaxApproximationDebug, _print_max_approximation_debug(), Random (+12 more)

### Community 116 - "Community 116"
Cohesion: 0.24
Nodes (7): ExactPatternMDPHeuristic, _extract_state_facts(), Admissible upper bound from exact small-pattern CoMDP+ MDPs.      Usage mirrors, Pair snap actions back into durative operations.          ``convert_problem`` sp, Admissible upper bound on P(goal before deadline) from ``state``.          Goals, test_from_problem_keeps_only_true_initial_facts(), Fact

### Community 119 - "Action"
Cohesion: 0.09
Nodes (15): replace_action(), ActionInstance, Plan, PlanKind, Enum, Return this `plan's` `Environment`., Returns the `Plan` `kind`, This function takes a function from `ActionInstance` to `ActionInstance` and ret (+7 more)

### Community 121 - "main"
Cohesion: 0.15
Nodes (19): Any, parse_run_metrics(), print_summary_table(), Shared utilities for experiment scripts.  Provides: - Heuristic alias mapping, Write a list of dicts to a CSV file, auto-detecting columns if not given., Print a simple fixed-width table to stdout., Launch run_domain.py as a subprocess and return (stdout_text, returncode)., Print high-signal run_domain.py lines (for Script 3 when verbose=False). (+11 more)

### Community 122 - "Community 122"
Cohesion: 0.18
Nodes (5): check_fix_time(), Add constrains to the `stn` according to the `action` and the `previous_action`, Checks if the time of the action execution needs to be fixed     action time ne, update_stn(), TestSTN

### Community 123 - "Community 123"
Cohesion: 0.17
Nodes (11): Algorithm Concepts To Preserve, Codex Project Instructions, Coding Style, Commands, GCP experiments (64-core CPU VM), Heuristic Bias Checks, max_approximation selector (standalone), Project Context (+3 more)

### Community 126 - "Community 126"
Cohesion: 0.47
Nodes (5): _cell_style(), main(), Build docs/experiment_results.xlsx from a reproducible table layout., Update only left/right border sides and keep existing top/bottom., _set_vertical_border()

### Community 127 - "Community 127"
Cohesion: 0.47
Nodes (5): build_mdp(), main(), MDP, One-off: time a SINGLE heuristic call (no MCTS) for baseline_admissible (dense), time_once()

### Community 128 - "Community 128"
Cohesion: 0.18
Nodes (11): flatten_dict_structure(), Fraction, This method takes a dict containing a List of tuples of 3 elements, and     ret, Constructs the `STNPlan` with 2 different possible representations:         one, add constraint so the time of the action is fixed and can't be changed, The end action is not yet to be chosen, the goal might achived before the end ac, Fraction, Return a `real` constant.      :param value: The `Fraction` that must be promo (+3 more)

### Community 129 - "shortcuts.py"
Cohesion: 0.04
Nodes (75): # TODO: changed to be not probabilistic effect, get_environment(), Returns the given environment if it is not `None`, returns the `GLOBAL_ENVIRONME, Always(), And(), AtMostOnce(), Bool(), Compiler() (+67 more)

### Community 130 - "Community 130"
Cohesion: 0.03
Nodes (43): UPProblemDefinitionError, UPValueError, ActionsSetMixin, Returns the action instance if the `problem` has the `action` with the given `na, Adds the given `action` to the `problem`.          :param action: The `action`, Adds the given `actions` to the `problem`.          :param actions: The `list`, This class is a mixin that contains a `set` of `actions` with some related metho, Returns the `Problem` environment. (+35 more)

### Community 131 - "Community 131"
Cohesion: 0.22
Nodes (8): PDBOutcome, One joint outcome of an action: fires with ``probability``., Concurrent durative semantics: independent durative actions overlap     instead, Reference: the OLD sequential DP (one action per recursion, charging its     ful, Concurrency only ADDS parallelism/retries, so the new DP must never score     be, _sequential_value(), TestConcurrentDurations, TestNoValueDecrease

### Community 132 - "Community 132"
Cohesion: 0.17
Nodes (11): 1. Heuristics added this session (strategy aliases), 2. Key files, 3. The three alignment prompts (genealogy — caused real naming confusion), 4. ROLLOUT_ALIGNED_H is INERT for dynamic + frontier (fixed this session), 5. Option A bug + audit (the crux of the last part), 6. What I did on Option A before the user took over, 7. CURRENT STATE — needs reconciliation (two implementations coexist), 8. Gotchas / environment (+3 more)

### Community 133 - "Community 133"
Cohesion: 0.09
Nodes (4): Fraction, Performs basic simplifications of the input expression.      Important NOTE:, Performs basic simplification of the given expression.          If a :class:`~, Simplifier

### Community 135 - "Community 135"
Cohesion: 0.32
Nodes (13): _clamp_probability(), _compute_clause_support(), _compute_precondition_support_result(), _format_clause(), _format_precondition_structure(), _is_atomic_fact(), _is_clause_container(), _is_dnf_structure() (+5 more)

### Community 136 - "Community 136"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 138 - "Community 138"
Cohesion: 0.16
Nodes (14): Bound, bound_add(), bound_is_negative(), bound_min(), bound_repr(), bound_tighter(), Strict-bound rational DBM for the symbolic-STN temporal PDB.  Part I of ``artifa, Restore closure after tightening edge ``(i, j)``. O(N^2). (+6 more)

### Community 141 - "Community 141"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 143 - "Community 143"
Cohesion: 0.33
Nodes (5): CoMDP+ No-Deadline Starter, Included starter scenarios, Run greedy baseline, Run smoke benchmarks (all 5 presets), Run tests for this starter package

### Community 152 - "Path"
Cohesion: 0.13
Nodes (9): Fraction, Add an unconstrained time point (every relation to it is +inf)., Assert ``x - y <= value`` (or ``<`` when strict); returns consistency., ``x < y + value``. The spec's strict order, with no epsilon., ``x - y == value``, as the two opposite bounds., Pin ``x`` to an absolute time (the spec's query conditioning I = x)., Existentially eliminate every variable outside ``keep``.          Because the ma, Hashable canonical form of the closed matrix, for merge keys.          Two close (+1 more)

### Community 153 - "plan.py"
Cohesion: 0.16
Nodes (9): Owns the rollout-aligned evaluation, its budget, cache and diagnostics., Reset the per-search rollout/time budget (call at each MCTS decision)., Aligned value of a node at remaining horizon R.          ``h_override`` is the p, Pick the comparison horizon H. Returns (H, raw_fallback_flag).          Invarian, RolloutAlignedEvaluator, FakeState, A elapsed=0/R=25, B elapsed=10/R=15, H_frontier=15., TestOptionAConfig (+1 more)

### Community 154 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 155 - "graphify reference: commit hook and native AGENTS.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native AGENTS.md integration, graphify reference: commit hook and native AGENTS.md integration

### Community 156 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 158 - "Thesis Ideation Prompts (Optional)"
Cohesion: 0.50
Nodes (3): Candidate Experiment Directions, Suggestion Style, Thesis Ideation Prompts (Optional)

### Community 159 - "TP-MCTS (Temporal Planning Monte Carlo Tree Search)"
Cohesion: 0.50
Nodes (3): domains, Quick Start, TP-MCTS (Temporal Planning Monte Carlo Tree Search)

### Community 160 - ".heuristic_expected_time"
Cohesion: 0.21
Nodes (4): Difference-bound matrix over named time points, with strict bounds.      ``m[i][, StrictDBM, TestClosureAndProjection, TestStrictCycles

### Community 161 - ".is_int_constant"
Cohesion: 0.31
Nodes (3): PDBCorrection, Holds a set of pattern databases and answers applicability queries.      For an, TestPDBCorrectionManager

### Community 162 - "FreeVarsExtractor"
Cohesion: 0.16
Nodes (10): _escape_house_actions(), _heuristic_for(), The clamp is min(marginal, pattern), so it can only tighten., The marginal bound pins to 1.0; the pattern keeps discriminating., The RPG sweep is redundant once every goal fact carries a pattern., _SyntheticAction, _SyntheticProbabilisticEffect, test_pure_strategy_matches_the_rpg_wrapped_one() (+2 more)

### Community 166 - "collect_global_frontier"
Cohesion: 0.23
Nodes (3): Prob_MatchCellar, Returns the user type defined in the global environment with the given `name` an, UserType()

### Community 167 - ".kind"
Cohesion: 0.32
Nodes (5): dominance_prune(), dominates(), Future-proof, single-alternative dominance: ``a`` makes ``b`` redundant     fore, Drop every row dominated by a sibling — keep the Pareto frontier over     (prob,, TestDominance

### Community 168 - ".is_global"
Cohesion: 0.33
Nodes (8): build_pdb_action(), _clamp_probability(), _extract_duration(), _outcome_assignment(), _parse_probabilistic_effect(), Horizon-indexed Pattern Database (PDB) correction for the probabilistic temporal, One probabilistic effect -> list of (prob, add, del) outcomes.      A residual n, Convert one action object into a :class:`PDBAction` (None if it has no     relax

### Community 169 - "presets.py"
Cohesion: 0.19
Nodes (10): at_or_past_tail_horizon(), crossed_cutoff(), elapsed_from_root(), _emit_fixed_tail_debug(), fixed_tail_ptrpg_horizon(), FixedTailSearchContext, True when the node is in the tail zone (no further MCTS expansion)., Horizon passed to PTRPG at tail evaluation (fixed for the search). (+2 more)

### Community 170 - ".check_stn"
Cohesion: 0.23
Nodes (3): build_fixed_tail_search_context(), FixedTailConfig, TestFixedTailPrefixFrac

### Community 172 - ".create_Snode"
Cohesion: 0.23
Nodes (8): _clamp01(), insert_path(), Enhanced OR insert: same 3-case skeleton as :func:`insert_or_absorb`, but, Drop-in for :func:`table_or_hazard` using the enhanced :func:`insert_path`     (, Fold a layer's arriving aux rows into the cumulative table with RETRY     semant, retry_fold_arrivals(), table_or_hazard_paths(), TestMutexPreservingSum

### Community 173 - ".And"
Cohesion: 0.23
Nodes (7): _build_split_problem(), Regression tests for baseline_cached cache propagation in MCTS (plan / combinati, baseline_cached must not falsely succeed when deadline is too tight., baseline_cached in plan() (C_MCTS) must succeed when deadline allows., Classical trpg path must still work after the cache wiring changes., Non-cached baseline strategy must still work (no cache threading)., TestMCTSBaselineCached

### Community 174 - "Pattern"
Cohesion: 0.29
Nodes (3): Pattern, Projected behaviour used to identify exchangeable ground actions., Find projected action identities that can be safely permuted.          Members m

### Community 176 - "TestTableStrategyEngine"
Cohesion: 0.25
Nodes (5): Rollout-aligned common-horizon PTRPG evaluation.  Fixes the cross-depth / cross-, RolloutAlignedDiagnostics, PrefixRolloutFn, RawEvalFn, StateHashFn

### Community 177 - "TestChainedFootprints"
Cohesion: 0.33
Nodes (4): _FakeAction, _FakeANode, _FakeSNode, TestMCTSUctFiltering

### Community 178 - "AnyChecker"
Cohesion: 0.31
Nodes (4): Section 8: D = 3, d(a1) = 2, d(a2) = 3 => 0 <= S1 <= 1 and 0 <= S2 <= 0., Spec section 2: introduce S_a, E_a with E_a - S_a = d(a), C <= S_a, E_a <= D., _start_action(), TestSpecSection8Windows

### Community 179 - "compare_paths_table_vs_admissible.py"
Cohesion: 0.33
Nodes (4): frontier_score(), Blended frontier-selection score (Option A / frontier_aligned_*).          front, Option A frontier-aligned selection score (frontier_aligned_*)., TestFrontierScore

### Community 180 - ".is_int_constant"
Cohesion: 0.31
Nodes (3): Running actions from the compiled model's inExecution(start-<a>) facts., One ILAO* table per pattern, solved offline, read (and lazily extended) online., WindowsILAOPDBHeuristic

### Community 181 - "sweep_paths_table_gap.py"
Cohesion: 0.60
Nodes (4): build_mdp(), main(), Per-call runtime: baseline_admissible vs baseline_admissible_paths_table (v3)., time_strategy()

### Community 184 - "Critical path PDB runtime investigation"
Cohesion: 0.25
Nodes (7): Critical path PDB runtime investigation, Irrelevant actions and patterns, Reproduction files, Size of the tested problem, Unprofiled timings, Validation and scope, Where the time goes

### Community 185 - "._frontier_aligned_value"
Cohesion: 0.25
Nodes (5): _dynamic_aligned_horizon(), Parent-local comparison horizon H_p = min over the parent's children of     the, Frontier-aligned value of a node, used ONLY for selection — never         backp, The STN attached to a search node, if this run keeps one., _stn_of()

### Community 190 - ".interval"
Cohesion: 0.29
Nodes (3): Tightest implied bound on ``x - y`` (both must be declared)., ``(lower, upper)`` on ``x`` relative to the origin, as raw bounds.          ``lo, ``(lo, lo_strict, hi, hi_strict)`` for ``x`` in origin-relative time.

### Community 192 - "cumulative_merge_truncate"
Cohesion: 0.47
Nodes (3): cumulative_merge_truncate(), Sum-PRESERVING truncation for a cumulative (free-summed) table.      Two passes,, TestCumulativeMergeTruncate

## Knowledge Gaps
- **93 isolated node(s):** `NASA rover with nine goal-specific PDBs`, `NASA rover critical path PDB timings`, `Size of the tested problem`, `Unprofiled timings`, `Where the time goes` (+88 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **40 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `TemporalProbabilisticRPGHeuristic` (6× useful, score=4.949501467)
- `DeltaSimpleTemporalNetwork` (5× useful, score=3.835341226)
- `CombinationState` (4× useful, score=2.795354685)
- `ActionQueue` (3× useful, score=2.09653585)
- `State` (2× useful, score=1.398002391)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TemporalProbabilisticRPGHeuristic` connect `Community 10` to `Community 2`, `Community 131`, `Community 4`, `Community 3`, `Community 13`, `Community 21`, `Community 22`, `Community 23`, `Community 27`, `Community 33`, `.is_int_constant`, `FreeVarsExtractor`, `Community 36`, `Community 37`, `.kind`, `Community 39`, `presets.py`, `Community 42`, `.copy_stn`, `.create_Snode`, `.check_stn`, `compare_paths_table_vs_admissible.py`, `Community 51`, `Community 53`, `sweep_paths_table_gap.py`, `._ensure_admissible_lp_bound`, `Community 62`, `cumulative_merge_truncate`, `TestTableStrategyEngine`, `TestChainedFootprints`, `Community 67`, `Community 68`, `Community 80`, `Community 83`, `Community 93`, `Community 94`, `Community 95`, `Community 99`, `Community 104`, `Community 106`, `Community 107`, `Community 111`, `Community 115`, `Community 127`?**
  _High betweenness centrality (0.251) - this node is a cross-community bridge._
- **Why does `FNode` connect `Community 1` to `Community 128`, `shortcuts.py`, `Community 98`, `Community 99`, `Community 35`, `Community 133`, `Community 38`, `Community 59`, `Community 105`, `Community 44`, `Community 17`, `Community 18`, `Community 114`, `Community 20`, `Community 91`, `Community 28`, `Community 29`, `Community 63`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Why does `Environment` connect `Community 17` to `shortcuts.py`, `Community 1`, `Community 3`, `Community 15`, `Community 24`, `Community 26`, `DurativeAction`, `Community 34`, `Community 35`, `Community 43`, `Community 59`, `Community 63`, `Community 64`, `Community 78`, `Community 91`, `Community 92`, `Community 98`, `Community 114`, `Action`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 33 inferred relationships involving `FNode` (e.g. with `create_action_with_given_subs()` and `Environment`) actually correct?**
  _`FNode` has 33 INFERRED edges - model-reasoned connections that need verification._
- **Are the 81 inferred relationships involving `TemporalProbabilisticRPGHeuristic` (e.g. with `PlanResult` and `SyntheticAction`) actually correct?**
  _`TemporalProbabilisticRPGHeuristic` has 81 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `C_MCTS` (e.g. with `TemporalSTNPDBHeuristic` and `WindowsILAOPDBHeuristic`) actually correct?**
  _`C_MCTS` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `Environment` (e.g. with `Action` and `CombinationAction`) actually correct?**
  _`Environment` has 32 INFERRED edges - model-reasoned connections that need verification._