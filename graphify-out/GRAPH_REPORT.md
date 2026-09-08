# Graph Report - TP_MCTS  (2026-08-24)

## Corpus Check
- 195 files · ~200,633 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4141 nodes · 8996 edges · 189 communities (150 shown, 39 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 860 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `63ac97da`
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
- .check_stn
- .copy_stn
- .create_Snode
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
- .calculate_shortest_path
- .check_stn
- Action
- .copy_stn
- Fact

## God Nodes (most connected - your core abstractions)
1. `FNode` - 326 edges
2. `TemporalProbabilisticRPGHeuristic` - 258 edges
3. `C_MCTS` - 113 edges
4. `Environment` - 81 edges
5. `get_environment()` - 60 edges
6. `MDP` - 57 edges
7. `Row` - 55 edges
8. `Segment` - 54 edges
9. `UPProblemDefinitionError` - 53 edges
10. `UPTypeError` - 50 edges

## Surprising Connections (you probably didn't know these)
- `Base_MCTS` --uses--> `ExactPatternMDPHeuristic`  [INFERRED]
  unified_planning/engines/solvers/mcts.py → comdp_plus_no_deadline/engines/exact_pattern_mdp.py
- `C_MCTS` --uses--> `ExactPatternMDPHeuristic`  [INFERRED]
  unified_planning/engines/solvers/mcts.py → comdp_plus_no_deadline/engines/exact_pattern_mdp.py
- `MCTS` --uses--> `ExactPatternMDPHeuristic`  [INFERRED]
  unified_planning/engines/solvers/mcts.py → comdp_plus_no_deadline/engines/exact_pattern_mdp.py
- `FixedTailConfig` --uses--> `TemporalProbabilisticRPGHeuristic`  [INFERRED]
  unified_planning/engines/solvers/fixed_tail_ptrpg_rollout.py → comdp_plus_no_deadline/engines/temporal_probabilistic_rpg.py
- `FixedTailSearchContext` --uses--> `TemporalProbabilisticRPGHeuristic`  [INFERRED]
  unified_planning/engines/solvers/fixed_tail_ptrpg_rollout.py → comdp_plus_no_deadline/engines/temporal_probabilistic_rpg.py

## Import Cycles
- None detected.

## Communities (189 total, 39 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (28): OutcomeDetail, _feasible_actions(), _fit_action_stn(), FixedTailExpectimaxEvaluator, Expectimax prefix evaluation for fixed-tail MCTS.  V(s) = max_a Q(s,a) over ST, Stop expanding expectimax when time budget or step depth is reached., V(s) using only STN-feasible actions (MCTS children), not all MDP-legal actions., STN-feasible legal actions (same filter as greedy_parallel / MCTS children). (+20 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (15): FNodeContent, FNode, object, Returns the `id` of this expression., Returns the `OperatorKind` that defines the semantic of this expression., Returns the `Environment` in which this expression exists., Returns the `Type` of this expression., Returns all the names contained in this expression. (+7 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (67): best_joint_add_distribution(), build_pattern(), build_patterns(), _clamp01(), _collapse_ages(), compute_earliest_times(), compute_gate(), conditional_hazards() (+59 more)

### Community 3 - "Community 3"
Cohesion: 0.16
Nodes (7): _chain_heuristic(), _ChainStubAdapter, Marginal table model for unit tests: each named add-fact lifts the value.     va, 2-step chain: A --a_to_b--> B --b_to_g--> G; goal G., The key regression: a chain-prefix action (a_to_b adds B, not the goal), SyntheticAction, TestActionContributionScoring

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (35): CachedPTRPGTable, _clamp_probability(), _extract_state_facts(), Fact, Debug snapshot for one temporal layer., Non-negative per-action scores from forward-layer precondition support, Output bundle for the duration-aware heuristic., Lower-bound estimate on P(all goals by deadline) using correlation-aware DP. (+27 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (15): DurativeAction, Fraction, Represents a durative action., Returns the `list` of the `Action` `preconditions`., Removes all the `Action preconditions`, Returns the `list` of the `Action effects`., Returns the `list` of the `Action effects`., Returns the `list` of the `Action effects`. (+7 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (42): ExpressionManager, BoolExpression, Expression, Fraction, object, Creates the unified_planning expressions if it hasn't been created yet in the en, Returns a conjunction of terms.         This function has polymorphic n-argumen, Returns an disjunction of terms.         This function has polymorphic n-argume (+34 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (9): Fraction, Returns this type lower bound., Returns this type upper bound., Class that manages the :class:`Types <unified_planning.model.Type>` in the :clas, Returns this `Environment's` boolean `Type`., Returns the `integer type` defined in this :class:`~unified_planning.Environment, Returns the `real type` defined in this :class:`~unified_planning.Environment` w, Returns the user type defined in this :class:`~unified_planning.Environment` wit (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (5): This method takes the args given as parameters to a walker method (walk_and, This walker takes the mapping from the usertype fluents to be removed from, Removes UserType Fluents from the given expression and returns the generated, Removes the UsertypeFluents from an Expression and returns the equivalent condit, UsertypeFluentsWalker

### Community 9 - "Community 9"
Cohesion: 0.06
Nodes (35): AndGammaCalibrator, AndGammaConfig, build_candidate_pairs(), build_components(), build_structural_context(), _clamp01(), classify_component(), ComponentInfo (+27 more)

### Community 10 - "Community 10"
Cohesion: 0.08
Nodes (16): DP-relevant add facts of an action, keyed by action name.          Returns the, Cache telemetry for ``survivor_pdb_lazy`` (hits, misses, sweeps)., Return (and optionally print) the headline mutex-survival metric.          Acc, Duration-aware optimistic relaxed heuristic with fixed temporal depth.      Co, Return (and optionally reset) the path-mutex survival / AND-feasibility, Product of component gammas for an action's preconditions (≥ 0)., Memoized front-end for :meth:`_compute_kmutex_actions_are_mutex`.          The, EXECUTION mutex for the K-bounded OR-layer max-collapse.          Deliberately (+8 more)

### Community 11 - "Community 11"
Cohesion: 0.27
Nodes (4): Create a new Snode for the state `state` with parent `parent`          In this, Traverse the tree until reaching a leaf node., Traverse the tree until reaching a leaf node.         Selection with max logic, Traverse the tree until reaching a leaf node.         Selection with root inter

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (8): Base_MCTS, plan(), Global frontier Option A: argmax aligned_value, expand selected node only., Choose a random action. Heustics can be used here to improve simulations., :param root_node: the root node of the MCTS tree         :return: returns the b, Return the most-visited child (robust child / argmax-N)., Execute the MCTS algorithm from the initial state given, with timeout in seconds, Simulate until a terminal state

### Community 13 - "Community 13"
Cohesion: 0.06
Nodes (30): Achiever, _clamp01(), _fact_sort_key(), _has_shared_fact(), marginal_consistent_or_hazard(), MarginalConsistentORBound, _PreparedFormula, Fact (+22 more)

### Community 14 - "Community 14"
Cohesion: 0.08
Nodes (50): alts_and(), alts_or(), _cap_groups(), _clamp01(), _coalesce_union(), cut_and_bound(), cut_components(), cut_emit_rows() (+42 more)

### Community 15 - "Community 15"
Cohesion: 0.04
Nodes (47): Takes in input an `Action` and returns the iterator over all the possible parame, get_environment(), Returns the given environment if it is not `None`, returns the `GLOBAL_ENVIRONME, get_all_fluent_exp(), get_ith_fluent_exp(), Returns the ith ground fluent expression., OperatorKind, Enum (+39 more)

### Community 16 - "Community 16"
Cohesion: 0.11
Nodes (15): _achievers_share_fact(), build_resolution_delta_schedule(), _grid_ceil(), Piece widths Δ_k that partition ``remaining`` (sum = ``remaining``).      Laye, Partition ``depth`` into resolution layer widths (see ``build_resolution_delta_s, Cumulative time anchors [0, …, depth] after largest-to-smallest delta reorganiza, Smallest anchor in ``anchors_asc`` that is >= ``t`` (clamped to the grid)., Anchor completion times in [first_completion, horizon], ascending. (+7 more)

### Community 17 - "Community 17"
Cohesion: 0.04
Nodes (21): Returns this `Action` `Environment`., Environment, IO, Returns the environment's `TypeChecker`., Returns the environment's `Factory`., Returns the environment's `Simplifier`., Returns the environment's `Substituter`., Returns the environment's `FreeVarsExtractor`. (+13 more)

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (7): _apply_function_to_effect(), Effect, This class represent an effect. It has a :class:`~unified_planning.model.Fluent`, Returns the `Fluent` that is modified by this `Effect`., Returns the `value` given to the `Fluent` by this `Effect`., Sets the `value` given to the `Fluent` by this `Effect`.          :param new_v, Returns this `Effect's Environment`.

### Community 19 - "Community 19"
Cohesion: 0.19
Nodes (13): pick_best_action(), _log_rollout_step(), pick_greedy_rollout_action(), ptrpg_guided_terminal_rollout(), PTRPG-guided terminal rollout for MCTS leaf evaluation.  Uses the same greedy, remaining_deadline(), resolve_rollout_policy(), rollout_config_from_args() (+5 more)

### Community 20 - "Community 20"
Cohesion: 0.03
Nodes (33): Return the given subexpression at the given position.          :param idx: The, Return the `Fluent` stored in this expression., Return the `Parameter` stored in this expression., Return the variable of the VariableExp., Return the `Variables` of the `Exists` or `Forall`., Return the `Object` stored in this expression., Return the `Timing` stored in this expression., Return the `Agent` stored in this expression. (+25 more)

### Community 21 - "Community 21"
Cohesion: 0.10
Nodes (23): combine_precondition_footprints(), PathMutexInstrumentation, prune_expired(), True iff the half-open windows ``[start, end)`` intersect.      Touching endpoin, Per-layer OR-hazard ``H_t(f)`` via the K-bounded :func:`insert_or_absorb`     ta, Drop registered segments that can no longer overlap anything new.      A segment, Best-effort UNION of one representative recent footprint per precondition     —, Accumulates the per-layer OR-hazard table HIT metrics. (+15 more)

### Community 22 - "Community 22"
Cohesion: 0.10
Nodes (27): BaseCombinationMDP, BaseMDP, evaluation_loop(), combination_greedy_plan(), _effective_temporal_depth(), _get_probabilistic_rpg_heuristic(), _get_temporal_probabilistic_rpg_heuristic(), PlanResult (+19 more)

### Community 23 - "Community 23"
Cohesion: 0.15
Nodes (11): _TprpgHeuristicAdapter, _aggregation_for_strategy(), _effective_temporal_depth(), _get_rollout_aligned_evaluator(), Build a RolloutAlignedConfig from unified_planning.parser CLI args., Build (once per MDP+suffix) a RolloutAlignedEvaluator bound to this MDP., Optional kwargs for atom_backtrack_exact_resolution (from unified_planning.parse, Pick the goal-aggregation for heuristic_score based on the strategy.      `bas (+3 more)

### Community 24 - "Community 24"
Cohesion: 0.05
Nodes (7): Fluent, Returns the `Fluent` `Type`., Returns the `Fluent` `signature`.         The `signature` is the `List` of `Par, Returns the `Fluent` arity.          IMPORTANT NOTE: this property does some c, Returns the `Fluent` `Environment`., Returns a fluent expression with the given parameters.          :param args: T, Returns the `Fluent` `name`.

### Community 25 - "Community 25"
Cohesion: 0.09
Nodes (14): Convert_problem, convert instantaneous actions from `model` actions to be `engines` actions, Finding mutex actions and adding a precondition that they can't be executed in p, Check if two actions are mutex          :param action: The checked action, Check if two actions are soft mutex          :param action: The checked action, returns all the negative end assignments of durative `action` to fluents in, returns all the negative start assignments of `action` to fluents in         if, returns all the positive start assignments of `action` to fluents         if du (+6 more)

### Community 26 - "Community 26"
Cohesion: 0.03
Nodes (58): Exception, SyntaxError, TypeError, Action, CombinationAction, DurativeAction, implAction, InstantaneousAction (+50 more)

### Community 27 - "Community 27"
Cohesion: 0.19
Nodes (9): build_mdp(), main(), Single-call comparison: baseline_admissible_survivor_pdb vs baseline_admissible., build_converted_problem(), main(), Probe whether the mutex-aware K-bounded OR layer (baseline_admissible_kmutex) ca, build_mdp(), main() (+1 more)

### Community 28 - "Community 28"
Cohesion: 0.17
Nodes (10): CompilationKind, CompilerMixin, Enum, Sets the default compilation kind.          :default: The default compilation, :param compilation_kind: The tested `CompilationKind`.         :return: True if, Method called by :func:`~unified_planning.engines.mixins.CompilerMixin.compile`, Enum representing the available compilation kinds currently in the library., Generic class for a compiler defining it's interface. (+2 more)

### Community 29 - "Community 29"
Cohesion: 0.04
Nodes (39): DurativeAction, InstantaneousAction, check_and_simplify_conditions(), check_and_simplify_preconditions(), create_action_with_given_subs(), create_effect_with_given_subs(), create_precondition_with_given_subs(), create_probabilistic_effect_with_given_subs() (+31 more)

### Community 30 - "Community 30"
Cohesion: 0.17
Nodes (4): C_MCTS, Per-action goal-backtrack marginal lift from this node's state, cached, Max over k sampled actions; each child gets fixed-tail leaf eval (K rollouts ins, TP MCTS solver implementation.     Contains STNs in each node

### Community 31 - "Community 31"
Cohesion: 0.12
Nodes (32): apply_heuristic_alias_overrides(), best_action_name(), build_mcts(), build_mdp(), configure_fixed_tail_cli(), configure_max_approx_cli(), configure_ptrpg_rollout_cli(), configure_rollout_aligned_cli() (+24 more)

### Community 32 - "Community 32"
Cohesion: 0.08
Nodes (40): aligned_value_for_node(), build_option_a_evaluator(), compute_H_frontier(), format_option_a_debug_row(), is_option_a_strategy(), option_a_config_from_cli(), option_a_ptrpg_suffix(), OptionAConfig (+32 more)

### Community 33 - "Community 33"
Cohesion: 0.13
Nodes (14): and_cumulative_bound(), and_has_mutex(), and_support_kernelized(), AndKernelResult, _fact_max(), The gate: is there ANY cross-fact certified mutex? No -> the whole     kerneliza, Full AND pipeline: gate -> components -> exact per component -> min.      Return, Admissible AND bound for cumulative (free-summed) per-fact tables — see the (+6 more)

### Community 34 - "Community 34"
Cohesion: 0.07
Nodes (5): Parameter, Represents an :func:`action parameter <unified_planning.model.Action.parameters>, Returns the `Parameter` `name`., Returns the `Parameter` `type`., Return the `Parameter` `Environment`

### Community 35 - "Community 35"
Cohesion: 0.07
Nodes (5): Represents a variable; a `Variable` has a name and a type., Returns the `Variable` name., Returns the `Variable` `Type`., Return the `Variable` `Environment`., Variable

### Community 36 - "Community 36"
Cohesion: 0.09
Nodes (22): admissible_and_support(), _clamp_probability(), cumulative_retry_update(), propagate_admissible_temporal_rpg(), Fact, Probabilistic Temporal RPG Heuristic — admissible upper-bound version.  Heuristi, Clamp a probability into ``[0, 1]`` (doc Section 9 domain guard)., AND layer (doc 5.1): admissible upper bound on the joint probability that all (+14 more)

### Community 37 - "Community 37"
Cohesion: 0.10
Nodes (28): _accumulate_alternatives(), cumulative_merge_truncate(), dominance_prune(), dominates(), exact_component_value(), guaranteed_mutex(), insert_path(), _merge_alternatives() (+20 more)

### Community 38 - "Community 38"
Cohesion: 0.11
Nodes (12): DagWalker, Returns True, independently from the children's value., Returns False, independently from the children's value., Returns None, independently from the children's value., Returns expression, independently from the childrens's value., Returns True if any of the children returned True., Returns True if all the children returned True., DagWalker treats the expression as a DAG and performs memoization of the     in (+4 more)

### Community 39 - "Community 39"
Cohesion: 0.11
Nodes (11): _non_singleton_component(), Tests for the baseline_survival_and_gamma temporal heuristic strategy and the A, atom_backtrack_exact_resolution_and_gamma: resolution backtrack + gamma., Minimal action-like object compatible with the heuristic's model builder., SyntheticAction, TestCalibrationStatistics, TestCaseA_NoDependency, TestCaseBCD_ComponentFactors (+3 more)

### Community 40 - "Community 40"
Cohesion: 0.16
Nodes (12): T, DeltaNeighbors, DeltaSimpleTemporalNetwork, Any, Adds the constraint `x - y <= b`. This gives an upper bound to the time, Returns the assignment to the given event in the minimal-makespan consistent sol, Check if there is a harder constraint from x to y, Inserts in this STN the constraints to represent both a lower bound and (+4 more)

### Community 41 - "Community 41"
Cohesion: 0.10
Nodes (36): _execution_conflict_table(), Optimal probability of achieving the pattern goal within ``remaining``.      Pas, Static overlap mutexes encoded by converted ``inExecution`` facts.      ``conver, solve_pattern(), _exhaustive_deterministic(), _pattern(), Tests for the exact pattern-MDP heuristic (``exact_pattern_mdp``).  The anchors, Redundant starts disappear and useful groups jump to their boundary. (+28 more)

### Community 42 - "Community 42"
Cohesion: 0.09
Nodes (22): _clamp01(), _fold_free_support(), kmutex_or_hazard(), KMutexInstrumentation, KMutexORResult, _merge_closest_pair(), Mutex-aware K-bounded OR-layer tightening for the admissible PTRPG.  Heuristic n, Fold a free (non-mutex) support into an existing row, clearing its footprint. (+14 more)

### Community 43 - "Community 43"
Cohesion: 0.12
Nodes (7): Represents a `STNPlan`. A Simple Temporal Network plan is a generalization of, Returns all the constraints given by this `STNPlan`. Subsumed constraints, Returns a new `STNPlan` where every `ActionInstance` of the current plan is repl, This function takes a `PlanKind` and returns the representation of `self`, Returns True if exists a time assignment for each STNPlanNode that         does, Returns the earliest tine node can be executed according to the STN constraints, STNPlan

### Community 44 - "Community 44"
Cohesion: 0.12
Nodes (8): Nasa_Rover, sample_rock_good Action, turn_on_dropping Action, turn_on_good_hand Action, communicate_rock_data Action, communicate_image_data Action, ObjectExp(), Returns an expression for the given object.      :param obj: The `Object` that

### Community 45 - "Community 45"
Cohesion: 0.08
Nodes (4): Fraction, Performs basic simplifications of the input expression.      Important NOTE:, Performs basic simplification of the given expression.          If a :class:`~, Simplifier

### Community 46 - "Community 46"
Cohesion: 0.06
Nodes (28): Convert_problem_combination, checks if one of the actions already in the combination is in mutex with the can, adds as a combination action to the problem the `combination`          :param, convert actions from `model` actions to be `engines` actions         This is fo, Finding mutex actions and adding a precondition that they can't be executed in p, Check if two actions are mutex          :param action: The checked action, Adding to the `conflicting_action`, and `action` a precondition that they would, The function adds as an action all combinations of durative actions that can run (+20 more)

### Community 47 - "Community 47"
Cohesion: 0.16
Nodes (4): Machine_Shop, Immersionpaint Action, OverallPreconditionTiming(), Returns the overall timing of an :class:`~unified_planning.model.Action`.

### Community 48 - "Community 48"
Cohesion: 0.06
Nodes (27): Grounder, GrounderHelper, Action, Returns an `Iterator` over all the possible grounded `Actions` of the `Problem`, Grounder class: the `Grounder` takes a :class:`~unified_planning.model.Problem`, Takes an instance of a :class:`~unified_planning.model.Problem` and the `GROUNDI, This class gives the capability of grounding a :class:`~unified_planning.model.P, Creates an instance of the GrounderHelper.          :param problem: The `Probl (+19 more)

### Community 49 - "Community 49"
Cohesion: 0.19
Nodes (21): _and_n_facts(), _and_pairwise(), build_action_specs(), _clamp01(), compute_correlation_preplanning(), CorrActionSpec, _extract_effect_delay_steps(), joint_add_distribution_from_action() (+13 more)

### Community 51 - "Community 51"
Cohesion: 0.11
Nodes (15): combination_plan(), _dynamic_aligned_horizon(), MCTS, Parent-local comparison horizon H_p = min over the parent's children of     the, Evaluate the temporal_probabilistic_rpg heuristic, threading the baseline_cached, Frontier-aligned value of a node, used ONLY for selection — never         backp, aligned_value(n): prefix-roll delta then PTRPG at the GLOBAL H_frontier, Original MCTS solver implementation. (+7 more)

### Community 52 - "Community 52"
Cohesion: 0.14
Nodes (15): _build_achiever_index(), generate_patterns(), grow_pattern(), pattern_covers_any_action(), PDBAction, Fact, Durative probabilistic action with explicit joint outcomes., Probability this action sets ``fact`` true in one application. (+7 more)

### Community 53 - "Community 53"
Cohesion: 0.18
Nodes (6): Action abstraction for duration-aware relaxed propagation., TemporalRelaxedActionModel, End-to-end: the baseline_admissible_paths_table strategy in the engine., TestTableStrategyEngine, SyntheticAction, TestPathMutexStrategy

### Community 54 - "Community 54"
Cohesion: 0.15
Nodes (9): FixedTailExpectimaxGuards, _MockAction, _MockState, _MockSTN, _MockSTNNode, Unit tests for fixed-tail expectimax prefix evaluation., TestFixedTailExpectimax, TestFixedTailExpectimaxMCTSSeed (+1 more)

### Community 55 - "Community 55"
Cohesion: 0.06
Nodes (22): combinationMDP, MDP, Apply the action to this state to produce the next state., Gets the add and delete effect of the prob_outcome index effect, :param action: draw the outcome of the probabilistic effects         :return: th, Enumerates next-state distribution for the given (state, action).          This, :return: the initial state of the problem, Checks if all the goal predicates hold in the `state`         and there are no a (+14 more)

### Community 56 - "Community 56"
Cohesion: 0.07
Nodes (8): ActionQueue, CombinationState, QueueNode, holds action and it's duration left, Compares two nodes based on the duration left, Actions currently in execution and the remaining duration left for their executi, Get the actions that have the smallest duration left.         There can be seve, Extract delta from each of the actions in data: duration_left = duration_left -

### Community 57 - "Community 57"
Cohesion: 0.08
Nodes (3): Walker used to retrieve the `Type` of an expression., Returns the `Type` of the expression.          :param expression: The expressi, TypeChecker

### Community 58 - "Community 58"
Cohesion: 0.09
Nodes (16): ClosedTimeInterval(), LeftOpenTimeInterval(), OpenTimeInterval(), Represents an `Interval` where the 2 bounds are :class:`~unified_planning.model., Returns the `TimeInterval's` lower bound., Returns the `TimeInterval's` upper bound., Returns `False` if this `TimeInterval` lower bound is included in the Interval,, Returns `False` if this `TimeInterval` upper bound is included in the Interval, (+8 more)

### Community 59 - "Community 59"
Cohesion: 0.11
Nodes (11): GlobalStartTiming(), Fraction, Class that used a :class:`~unified_planning.model.Timepoint` to define from when, Returns the `delay` set for this `Timing` from the `timepoint`., Returns `True` if this `Timing` refers to the global timing in the `Plan` and no, Returns `True` if this `Timing` is from the start, `False` if it is from the end, Returns `True` if this `Timing` is from the end, `False` if it is from the start, Returns the start timing of an :class:`~unified_planning.model.Action`.     Cre (+3 more)

### Community 60 - "Community 60"
Cohesion: 0.20
Nodes (3): Full_Conc, Returns the start timing of an :class:`~unified_planning.model.Action`.      F, StartPreconditionTiming()

### Community 61 - "Community 61"
Cohesion: 0.09
Nodes (14): Engine, EngineMeta, OperationMode, Enum, type, Sets the flag deciding if a fail on the problem's :func:`kind <unified_planning., Manages entering a Context (i.e., with statement), Manages exiting from Context (i.e., with statement) (+6 more)

### Community 62 - "Community 62"
Cohesion: 0.33
Nodes (4): frontier_score(), Blended frontier-selection score (Option A / frontier_aligned_*).          front, Option A frontier-aligned selection score (frontier_aligned_*)., TestFrontierScore

### Community 63 - "Community 63"
Cohesion: 0.09
Nodes (10): ExpressionQuantifiersRemover, This walker is used to remove all the quantifiers from an expression by substitu, This method takes in input an expression that might contain quantifiers and a `p, FluentsSubstituter, Performs fluents substitution into a expression, maintaining the same args, Returns the expression where every FluentExp that has as fluent one of, Expression, Performs substitution into an expression (+2 more)

### Community 64 - "Community 64"
Cohesion: 0.04
Nodes (18): OrderedDict, Action, InstantaneousAction, Represents an instantaneous action., This is the `Action` interface., Returns the `list` of the `Action` `preconditions`., Removes all the `Action preconditions`, Returns the `list` of the `Action effects`. (+10 more)

### Community 65 - "Community 65"
Cohesion: 0.20
Nodes (8): _normalize_max_approximation_selection(), value_mode that drives BOTH MCTS expansion ordering and leaf rollout with     t, ``selection_type='max_approximation'`` is the single switch (matching the     g, _uses_fixed_tail_deprecated_ptrpg_rollout(), _uses_max_approximation_value_mode(), _uses_ptrpg_guided_rollout_value_mode(), validate_fixed_tail_ptrpg_rollout_config(), validate_ptrpg_guided_rollout_config()

### Community 66 - "Community 66"
Cohesion: 0.20
Nodes (6): Place a rock under the car Action, Search a rock Action             the robot can find a one of the rocks, Push Gas Pedal Action         The probability of getting the car out is lower t, Push Car Action             The probability of getting the car out is higher th, Init things that can be pushed, Stuck_Car

### Community 67 - "Community 67"
Cohesion: 0.18
Nodes (11): Tunables for rollout-aligned common-horizon evaluation., RolloutAlignedConfig, FakeState, _make_evaluator(), Tests for rollout-aligned common-horizon PTRPG:    * the MDP-agnostic wrapper (R, Spec sanity (#10): align frontier nodes to the deepest elapsed.      Frontier A:, SyntheticAction, TestBaselineSurvivalResolution (+3 more)

### Community 68 - "Community 68"
Cohesion: 0.17
Nodes (6): _env_int(), _env_str(), Read a (lower-cased, stripped) string from the environment., Precompute, for each fact, a list of (action_model, p_a, prec_avail_tables)., Approximate when an action's add effects should become available.          For, Read an int from the environment, falling back to ``default`` on error.

### Community 69 - "Community 69"
Cohesion: 0.09
Nodes (10): EndTiming(), GlobalEndTiming(), Class used to define the point in the time from which a :class:`~unified_plannin, Creates a new `Timepoint`.          It is typically used to refer to:, Returns the `kind` of this `Timepoint`; the `kind` defines the semantic of the `, Returns the `container` in which this `Timepoint` is defined or `None` if it ref, Returns the `Timepoint` from which this `Timing` is considered., Returns the end timing of an :class:`~unified_planning.model.Action`.      For (+2 more)

### Community 70 - "Community 70"
Cohesion: 0.12
Nodes (12): MachineShopNoDeadline, NasaRoverNoDeadline, Stuck Car (1 object) variant with no deadline., Machine Shop variant with same goals and no deadline., Nasa Rover variant with identical goals and no deadline constraint., StuckCar1oNoDeadline, Place a rock under the car Action, Search a rock Action             the robot can find a one of the rocks (+4 more)

### Community 72 - "Community 72"
Cohesion: 0.25
Nodes (8): _build_split_problem(), _name(), TestSelectorConstruction, Greedy MDP dispatcher (same as ``plan()``) until goal, dead end, or deadline., simulate_greedy_mdp_until_terminal(), build_heuristic_adapter(), MaxApproximationConfig, Tests live under comdp_plus_no_deadline/tests (pytest collection path).  Run:

### Community 73 - "Community 73"
Cohesion: 0.17
Nodes (20): _add_mask(), best_joint_outcomes(), build_pattern(), _clamp01(), _conflict_table(), _env_int(), joint_outcomes(), _mask_of() (+12 more)

### Community 74 - "Community 74"
Cohesion: 0.12
Nodes (4): C_ANode, Adds to the SNode the possible actions as children.         If a specific child, Action node with consistency STN check, add constraints to the STN according to this `self` action         If this pare

### Community 75 - "Community 75"
Cohesion: 0.33
Nodes (4): This method retrieves the value in the state.         NOTE that the searched va, This method returns the predicates of the state          :return: The predicat, This is an abstract class representing a classical `Read Only state`, ROState

### Community 76 - "Community 76"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native AGENTS.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 77 - "Community 77"
Cohesion: 0.20
Nodes (8): PatternSolver, Memoised backward induction over boundary states of one pattern.      ``memo`` I, Canonicalise remaining-time multisets inside symmetry classes., Return true when starting ``op`` can only occupy its running slot., Next deadline-aligned duration-GCD boundary in remaining time., Optimise one same-time dispatch group without adding PDB rows., Advance to the next completion or required decision-grid boundary., Fold the joint outcome distribution of simultaneously ending ops.          Disti

### Community 78 - "Community 78"
Cohesion: 0.12
Nodes (9): Fraction, Returns `True` if the expression is a constant, `False` otherwise., Returns the constant value stored in this expression., Return constant `boolean` value stored in this expression., Return constant `real` value stored in this expression., Test whether the expression is a `boolean` constant., Test whether the expression is a `real` constant., Test whether the expression is the `True` Boolean constant. (+1 more)

### Community 79 - "Community 79"
Cohesion: 0.19
Nodes (7): _extract_state_facts(), Any, Fact, Fact-level optimistic abstraction of an action., Propagate relaxed fact probabilities from ``state``.          If the fact-acti, Score a state with either product or minimum goal aggregation.          ``prod, RelaxedActionModel

### Community 80 - "Community 80"
Cohesion: 0.21
Nodes (15): _action_name(), _apply_action_set_sampled(), _build_action_set(), MaxApproximationDebug, _print_max_approximation_debug(), Random, Approximate max action-set selection via goal-backtrack groups.  Builds a valid, Greedy goal-backtrack group builder.      Repeatedly commits the legal, non-mute (+7 more)

### Community 81 - "Community 81"
Cohesion: 0.18
Nodes (4): LinkedList, updates the value of the list according to the lower_bound and upper_bound, Works only  if the rewards are not negative         update the max_value accord, Returns the value of the intervals between the lower and upper bound

### Community 82 - "Community 82"
Cohesion: 0.12
Nodes (7): Precondition, This class represent an precondition. It has a :class:`~unified_planning.model.F, check if the effect and the precondition are the same, Returns the `Fluent` of this `precondition`., Returns the `value` of the `Fluent` needed for the action execution., Sets the `value` needed to the `Precondition` of the `Fluent`.          :param, Returns this `Precondition's Environment`.

### Community 83 - "Community 83"
Cohesion: 0.33
Nodes (4): Attach (or clear) a pre-built PDB correction for the ``baseline_pdb``         s, Generate goal-directed patterns, build their PDBs, and attach the         resul, Return the attached PDB correction, lazily auto-building one for the         ``, PDBCorrection

### Community 84 - "Community 84"
Cohesion: 0.12
Nodes (9): DemoAction, LayerTrace, PropagationResult, Debug snapshot for one propagation layer., Output bundle returned by ``heuristic_propagate``., Build an engine ``State`` from problem initial values for PE evaluation., _reference_state_from_problem(), State (+1 more)

### Community 86 - "Community 86"
Cohesion: 0.12
Nodes (7): Interval, Class that defines an `interval` with 2 :class:`expressions <unified_planning.mo, Returns the `Interval's` lower bound., Returns the `Interval's` upper bound., Returns the `Interval's` `Environment`., Returns `True` if the `lower` bound of this `Interval` is not included in the `I, Returns `True` if the `upper` bound of this `Interval` is not included in the `I

### Community 87 - "Community 87"
Cohesion: 0.25
Nodes (4): Protocol, HeuristicAdapter, Relaxed goal value of a raw fact set (higher = closer to goal)., Facts the action contributes to the relaxed table.

### Community 88 - "Community 88"
Cohesion: 0.07
Nodes (26): ActionScoreEntry, is_active(), _action_name_key(), greedy_matched_value_target(), _heuristic_value(), _null_ctx, plan(), rank_actions_by_score() (+18 more)

### Community 89 - "Community 89"
Cohesion: 0.14
Nodes (4): Best_No_Parallel, Simple, Returns the user type defined in the global environment with the given `name` an, UserType()

### Community 91 - "Community 91"
Cohesion: 0.15
Nodes (16): ClosedDurationInterval(), Duration, DurationInterval, FixedDuration(), LeftOpenDurationInterval(), OpenDurationInterval(), PreconditionTimepointKind, Enum (+8 more)

### Community 92 - "Community 92"
Cohesion: 0.11
Nodes (7): This class represents a node of the `STNPlan`.      :param kind: The `Timepoin, Returns the end time according to the STN when the actions are performed in the, Legal interval for this node in the current plan., Returns the latest tine node can be executed according to the STN constraints, Adds the end action as a chosen action          - The end action must be before, add a deadline to the STN: end plan - start plan <= deadline         :param dea, STNPlanNode

### Community 93 - "Community 93"
Cohesion: 0.14
Nodes (8): Relaxed action mutex for parallel set construction (not admissible)., Memoized front-end for :meth:`_compute_path_actions_mutex` (incl. self)., SOUND action-mutex for the path-mutex layer, INCLUDING self-mutex.          De, Names of actions participating in >= 1 certified mutex pair (incl.         self, Per action: the mutex PARTNERS (actions it certifiably conflicts with,, Build the structural context, per-action components and calibrator once., Build name->model and fact->deleters indices once per heuristic object., Action-mutex extended beyond pure Graphplan delete-interference.          Two

### Community 94 - "Community 94"
Cohesion: 0.17
Nodes (4): baseline_admissible_resolution: goal-directed backward pass over the     2^(k/2), baseline_admissible_resolution_forward: forward anchor-jump with a     per-block, TestAdmissibleResolutionForwardStrategy, TestAdmissibleResolutionStrategy

### Community 95 - "Community 95"
Cohesion: 0.13
Nodes (4): Focused tests for the atom_backtrack_exact_unbiased temporal heuristic strategy., Minimal action-like object compatible with the heuristic's action model builder., SyntheticAction, TestAtomBacktrackExactUnbiased

### Community 96 - "Community 96"
Cohesion: 0.22
Nodes (3): Init all actions into the new actions list         ensures the end actions can, Calculates the heuristic based on the current state and time, TRPG

### Community 98 - "Community 98"
Cohesion: 0.11
Nodes (11): handles, MetaNodeTypeHandler, object, type, Call the correct walk_* function of cls for the given expression., Decorator for walker functions.     Use it by specifying the nodetypes that nee, Metaclass used to intepret the nodehandler decorator., Base Abstract Walker class.     Do not subclass directly, use DagWalker or Tree (+3 more)

### Community 99 - "Community 99"
Cohesion: 0.13
Nodes (22): main(), Quick check: fixed-tail prefix-frac bootstrap and MCTS backups are sensible., at_or_past_tail_horizon(), build_fixed_tail_search_context(), crossed_cutoff(), elapsed_from_root(), _emit_fixed_tail_debug(), fixed_tail_bootstrap_value() (+14 more)

### Community 101 - "Community 101"
Cohesion: 0.24
Nodes (3): NamesExtractor, This walker returns all the names contained in an expression., Returns the set of names contained in this expression.          :param express

### Community 103 - "Community 103"
Cohesion: 0.13
Nodes (9): FixedTailRandomRolloutConfig, FixedTailRandomRolloutEvaluator, random_rollout_config_from_args(), _build_split_problem(), Tests for ephemeral fixed-tail random rollout leaf evaluation (Option A)., TestFixedTailRandomRolloutEvaluator, TestFixedTailRandomRolloutMCTS, `Enum` representing all the possible :func:`kinds <unified_planning.model.Timepo (+1 more)

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
Cohesion: 0.33
Nodes (4): _FakeAction, _FakeANode, _FakeSNode, TestMCTSUctFiltering

### Community 110 - "Community 110"
Cohesion: 0.16
Nodes (14): main(), Batch greedy_parallel runtime: baseline vs resolution backward/forward (alpha=2), build_mdp(), parse_args(), Any, Namespace, Heuristic Per-Call Runtime Benchmark ======================================  Mea, Build and compile an MDP from scratch (fresh caches on each call). (+6 more)

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
Cohesion: 0.10
Nodes (9): QuantifierSimplifier, Same to the :class:`~unified_planning.model.walkers.Simplifier`, but does not ex, Simplifies the expression and the quantifiers in it.         The quantifiers ar, Add children to the stack., Apply function to the node and memoize the result.         Note: This function, Same to the :class:`~unified_planning.model.walkers.QuantifierSimplifier`, but t, Evaluates the given expression in the given `State`.         :param expression:, This method needs to be updated from the QuantifierRemover in order to use the S (+1 more)

### Community 115 - "Community 115"
Cohesion: 0.13
Nodes (8): HeuristicCallMetrics, Use as: `with _WrapperTimer(): ...`, Use as: `with _WorkerTimer() as wt: ...; wt.hit = result.cache_hit`, Accumulates per-call timing for wrapper and worker heuristic levels., Return a dict of aggregated metrics ready for display., Human-readable report string., _WorkerTimer, _WrapperTimer

### Community 116 - "Community 116"
Cohesion: 0.24
Nodes (7): ExactPatternMDPHeuristic, _extract_state_facts(), Admissible upper bound from exact small-pattern CoMDP+ MDPs.      Usage mirrors, Pair snap actions back into durative operations.          ``convert_problem`` sp, Admissible upper bound on P(goal before deadline) from ``state``.          Goals, test_from_problem_keeps_only_true_initial_facts(), Fact

### Community 119 - "Action"
Cohesion: 0.06
Nodes (24): lift_action_instance(), "map" is a map from every action in the "grounded_problem" to the tuple     (or, replace_action(), AbstractProblem, This is an abstract class that represents a generic `planning problem`.      T, Returns the `Problem` `Environment`., Returns the `Problem` `name`., Sets the `Problem` `name`. (+16 more)

### Community 121 - "main"
Cohesion: 0.14
Nodes (20): Any, Path, parse_run_metrics(), print_summary_table(), Shared utilities for experiment scripts.  Provides: - Heuristic alias mapping, Launch run_domain.py as a subprocess and return (stdout_text, returncode)., Print high-signal run_domain.py lines (for Script 3 when verbose=False)., Extract key metrics from run_domain.py stdout. (+12 more)

### Community 122 - "Community 122"
Cohesion: 0.27
Nodes (8): common_footprint(), insert_or_absorb(), Mutex evidence guaranteed by BOTH rows after a merge = the intersection., Insert ``r_new`` into the K-bounded OR fact table, never dropping it.      Optio, _name_mutex(), Symmetric action-mutex from unordered name pairs (handles a==b too)., The three cases of insert_or_absorb., TestInsertOrAbsorb

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
Cohesion: 0.03
Nodes (72): # TODO: changed to be not probabilistic effect, Always(), And(), AtMostOnce(), Bool(), Compiler(), Div(), Dot() (+64 more)

### Community 130 - "Community 130"
Cohesion: 0.02
Nodes (59): UPExpressionDefinitionError, UPProblemDefinitionError, UPValueError, ActionsSetMixin, Returns the action instance if the `problem` has the `action` with the given `na, Adds the given `action` to the `problem`.          :param action: The `action`, Adds the given `actions` to the `problem`.          :param actions: The `list`, This class is a mixin that contains a `set` of `actions` with some related metho (+51 more)

### Community 131 - "Community 131"
Cohesion: 0.22
Nodes (8): PDBOutcome, One joint outcome of an action: fires with ``probability``., Concurrent durative semantics: independent durative actions overlap     instead, Reference: the OLD sequential DP (one action per recursion, charging its     ful, Concurrency only ADDS parallelism/retries, so the new DP must never score     be, _sequential_value(), TestConcurrentDurations, TestNoValueDecrease

### Community 132 - "Community 132"
Cohesion: 0.17
Nodes (11): 1. Heuristics added this session (strategy aliases), 2. Key files, 3. The three alignment prompts (genealogy — caused real naming confusion), 4. ROLLOUT_ALIGNED_H is INERT for dynamic + frontier (fixed this session), 5. Option A bug + audit (the crux of the last part), 6. What I did on Option A before the user took over, 7. CURRENT STATE — needs reconciliation (two implementations coexist), 8. Gotchas / environment (+3 more)

### Community 135 - "Community 135"
Cohesion: 0.32
Nodes (13): _clamp_probability(), _compute_clause_support(), _compute_precondition_support_result(), _format_clause(), _format_precondition_structure(), _is_atomic_fact(), _is_clause_container(), _is_dnf_structure() (+5 more)

### Community 136 - "Community 136"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 141 - "Community 141"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 143 - "Community 143"
Cohesion: 0.33
Nodes (5): CoMDP+ No-Deadline Starter, Included starter scenarios, Run greedy baseline, Run smoke benchmarks (all 5 presets), Run tests for this starter package

### Community 152 - "Path"
Cohesion: 0.21
Nodes (12): Action, and_components(), and_emit_rows(), _facts_mutex(), _footprints_conflict(), guaranteed_rep(), The pair occupies overlapping time AND carries mutex actions.      ``mutex_fn``, Segments guaranteed by EVERY route of the fact = intersection over its     rows' (+4 more)

### Community 153 - "plan.py"
Cohesion: 0.17
Nodes (7): EndPreconditionTiming(), ParamPrecondition(), PreconditionTimepoint, Returns the `kind` of this `Timepoint`; the `kind` defines the semantic of the `, Returns the end timing of an :class:`~unified_planning.model.Action`.      For, Class used to define the precondition point in the time from which a :class:`~un, Creates a new `PreconditionTimepoint`.          It is used to refer to:

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
Cohesion: 0.29
Nodes (4): Estimate E[T_goal] — expected steps to achieve all goal facts — without, Return the per-step survival factor in the geometric tail for this fact, Return failure_fact(s) given failure_fact(s-1) = prev_failure.          step_s, Compute E[T_fact] using the precomputed availability tables.          No recur

### Community 161 - ".is_int_constant"
Cohesion: 0.31
Nodes (3): PDBCorrection, Holds a set of pattern databases and answers applicability queries.      For an, TestPDBCorrectionManager

### Community 162 - "FreeVarsExtractor"
Cohesion: 0.33
Nodes (3): FreeVarsExtractor, This expression walker returns all the `fluent` expression in the given expressi, Returns all the `fluent expressions` in the given expression.          :param

### Community 166 - "collect_global_frontier"
Cohesion: 0.25
Nodes (5): collect_global_frontier(), node_elapsed(), All open/expandable S-nodes in the tree (global frontier F)., Elapsed time at an S-node (STN end time of the incoming action edge)., TestCollectFrontier

### Community 167 - ".kind"
Cohesion: 0.33
Nodes (4): cut_or_hazard(), Marginal OR hazard over one layer's arriving achiever rows — VALUE ONLY.     In, cut_or_hazard = max-weight independent set of the certified-mutex graph.     Mut, TestCutOrHazardMWIS

### Community 168 - ".is_global"
Cohesion: 0.33
Nodes (8): build_pdb_action(), _clamp_probability(), _extract_duration(), _outcome_assignment(), _parse_probabilistic_effect(), Horizon-indexed Pattern Database (PDB) correction for the probabilistic temporal, One probabilistic effect -> list of (prob, add, del) outcomes.      A residual n, Convert one action object into a :class:`PDBAction` (None if it has no     relax

### Community 170 - ".check_stn"
Cohesion: 0.25
Nodes (3): Global open/expandable leaf nodes across the tree (spanning depths)., One Option A iteration: pick the globally best open-leaf node by         fronti, Option A: pick the child to descend/expand by a frontier-aligned score.

### Community 172 - ".create_Snode"
Cohesion: 0.25
Nodes (3): Standard backprop of a freshly expanded value up to the root., Expand ONE child of the selected ORIGINAL node, standard backprop.         The, Create a new Snode for the state `state` with parent `parent`

### Community 174 - "Pattern"
Cohesion: 0.29
Nodes (3): Pattern, Projected behaviour used to identify exchangeable ground actions., Find projected action identities that can be safely permuted.          Members m

### Community 176 - "TestTableStrategyEngine"
Cohesion: 0.33
Nodes (4): RolloutAlignedDiagnostics, PrefixRolloutFn, RawEvalFn, StateHashFn

### Community 178 - "AnyChecker"
Cohesion: 0.33
Nodes (3): AnyChecker, This expression walker checks if any subexpression matches a given predicate., Checks if any of the subexpression matches the predicate.          :param expr

### Community 179 - "compare_paths_table_vs_admissible.py"
Cohesion: 0.67
Nodes (3): build_mdp(), main(), Single-call comparison: baseline_admissible_paths_table vs baseline_admissible.

### Community 181 - "sweep_paths_table_gap.py"
Cohesion: 0.60
Nodes (4): build_mdp(), main(), Per-call runtime: baseline_admissible vs baseline_admissible_paths_table (v3)., time_strategy()

## Knowledge Gaps
- **85 isolated node(s):** `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)`, `Step 1 - Ensure graphify is installed`, `Step 2 - Detect files` (+80 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **39 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TemporalProbabilisticRPGHeuristic` connect `Community 10` to `Community 2`, `Community 3`, `Community 4`, `Community 131`, `Community 138`, `Community 12`, `Community 13`, `Community 14`, `Community 16`, `Community 21`, `Community 22`, `Community 23`, `Community 27`, `Community 30`, `.heuristic_expected_time`, `Community 33`, `.is_int_constant`, `Community 36`, `Community 37`, `.kind`, `Community 39`, `Community 42`, `.copy_stn`, `Community 46`, `TestChainedFootprints`, `compare_paths_table_vs_admissible.py`, `Community 51`, `Community 53`, `sweep_paths_table_gap.py`, `._ensure_admissible_lp_bound`, `Community 62`, `Community 67`, `Community 68`, `Community 72`, `Community 80`, `Community 83`, `Community 87`, `Community 88`, `Community 93`, `Community 94`, `Community 95`, `Community 99`, `Community 106`, `Community 107`, `Community 111`, `Community 122`, `Community 127`?**
  _High betweenness centrality (0.261) - this node is a cross-community bridge._
- **Why does `FNode` connect `Community 1` to `Community 128`, `shortcuts.py`, `Community 8`, `Community 15`, `Community 17`, `Community 18`, `Community 20`, `plan.py`, `Community 29`, `Community 31`, `FreeVarsExtractor`, `Community 35`, `Community 38`, `Community 44`, `.And`, `Community 45`, `Community 48`, `AnyChecker`, `.is_int_constant`, `Community 57`, `Community 58`, `Community 59`, `Community 63`, `Community 69`, `Community 78`, `Community 86`, `Community 91`, `Community 98`, `Community 101`, `Community 103`, `Community 105`, `Community 114`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **Why does `C_MCTS` connect `Community 30` to `Community 10`, `Community 11`, `Community 12`, `Community 19`, `Community 23`, `Community 26`, `Community 31`, `.check_stn`, `.create_Snode`, `Community 51`, `.create_Snode_root_interval`, `Community 54`, `Community 65`, `Community 88`, `Community 92`, `Community 99`, `Community 103`, `Community 109`, `Community 116`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 33 inferred relationships involving `FNode` (e.g. with `create_action_with_given_subs()` and `Environment`) actually correct?**
  _`FNode` has 33 INFERRED edges - model-reasoned connections that need verification._
- **Are the 82 inferred relationships involving `TemporalProbabilisticRPGHeuristic` (e.g. with `PlanResult` and `SyntheticAction`) actually correct?**
  _`TemporalProbabilisticRPGHeuristic` has 82 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `C_MCTS` (e.g. with `ExactPatternMDPHeuristic` and `TemporalProbabilisticRPGHeuristic`) actually correct?**
  _`C_MCTS` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `Environment` (e.g. with `Action` and `CombinationAction`) actually correct?**
  _`Environment` has 32 INFERRED edges - model-reasoned connections that need verification._