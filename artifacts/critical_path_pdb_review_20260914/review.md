# Critical path PDB runtime investigation

Measured on 14 September 2026 using the existing repository-root implementation, Python 3.12, and PYTHONHASHSEED=0. The heuristic and its tests were not edited. The attached design document was treated as reference material. The benchmark uses the real Prob_Conc domain, Grounder, Convert_problem, MDP, and EngineModel.

## Size of the tested problem

The joint benchmark has four durative actions and zero garbage actions. The actions are eight, four, two, and one, with durations 8, 4, 2, and 1. Compilation produces eight snap actions, one start and one end per durative action. There are no zero-duration durative actions. The PDB's noop processes the next running end.

The graph uses eight distinct Boolean atoms: got(a), got(b), got(c), got(d), and the four corresponding inExecution flags. Initially none is true. At most four are simultaneously true in any reached state. The domain additionally declares got(e), but it stays false and never appears in a reached fact set when there are no garbage actions.

Each goal/action pair can be unfinished and idle, running, or achieved. This gives 3^4 = 81 distinct fact sets. Including the committed order of running ends gives 168 (F,Q) states. Counts by queue length 0, 1, 2, 3, 4 are 16, 32, 48, 48, 24. There are 508 successor edges. These counts are independent of the tested horizon because graph construction does not use the deadline.

## Unprofiled timings

| Horizon | Graph build | Backward solve | Stored entries | Maximum per state | Initial-state value |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.057 s | 1.169 s | 3,297 | 45 | 0.7995259852 |
| 16 | 0.054 s | 19.892 s | 10,293 | 191 | 0.9840789900 |
| 25 | 0.037 s | 196.803 s | 22,671 | 553 | 0.9988276286 |

These are single-run measurements, not averages. Absolute times differ from the previous chat's 118 seconds, while state and entry counts agree exactly. Profiling was disabled for these measurements. All three runs report complete and converged. An existing-table initial-state lookup at D=25 took 0.153 milliseconds. The initial state itself has only 18 entries at D=25; the 22,671 entries are spread over all states and their running-action configurations.

## Where the time goes

At D=16, a separate cProfile run attributes 41.649 of 51.564 seconds to _outcome_sum, including its calls to prune. All prune calls together account for 28.532 seconds. These inclusive times overlap and must not be added. Profiling changes runtime; use the unprofiled table for wall-clock comparisons.

The counted D=16 run performs 3,780 state backups, 9,320 branch backups, and 15,837 pruning calls. It passes 4,425,472 candidate entries through pruning, with 10,282 candidates in the largest individual call, before retaining 10,293 final entries.

_outcome_sum forms combinations of entries from different stochastic outcomes. prune sorts and deduplicates candidates, then compares each one against previously retained candidates using probability and every open tail. Retries repeatedly improve successor values and trigger further backups. Consequently, a small number of logical states can still require millions of intermediate value combinations and comparisons.

The solver already schedules only affected predecessor states. The narrower proposal to cache unchanged outgoing branches is a different optimization. In the counted D=16 run, only 95 of 9,320 branch calls (1.02%) have exactly unchanged child-entry lists. Their outcome combination and pruning take 0.341 seconds out of 24.194 seconds for all branch combinations (1.41%), within a 28.485-second instrumented run. These timings exclude the regression lists constructed before calling _outcome_sum. This evidence does not establish that branch caching would bring D=25 below 30 seconds. Improvements to outcome combination and dominance pruning deserve investigation; a measured prototype is needed to quantify any speedup.

## Irrelevant actions and patterns

The current CriticalPathPDB class does not automatically compute goal relevance or project facts. EngineModel retains complete fact sets and optionally accepts allowed_ops from its caller.

| Garbage actions supplied | Total durative actions | Reached states | Distinct atoms appearing in F |
| ---: | ---: | ---: | ---: |
| 0 | 4 | 168 | 8 |
| 1 | 5 | 872 | 10 |
| 2 | 6 | 4,960 | 11 |

These garbage experiments only build the graph; they do not solve its value tables. Explicit goal-relevance closure selects the four useful actions in all three cases. The design document's small untimed backward-table example therefore does not imply that this forward implementation automatically ignores added garbage actions.

With a separate explicitly restricted one-goal model for each goal, all four D=25 PDBs together build and solve in 12.787 milliseconds. They contain 12 states and 95 entries in total. Their values are 1, 0.999271, 0.9996903707, and 0.9998658931. Their product is 0.9988276287705012. This product is exact for this domain's independent goal processes with unrestricted cross-action concurrency; it is not a generally valid aggregation rule for interacting goals.

## Validation and scope

All 25 existing critical-path PDB tests pass. The independent-domain closed form is the product, over the four actions, of 1 - (1-p)^floor(D/d). Joint values agree to numerical tolerance.

The objective measured here is binary reward for reaching the full goal before the deadline. No intermediate reward is included. General admissibility is not established by these tests. The implementation is intended as an optimistic temporal relaxation; its per-state probability tolerance also matters: at D=25 its result is approximately 1.26e-10 below the independent closed form. A literal guaranteed floating-point upper bound would need a certified stopping/error treatment. The existing known running-window relaxation remains a separate source of optimism.

## Reproduction files

- benchmark.py: unprofiled horizon runs, individual goals, and separate cProfile run.
- inspect_work.py: counts candidate processing, unchanged branch calls and their time; small garbage graph checks.
- results.json: raw benchmark measurements.
- work_counts.json: operation counts and graph checks.
- profile_D16.txt and profile_D16.prof: profile output.

Main source locations: critical_path_pdb.py lines 86 (pruning), 247 (forward expansion), 304 (worklist solve), 335 (state backup), and 360 (outcome combination); temporal_stn_pdb.py lines 173-188 (adapter and action restriction); probabilistic_conc.py (four actions and garbage generator).
