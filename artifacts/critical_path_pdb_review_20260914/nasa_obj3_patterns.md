# NASA rover with nine goal-specific PDBs

NASA rover, object_amount=3, deadline/horizon=25, regular domain. One separately built critical-path PDB per goal: three image goals and six rock goals. Measured on 14 September 2026 using Python 3.12 with PYTHONHASHSEED=0. All nine tables were built sequentially and retained for subsequent lookups; no symmetry reuse or parallel speedup was applied. These are single measurements per goal.

All nine graph builds completed and all nine backward solves converged under the implementation's tolerance. No state, update, or external time limit was reached. The heuristic itself was not modified.

Total pattern build-and-solve time: 320.212 seconds. Sequential batch wall time including pattern selection and measurement bookkeeping: 320.531 seconds. Shared domain grounding and compilation: an additional 0.449 seconds.

Graph construction totals 6.098 seconds; backward calculation totals 314.101 seconds. All tables together contain 33,690 states and 373,485 value entries. The median time for a batch of nine existing-table initial-state lookups is 0.280 milliseconds, across 31 repeated batches. Lookup values match the values recorded after construction.

| Goal | Graph build (s) | Backward solve (s) | Build + solve (s) | States | Entries | Value |
|---|---:|---:|---:|---:|---:|---:|
| communicated_image_data(o0) | 0.005 | 0.009 | 0.015 | 14 | 128 | 0.9741254400 |
| communicated_image_data(o1) | 0.005 | 0.012 | 0.017 | 14 | 128 | 0.9741254400 |
| communicated_image_data(o2) | 0.003 | 0.012 | 0.015 | 14 | 128 | 0.9741254400 |
| communicated_rock_data(x0) | 0.809 | 55.226 | 56.037 | 5,608 | 62,306 | 0.9986773006 |
| communicated_rock_data(x1) | 0.799 | 57.508 | 58.309 | 5,608 | 62,228 | 0.9986773006 |
| communicated_rock_data(x2) | 1.123 | 52.608 | 53.733 | 5,608 | 61,988 | 0.9986773006 |
| communicated_rock_data(x3) | 0.921 | 53.811 | 54.734 | 5,608 | 62,157 | 0.9986773006 |
| communicated_rock_data(x4) | 1.130 | 54.166 | 55.298 | 5,608 | 62,130 | 0.9986773006 |
| communicated_rock_data(x5) | 1.303 | 40.750 | 42.055 | 5,608 | 62,292 | 0.9986773006 |

Each image pattern retains 3 durative actions, 3 varying domain facts, 3 execution flags, and 33 facts that remain true throughout its graph. Each rock pattern retains 23 durative plus 4 instantaneous operations, 15 varying domain facts, 13 varying execution flags, and 29 facts that remain true throughout its graph. Some retained operations are statically inapplicable, explaining why the count of observed execution flags is smaller than the retained durative-action count.

The full compiled model has 99 operations and 246 declared ground Boolean atoms. Pattern selection uses the existing goal_relevance_closure helper, with the selected singleton goal explicitly passed to EngineModel. This restricts the action set and retains the full fact sets; it is not a fixed-size fact projection. One goal therefore does not mean one retained domain fact.

The largest pattern has 5,608 states and 4,640 expanded nonterminal states, below the default 50,000 expanded-state budget. Each pattern had an external 180-second time limit, which was not reached.

Values concern binary reward for achieving the selected goal by the deadline. Graph completeness and numerical convergence do not certify exactness or a formal admissibility guarantee for the temporal relaxation. No product or other joint-goal aggregation was computed.

Reproduction: benchmark_nasa_obj3_patterns.py. Raw results: nasa_obj3_patterns.json.
