# NASA rover critical path PDB timings

Measured on 14 September 2026 with the existing heuristic, Python 3.12, PYTHONHASHSEED=0, one rover, two rocks, one image objective, and horizon 25. One run per case. Domain grounding and conversion take an additional 0.06-0.10 seconds, excluded from table-build timings below. The solver was not modified.

| Model | Durative + instantaneous actions | Varying domain facts + execution flags | Constant true facts retained | States | Value entries | Build and solve | Complete |
|---|---:|---:|---:|---:|---:|---:|---|
| image | 3 + 0 | 3 + 3 | 11 | 14 | 128 | 0.017 s | yes |
| rock | 23 + 4 | 15 + 13 | 7 | 5,608 | 62,182 | 46.083 s | yes |
| full | 29 + 4 | 19 + 19 | 7 | 86,996 | 143,133 | 43.959 s | no |

Image and rock select one goal and restrict operations using the existing goal_relevance_closure helper. Facts are retained in full; these are goal-specific models, not arbitrary fixed-fact projections. Counts describe facts that actually vary among discovered states. Some retained operations are statically inapplicable. The full compiled model declares 56 ground Boolean atoms.

The full case hit the default 50,000 expanded-state limit. It discovered 86,996 states, leaving 36,996 unexpanded. Graph construction took 5.476 seconds and backward calculation 38.373 seconds. Its returned value was 1.0. Unexpanded states receive a vacuous value of 1, so this 43.959-second result is an incomplete budget-limited calculation, not a measured runtime for the completed full-problem PDB. No uncapped runtime was measured.

The complete rock-goal graph took 0.362 seconds to build; the backward calculation took 45.710 seconds. It returned 0.998677300644. The complete image-goal graph took 0.0023 seconds to build and 0.0146 seconds to solve, returning 0.97412544. Neither case hit the state, update, or external 180-second time limit.

All values concern binary goal reward by the deadline. Completeness here means graph expansion and convergence under the implementation's tolerance. It does not establish exactness or a formal admissibility guarantee; the existing temporal relaxation and numerical tolerance remain.

Raw measurements: nasa_image.json, nasa_rock.json, nasa_full.json. Reproduction: benchmark_nasa.py, with NASA_PDB_CASE set to image, rock, or full, and PYTHONHASHSEED=0.
