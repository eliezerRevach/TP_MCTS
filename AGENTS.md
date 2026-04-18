# Codex Project Instructions

## Project Context
- This repository is a prototype for TP-MCTS thesis experiments, not production software.
- The main research topic is online temporal planning in stochastic domains with durative and concurrent actions.
- The thesis continuation should be framed as reward maximization under a deadline. Goal reachability is a special case where reward is `1` for reaching the goal before the deadline and `0` otherwise.
- Reference paper text: `docs/Temporal_Planning___IJCAI25_Camera_Ready-2.txt`.
- Original Cursor rules live in `.cursor/rules/`; keep this file aligned with those rules when they change.

## Repository Map
- `unified_planning/`: main local planning framework, domains, STN code, MCTS, RTDP, greedy baselines, and heuristics.
- `unified_planning/engines/solvers/mcts.py`: TP-MCTS-related solver logic.
- `unified_planning/engines/heuristics/trpg.py`: temporal relaxed planning graph heuristic code.
- `unified_planning/plans/stn/`: Simple Temporal Network implementation.
- `unified_planning/domains/`: benchmark domains such as stuck car, hosting, NASA rover, machine shop, simple, concurrency, and probabilistic variants.
- `scripts/`: experiment runners and result-building scripts.
- `results/`: generated experiment CSV/XLSX-style outputs.
- `experiments.ipynb` and `demo.ipynb`: active notebooks for experiments and demonstrations.
- `comdp_plus_no_deadline/`: starter environment for no-deadline CoMDP+ experiments and related tests.
- `tp_mcts/`: duplicated or nested copy of much of the project; prefer editing the repository-root modules unless the user explicitly asks to work inside this copy.

## Research Rules
- Be explicit about heuristic status: admissible, inadmissible, or admissible only in expectation.
- For expectation-based admissibility, state the guarantee type: worst-case bound, confidence bound, or no formal guarantee.
- Do not present average empirical performance as admissibility unless a formal bound is provided.
- Separate time-to-goal estimation from probability-of-success estimation. Conservative timing does not automatically imply conservative success probability.
- Keep reward behavior in view, including intermediate rewards when relevant; do not reduce analysis only to binary goal success unless that is the stated objective.
- Treat root planning graph reuse cautiously because deeper nodes commit to one realized action path and reuse can become optimistic.
- Helpful actions may bias exploration, but should not silently restrict the legal action set.
- Temporal constraints can break naive value propagation assumptions. Convergence or optimality arguments must state optimism assumptions in estimates and propagation.

## Heuristic Bias Checks
- Use the m-doors family as a recurring sanity check for heuristic bias.
- Track the known failure mode where a heuristic suggests collecting all keys once and then moving `n` steps, while real cost can behave like about `2*m*n` because each door may require returning for a different key.
- Heuristic changes should be evaluated against m-doors-like or similar polynomial-gap counterexamples.
- If using average-case calibration, report confidence, variance, and failure probability on adversarial structures.

## Algorithm Concepts To Preserve
- TP-MCTS decides both which action to dispatch and when to dispatch it.
- Offline compilation may split durative actions into `start` and `end` snap actions and track execution with `InExecution(a)` fluents.
- Each MCTS node may carry both a state and an STN; STN updates enforce action ordering, duration, and deadline constraints.
- PTRPG-style evaluation heuristics estimate value instead of using standard rollouts.
- Backpropagation variants may include earliest feasible scheduling and root-interval value functions over feasible root execution time intervals.
- Historical baselines include MW-RTDP and MW-MCTS, plus local greedy and RTDP implementations.

## Coding Style
- Prefer minimal, clear code that supports fast thesis experiments.
- Avoid production-style robustness unless needed for the experiment or requested by the user.
- Keep changes scoped to the asked experiment, algorithm, or analysis.
- Preserve assignment/thesis-facing APIs, notebook outputs, and experiment result formats unless the user explicitly asks to change them.
- Prefer readable Python and direct implementations over broad abstractions.
- Do not delete generated results, logs, notebooks, or pickle domains unless the user explicitly asks.
- Be careful with existing local modifications; treat them as user work.

## Commands
- Run targeted tests when changing core logic:
  - `python -m pytest unified_planning/tests`
  - `python -m pytest unified_planning/engines/solvers`
  - `python -m pytest comdp_plus_no_deadline/tests`
- Run no-deadline smoke experiments with:
  - `python -m comdp_plus_no_deadline.run_smoke`
- Run a no-deadline scenario with:
  - `python -m comdp_plus_no_deadline.run_no_deadline --scenario easy_nasa_rover_1 --domain_type combination --runs 5 --max_steps 250 --seed 123`
- Original README example:
  - `bash runexp -l log/nasa_rover.log run_domain.py --domain nasa_rover --deadline 35 --runs 3`

## Reporting Expectations
- In writeups, clearly separate what comes from the TP-MCTS paper from what is newly proposed or modified in this thesis continuation.
- When discussing results, include success behavior, reward behavior, deadline effects, and temporal feasibility effects.
- When making claims about admissibility or optimism, name the assumptions and identify counterexamples or gaps.
