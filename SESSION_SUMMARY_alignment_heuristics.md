# Session summary — bias-correction & alignment heuristics (TP-MCTS)

Persistent memory of a long working session that added several PTRPG heuristic
variants and the MCTS "alignment" family. Written after the user re-implemented
Option A themselves (Cursor Composer) in a separate module; see "Current state"
for the merge situation that still needs cleanup.

---

## 1. Heuristics added this session (strategy aliases)

Parser numbering in `unified_planning/parser.py` (`_parse_temporal_heuristic_strategy`):

| # | strategy | what it is |
|---|----------|-----------|
| 11 | `baseline_survival_and_gamma` | baseline_survival + component-wise AND-layer gamma correction on R(a) |
| 12 | `atom_backtrack_exact_resolution_and_gamma` | resolution backtrack (log/exp layers) + AND-layer gamma |
| 13 | `baseline_survival_resolution` | survival DP over log-spaced (exp-width) layers: `P_{t-k}` with k = exp gap |
| 14/15/16 | `rollout_aligned_{baseline,survival,resolution_survival}` | rollout-aligned common-horizon PTRPG (dynamic **parent-local** H_p) |
| 17/18/19 | `frontier_aligned_{baseline,survival,resolution_survival}` | "Option A" — but my impl was **mistakenly parent-local** (see §5) |
| 20/21/22 | `frontier_aligned_option_a{,_survival,_resolution}` | the **correct** global-frontier Option A (USER re-implemented) |

`*_and_gamma` and `*_survival_resolution` are the pairwise combos. **There is NO
triple** survival+resolution+gamma (`baseline_survival_resolution_and_gamma`) —
asked about, never built. `baseline_survival_resolution` uses
`compute_precondition_support` directly (no gamma).

---

## 2. Key files

- `comdp_plus_no_deadline/engines/and_gamma.py` — AND-layer gamma: `AndGammaConfig`,
  `StructuralContext` (component/edge detection: mutex/negative/positive/unknown),
  `build_components`, `AndGammaCalibrator` (pair stats w/ absence data, broad+exact
  caches, shrinkage `λ=n/(n+n0)`, gamma stability bounds, candidate-pair universe,
  `RolloutSimulator`, diagnostics). Default gamma table: positive 1.2, negative 0.7,
  mutex 0.3, singleton/unknown 1.0. Lazy rollout calibration is OFF by default.
- `comdp_plus_no_deadline/engines/rollout_aligned.py` — `RolloutAlignedConfig`,
  `RolloutAlignedEvaluator.evaluate(state, R, h_override)`, `_resolve_horizon`
  (dynamic vs fixed), boundary modes, dead-end→0/goal→1, budgets, diagnostics,
  and the pure `frontier_score(existing, aligned, λ, exploration)` helper.
- `comdp_plus_no_deadline/engines/frontier_aligned_option_a.py` — **USER's clean
  Option A module**: `OPTION_A_STRATEGIES`, `collect_global_frontier`,
  `compute_H_frontier` (+assert), `aligned_value_for_node` (prefix→delta→PTRPG at
  H_frontier, +assert), `select_frontier_node`, `format_option_a_debug_row`,
  `build_option_a_evaluator` (`OptionAConfig.common_horizon_H=999` to prove fixed-H
  is never read). This is the canonical Option A.
- `comdp_plus_no_deadline/engines/temporal_probabilistic_rpg.py` — added
  `_heuristic_propagate_baseline_survival_resolution`,
  `_heuristic_propagate_baseline_survival_and_gamma` (refactored
  `_heuristic_propagate_baseline_survival` to accept optional `r_estimator`),
  `_heuristic_propagate_atom_backtrack_exact_resolution_and_gamma` (refactored
  resolution to accept `gamma_factor_fn`), `_ensure_and_gamma_built`,
  `_and_gamma_factor`; dispatch + `_normalize_strategy` updates.
- `unified_planning/engines/solvers/mcts.py` — alignment integration (see §5/§6).
- `unified_planning/parser.py` — aliases 11–22 + rollout-aligned CLI knobs +
  `--frontier-aligned-debug`.
- `scripts/run_mcts_heuristic_comparison.py` + `scripts/experiment_common.py` —
  forward ALL `--rollout-aligned-*` knobs (incl. boundary-mode/lambda-align/fixed-h
  — these were initially MISSING from the middle layer and caused
  `error: unrecognized arguments`; fixed).
- `experiments.ipynb` — Config cell lists all heuristic keys + new params
  (`ENABLE_ROLLOUT_CALIBRATION`, `ROLLOUT_ALIGNED_H/REDO/FIXED_H/BOUNDARY/LAMBDA_ALIGN`,
  budgets); Script 1 cell forwards them capability-gated.
- Tests: `comdp_plus_no_deadline/tests/test_baseline_survival_and_gamma.py`,
  `comdp_plus_no_deadline/tests/test_rollout_aligned.py` (incl. A/B Option A sanity).

---

## 3. The three alignment prompts (genealogy — caused real naming confusion)

1. **Original** `rollout_aligned_common_horizon_PTRPG`: FIXED `common_horizon_H`;
   `H=min(common_horizon_H,R)`, prefix-roll Δ=R−H, PTRPG suffix, avg over redo.
   Aligned value **is** the heuristic. **No lambda_align.**
2. **Dynamic parent-local**: `H_p = min remaining over a PARENT's children`. Added a
   scoring blend `(1−λ)Q + λ V_aligned`. → became `rollout_aligned_*` default
   (FIXED_H=True reproduces prompt 1). **I implemented dynamic H, DEFERRED the λ blend.**
3. **Option A** `frontier_aligned_PTRPG_selection`: GLOBAL frontier,
   `H_frontier = deadline − max_elapsed_in_F`, aligned value used as a
   SELECTION score, expand original node, standard backprop.

`lambda_align`: belongs to prompts 2 & 3 by spec. **Implemented only for
frontier (Option A).** `rollout_aligned_*` parses it but ignores it (deferred).

---

## 4. ROLLOUT_ALIGNED_H is INERT for dynamic + frontier (fixed this session)

`_resolve_horizon` reads `common_horizon_H` (=ROLLOUT_ALIGNED_H) **only** when
`use_dynamic_H=False` (fixed mode, i.e. `rollout_aligned_*` + `--rollout-aligned-fixed-h`).
Dynamic mode → `H=min(h_override,R)` (no cap); no override → `H=R`. Frontier forces
`use_dynamic_H=True`. Removed `cap_dynamic_h_at_fixed`. Tests prove inertness.
So ROLLOUT_ALIGNED_H affects ONLY `rollout_aligned_*` with FIXED_H=True.

---

## 5. Option A bug + audit (the crux of the last part)

My `frontier_aligned_*` (17/18/19) was a **mix-up**: it used `snode.children`
(the parent's siblings) as the "frontier" and `_dynamic_aligned_horizon`
(parent-local) — i.e. it was basically `rollout_aligned` + a λ-blend in `uct()`,
NOT global Option A. Audit findings:
- violated "must not use parent-local logic" (#3), "select globally" (#4),
  "don't backprop aligned score" (#7 — it seeded `node.value`).
- ROLLOUT_ALIGNED_H inert (ok), rollout endpoints discarded (ok).

**Debug trace proved it** (machine_shop): the parent-local "frontier" = root's 6
`start_*` children, all at `elapsed=0` → `deepest_elapsed=0`, `delta=0` everywhere →
no prefix rollout, `H_frontier=remaining` → alignment is a NO-OP; `aligned=0`, `Q=0`
→ selection ≈ pure exploration → **score 0**.

**Why ~900s before:** parent-local + AND-gamma calibration ON + resolution_survival
suffix; the `wait_no_overshoot` prefix tries each legal action with rollback. With
`frontier_aligned_baseline` + calibration off + redo=1, runtime was **~30s/run**.

---

## 6. What I did on Option A before the user took over

- **"Fix flags first"** (user's choice): `heuristic()` now backs up **standard PTRPG**
  for frontier (not aligned); aligned value is SELECTION-ONLY via
  `_frontier_aligned_value`; added `--frontier-aligned-debug` / `FRONTIER_ALIGNED_DEBUG=1`
  trace; added evaluator-level A/B sanity test (A elapsed0/R25, B elapsed10/R15 →
  H_frontier=15, A rolls 10 then PTRPG(15), B direct PTRPG(15)).
- **Global-frontier driver (inline, my version)** added to `C_MCTS`:
  `_collect_frontier`, `_frontier_aligned_value_global` (lazy cache per H_frontier),
  `_pick_expand_action`, `_backprop_to_root`, `_expand_and_backprop`,
  `_frontier_iteration` (prints `[frontier_aligned/global]`), dispatched in `search()`.
  Verified: global frontier spans depths, prints DEGENERATE note when all elapsed equal,
  ~30s/run. (baseline suffix still scored 0 on machine_shop — that's heuristic weakness,
  the alignment mechanism is correct; A/B math proven by unit test.)

---

## 7. CURRENT STATE — needs reconciliation (two implementations coexist)

`unified_planning/engines/solvers/mcts.py` now contains BOTH:
- **My inline driver** (~lines 993–1140): `_collect_frontier`, `_frontier_iteration`
  (`[frontier_aligned/global]`), etc. Triggered for `frontier_aligned_*` (17/18/19).
- **User's module driver** (~1141+): `_option_a_frontier_iteration`
  (`[frontier_aligned_option_a]`) using `frontier_aligned_option_a.py`. Triggered for
  `frontier_aligned_option_a*` (20/21/22).

`search()` dispatch (~line 596): `use_option_a_driver` (user's) takes precedence over
`use_frontier_driver` (mine). In my last test, `frontier_aligned_option_a` produced
**no debug output** — the user's `_option_a_frontier_iteration` either wasn't reached
or its debug gate differs; the user is testing their own version.

### Open cleanup items
- **Pick one Option A**: keep the USER's clean `frontier_aligned_option_a.py` module;
  REMOVE my redundant inline `_frontier_iteration`/`_collect_frontier`/
  `_frontier_aligned_value_global`/`_expand_and_backprop`/`_backprop_to_root` and decide
  fate of `frontier_aligned_*` (17/18/19, my parent-local-turned-global).
- Verify the user's `_option_a_frontier_iteration` fires its debug + works e2e.
- `lambda_align` blend for `rollout_aligned_*` still DEFERRED (parsed, unused there).
- `expected_stochastic_rounding` boundary mode parsed but treated as `overshoot`
  (not implemented).

---

## 8. Gotchas / environment

- **nasa_rover.py has a pre-existing UNCOMMITTED typo** `calidbrated` (→`calibrated`)
  ~line 409 in `take_image_action`; crashes nasa_rover construction
  (`TypeError: 'NoneType' object is not callable`). NOT mine. Tests deselect
  `test_no_deadline_setup.py::...::test_domain_has_no_deadline`.
- **Windows console cp1252** crashes on `─`/`Σ`/unicode when piping; prefix runs with
  `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.
- Importing `unified_planning.engines.solvers.mcts` directly hits a **circular import**;
  run via `run_domain.py` (set `PYTHONPATH=$PWD`).
- **Colab clones from GitHub** (`/content/tp_mcts`) — local edits must be pushed for
  Colab to see them. Notebook cells are capability-gated to degrade gracefully.
- Full `comdp_plus_no_deadline/tests` suite ≈ **104 passed** (1 deselected nasa typo).

---

## 9. First-test config for Option A (per the spec)

```
heuristic = frontier_aligned_option_a   (or frontier_aligned_baseline)
redo = 1 ; lambda_align = 1.0 ; boundary = wait_no_overshoot
and_gamma_rollout_calibration = False   (frontier suffix = baseline -> gamma is moot anyway)
FRONTIER_ALIGNED_DEBUG=1                 (env) or --frontier-aligned-debug
```
Asserts in code: `H_frontier == deadline - deepest_elapsed`;
`delta == deepest_elapsed - elapsed(node)`; deepest nodes delta=0,
aligned_value == PTRPG(state, H_frontier).
