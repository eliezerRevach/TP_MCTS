"""
Shared utilities for experiment scripts.

Provides:
- Heuristic alias mapping (user-facing names -> internal strategy keys)
- Subprocess runner that captures stdout and parses solver metrics
- CSV writer helper

Resolution heuristic (`atomic_exact_resolution` / `atom_backtrack_exact_resolution`):
pass `resolution_alpha`, `resolution_forced_minimum`, `resolution_reference_t`
into `run_domain_subprocess`, or append the matching `--resolution-*` flags via
`extra_args`.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Heuristic naming
# ---------------------------------------------------------------------------

# Map user-facing short names to internal (strategy, heuristic_name) pairs.
# heuristic_name is either "temporal_probabilistic_rpg" or "trpg".
HEURISTIC_ALIASES: dict[str, dict[str, str]] = {
    "baseline": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline",
        "label": "baseline",
    },
    "baseline_cached": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_cached",
        "label": "baseline_cached",
    },
    # Inadmissible, EVENT-DRIVEN forward expansion with independence operators.
    # Unlike baseline (unit-layer sweep), this advances time anchor-to-anchor: at
    # delta_t expand all applicable actions, land each effect at delta_t+d(a), then
    # jump to the next scheduled arrival time (steps by durations, not unit 1). The
    # retry recursion carries the previous anchor P(delta_t)=P(delta_{t-1})+
    # (1-P(delta_{t-1}))H_t with noisy-AND product R(a)=prod_f P(f) and noisy-OR
    # H_t(f)=1-prod_e(1-B_e). Coarse grid re-fires actions once per anchor
    # (serialized) => everywhere <= baseline/baseline_admissible, NOT admissible.
    "baseline_forward": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_forward",
        "label": "baseline_forward",
    },
    # Admissible upper-bound PTRPG (PTRPG_Cleaned.docx). Same layered forward DP
    # as baseline, but the AND/precondition layer uses the Frechet MIN bound
    # R_t(a)=min_f P_t(f) (instead of the independence product) and the OR/fact
    # layer uses the UNION bound H_t(f)=min(1,sum_e B_e) (instead of noisy-OR).
    # Both are always-valid upper bounds (min>=prod, union>=noisy-OR), so the
    # heuristic is VERY OPTIMISTIC but ADMISSIBLE — it never under-estimates
    # reachability under the delete-relaxed temporal RPG envelope.
    "baseline_admissible": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible",
        "label": "baseline_admissible",
    },
    # Synonym: shorthand for baseline_admissible.
    "admissible": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible",
        "label": "baseline_admissible",
    },
    # baseline_admissible operators (Frechet-min AND + union-bound OR) computed as
    # a GOAL-DIRECTED BACKWARD recursion that evaluates achievers only at the
    # 2^(k/2) resolution anchors instead of every completion layer. The skipped
    # completion layers are NOT dropped: each anchor block charges its full width
    # n_b at the block's latest (most optimistic) time, so it stays an ADMISSIBLE
    # upper bound -- looser than dense baseline_admissible, but faster (log layers).
    # Resolution knobs via resolution_alpha / resolution_forced_minimum /
    # resolution_reference_t (same as atom_backtrack_exact_resolution).
    "baseline_admissible_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_resolution",
        "label": "baseline_admissible_resolution",
    },
    # Synonym: shorthand for baseline_admissible_resolution.
    "admissible_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_resolution",
        "label": "baseline_admissible_resolution",
    },
    # FORWARD anchor-jump analogue of baseline_admissible_resolution: incremental
    # forward sweep over only the 2^(k/2) anchors, jumping P with the block closed
    # form P=1-(1-P)(1-H)^n_b (H = union hazard copied over the n_b skipped EARLIER
    # layers; end-of-block anchors keep it ADMISSIBLE and tighter than the backward
    # variant). Computes all facts (no goal scoping) but is cache-friendly/incremental.
    # Same resolution knobs as baseline_admissible_resolution.
    "baseline_admissible_resolution_forward": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_resolution_forward",
        "label": "baseline_admissible_resolution_forward",
    },
    # Synonym: shorthand for baseline_admissible_resolution_forward.
    "admissible_resolution_forward": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_resolution_forward",
        "label": "baseline_admissible_resolution_forward",
    },
    # baseline_admissible with the OR/fact layer tightened by the marginal-
    # consistent LP bound (PTRPG_Cleaned.docx Section 9.3). Per arrival fact it
    # solves max P(OR-of-AND achievers) over all local joint distributions
    # consistent with the stored marginal upper bounds, instead of the capped
    # union bound. Still ADMISSIBLE (the true local joint is feasible) but never
    # looser than the union bound and strictly tighter when achievers share
    # preconditions. Falls back to the union bound when the local fact set exceeds
    # TP_MCTS_ADMISSIBLE_LP_MAX_LOCAL_FACTS (default 8). Knobs (env vars, set in
    # the notebook config block): TP_MCTS_ADMISSIBLE_LP_MAX_LOCAL_FACTS,
    # TP_MCTS_ADMISSIBLE_LP_VALUE_MODE ("union" safe default | "independent").
    "baseline_admissible_lp": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_lp",
        "label": "baseline_admissible_lp",
    },
    # Synonym: shorthand for baseline_admissible_lp.
    "admissible_lp": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_lp",
        "label": "baseline_admissible_lp",
    },
    # baseline_admissible with the OR/fact layer tightened by a mutex-aware
    # K-bounded bound. Per arrival fact the achiever contributions are bucketed
    # into <= K rows by certified action-mutex; the hazard is
    # sum(free rows) + max(surviving mutex clique) instead of the capped union
    # bound. Always <= baseline_admissible (max <= sum), and reduces to it
    # exactly when no two achievers landing at a cell are mutex. Headline metric:
    # fraction of OR-nodes where a pure mutex clique of size >= 2 survived (best
    # observed on machine_shop free(m), the exclusive-achiever case). Knob:
    # TP_MCTS_KMUTEX_K (default 3); TP_MCTS_KMUTEX_DEBUG=1 prints surviving cliques.
    "baseline_admissible_kmutex": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_kmutex",
        "label": "baseline_admissible_kmutex",
    },
    # Synonym: shorthand for baseline_admissible_kmutex.
    "admissible_kmutex": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_kmutex",
        "label": "baseline_admissible_kmutex",
    },
    # baseline_admissible with a TEMPORAL path-mutex tightening. Instead of a
    # scalar P_t(f), carries <= K timed achiever-paths per fact and combines them
    # by segment-overlap mutex: two alternative paths that share a mutex action in
    # overlapping time windows -- INCLUDING an action mutex with itself when it
    # occupies a resource (deletes a precondition it needs, e.g. a car driving
    # [0,15] and [5,20]) -- collapse via max instead of summing as independent
    # retries; a conjunctive AND path is dropped when its chosen achievers cannot
    # run in parallel ("at least one mutex breaks the parallel"). K via
    # TP_MCTS_KMUTEX_K; headline metric via log_pathmutex_summary().
    "baseline_admissible_paths": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_paths",
        "label": "baseline_admissible_paths",
    },
    # Synonym: shorthand for baseline_admissible_paths.
    "admissible_paths": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_paths",
        "label": "baseline_admissible_paths",
    },
    # baseline_admissible with a TABLE-FLOWING path representation: a K-row
    # achiever-path table flows through the whole RPG so the AND layer (action
    # preconditions AND the goal) can apply cross-fact temporal-mutex bounds.
    # Goal score uses aggregation="kernelized" (and_cumulative_bound over goal
    # path tables); KMUTEX_K caps rows per fact-layer. ADMISSIBLE upper bound.
    "baseline_admissible_paths_table": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_paths_table",
        "label": "baseline_admissible_paths_table",
    },
    # Synonym: shorthand for baseline_admissible_paths_table.
    "admissible_paths_table": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_paths_table",
        "label": "baseline_admissible_paths_table",
    },
    # baseline_admissible tightened in the TEMPORAL direction by exact joint
    # sweeps over small goal-directed patterns. The marginal-only cumulative
    # bound min(1, sum_t H_t) is the tightest bound derivable from marginals
    # alone (the exact recursion needs P(achieved at t | NOT by t-1), which the
    # marginals do not determine), and it saturates to 1.0 as the deadline
    # grows. Here a few facts are tracked JOINTLY, so the conditional is a ratio
    # of stored values and there is no cap to hit: cross-layer discrimination
    # survives at long deadlines. Patterns grow backwards from each goal fact,
    # attaching whichever freed precondition the abstraction inflates most;
    # everything outside stays relaxed (freed but gated by relaxed
    # reachability), and the sweep clamps to min(marginal, pattern) — so this is
    # an ADMISSIBLE upper bound. Knobs: SURVIVOR_PDB_* below.
    "baseline_admissible_survivor_pdb": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_survivor_pdb",
        "label": "baseline_admissible_survivor_pdb",
    },
    # Synonym: shorthand for baseline_admissible_survivor_pdb.
    "survivor_pdb": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_admissible_survivor_pdb",
        "label": "baseline_admissible_survivor_pdb",
    },
    # PURE survivor-PDB: the pattern DP with NO RPG sweep in the per-node path.
    # Once every goal fact has a pattern the RPG contributes nothing -- the
    # pattern bound is exact inside its own sub-world and is always the binding
    # side of min(marginal, pattern), so the sweep computes a number that is
    # then discarded. Only the achiever GATES are still needed from outside the
    # pattern, and those are pure delete-relaxed reachability (a cheap fixpoint,
    # no probabilities). Pattern solves are memoized on the ABSTRACT state, so
    # many concrete search nodes share one sweep. Same values as
    # baseline_admissible_survivor_pdb, ~15x faster than baseline_admissible
    # (~1.1ms vs ~16ms per node on nasa_rover obj2) -- which at a fixed search
    # time is many more MCTS iterations. ADMISSIBLE.
    "survivor_pdb_pure": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "survivor_pdb_pure",
        "label": "survivor_pdb_pure",
    },
    # LAZY survivor-PDB: same abstraction as survivor_pdb_pure, cached on a
    # horizon-INDEPENDENT key. NOT value-identical -- see the caveat below.
    #
    # survivor_pdb_pure keys the solve memo on (pattern, projection, gates,
    # depth) and the pattern structure on (goals, depth). depth is
    # min(configured, floor(deadline - current_time)), so it moves with the
    # clock: every distinct time in the tree splits the cache, and the whole
    # table is thrown away as the deadline draws in -- including the pattern
    # BUILD, which runs a full baseline_admissible sweep for its growth scoring
    # (~110ms on nasa_rover(3) vs ~1ms for a cached solve).
    #
    # Dropping depth is exact, not an approximation: in solve_survivor_pattern
    # an achiever fires at layer t only when t >= gate, so layer t depends on no
    # achiever gated later than t. A sweep to horizon H therefore contains the
    # sweep to any d <= H as a literal PREFIX, and a query at remaining time d
    # is a column read. So: sweep once to the largest depth requested, index by
    # remaining. Entries then survive across MCTS decisions, and nodes at
    # different clocks sharing a projection+gates share one sweep.
    #
    # gates deliberately STAY in the key. Deriving them from the pattern facts
    # alone would make the value a function of the abstract state (a true PDB,
    # ~12x fewer sweeps) but was measured on nasa_rover(3) to move the estimate
    # from 0.083 to 0.953 at deadline 12 -- looser on 60/60 sampled states. The
    # gates carry nearly all of the discriminating information.
    #
    # Given the SAME pattern set this returns exactly the same numbers as
    # survivor_pdb_pure -- verified on nasa_rover(3), 25 states at depth 15,
    # fresh heuristic instance per state: 0/25 differ.
    #
    # CAVEAT -- live numbers can still differ from pure, via a state-dependence
    # that ALREADY EXISTS in pure: build_survivor_patterns takes
    # initial_facts=state_facts, so a pattern is built from whichever state
    # first populates its cache key and is reused for every later state sharing
    # that key. pure keys on (targets, depth); this keys on (targets). Different
    # key -> different builder state -> different Phi -> different (still
    # admissible) bound. With shared instances, 10 of those same 25 states
    # differed. Neither is "more correct": the pattern build is not a function
    # of the pattern alone, which is worth fixing on its own terms.
    #
    # Smaller second effect: build_survivor_patterns does read the horizon. On
    # nasa_rover(3) the pattern set is identical for horizons 12..25 and only
    # changes at 10 and below (9 patterns -> 6).
    #
    # Speedup scales with how many distinct depths the search visits (sweeps stay
    # flat while queries grow): 1.05x at 1 depth, 2.74x at 5, ~4x at 10-20.
    #
    # ADMISSIBLE.
    "survivor_pdb_lazy": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "survivor_pdb_lazy",
        "label": "survivor_pdb_lazy",
    },
    # Synonym.
    "survivior_pdb_lazy": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "survivor_pdb_lazy",
        "label": "survivor_pdb_lazy",
    },
    # TIME-LOOP survivor-PDB: survivor_pdb_lazy plus the one-instance-at-a-time
    # phase. Identical patterns, gates and cache; ONE mechanism changes.
    #
    # What it fixes: solve_survivor_pattern re-fires every applicable achiever
    # at EVERY layer, so an action of duration d draws d independent coins
    # during the single execution it is actually running. Over a horizon H it
    # draws H times where at most H/d executions can physically finish, and the
    # inflation GROWS with the deadline -- the same saturation that makes MCTS
    # layers look alike.
    #
    # The mechanism (v18 section 3): an action starts once its preconditions
    # hold, occupies d(a), and only then may start again. With c(a) = time since
    # its last precondition arrived, mod d(a), the attempts that COMPLETE in a
    # step of width Delta are n(a,Delta) = floor((c(a)+Delta)/d(a)) and each
    # achiever contributes peff = 1-(1-p)^n. Unit steps => n is an indicator
    # that is 1 once every d layers, so a duration-5 action draws 5x fewer
    # coins. The counter lives in the state next to the fact mask, one per
    # distinct (pattern preconditions, duration) pair, kept modulo d with a lap
    # bit -- so it stays bounded, and a unit-duration domain pays nothing and
    # returns exactly the lazy numbers.
    #
    # Two anchors: the first completion is at max(tau+d, gate) -- pattern
    # preconditions held long enough AND the freed preconditions' relaxed gate.
    # The counter only knows tau modulo d, so both progressions are admitted and
    # the larger count is taken; the true one is among them, so the bound holds.
    # Firing density <= 2/d, and exactly 1/d when the anchors agree mod d.
    #
    # ADMISSIBLE, and pointwise <= survivor_pdb_lazy: the phase only removes
    # completions no single-instance execution could have made. The bounded-
    # state fallback forgets counters, which restores the fire-every-layer
    # behaviour (still an upper bound, just untightened).
    #
    # Env knobs: TP_MCTS_SURVIVOR_LOOP_INTEREST_POINTS=1 restricts the sweep to
    # the interest points T (v18 section 1) instead of every integer layer --
    # exact within the pattern's model, cheaper, off by default so loop differs
    # from lazy in exactly one mechanism. TP_MCTS_SURVIVOR_LOOP_CONCURRENCY=N
    # allows N overlapping instances of one grounded action (default 1 = this
    # codebase's semantics). SURVIVOR_PDB_* knobs apply unchanged.
    "survivor_pdb_loop": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "survivor_pdb_loop",
        "label": "survivor_pdb_loop",
    },
    # Synonyms.
    "survivor_pdb_rpg_loop": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "survivor_pdb_loop",
        "label": "survivor_pdb_loop",
    },
    "survivor_loop": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "survivor_pdb_loop",
        "label": "survivor_pdb_loop",
    },
    # ---- DIRECT-VALUE variants: NOT the PTRPG ----------------------------
    # Same propagation as the strategy they wrap (they share its cache), but the
    # value goes to TP-MCTS as the PATTERN computed it instead of being
    # re-derived by the PTRPG's goal scorer.
    #
    # The PTRPG is one heuristic: a per-fact per-layer DP whose goal score is the
    # INDEPENDENCE PRODUCT of the goal marginals. Every strategy in
    # temporal_probabilistic_rpg.py currently inherits that product, including
    # the pattern/PDB family -- which re-imposes, at the very last step, exactly
    # the assumption the joint distribution exists to remove. Worse, it is
    # UNSOUND: a product of upper bounds is not an upper bound on a conjunction
    # when the goals correlate, while
    #     min(U(A), U(B)) >= min(P(A), P(B)) >= P(A and B)
    # always holds. exact_pattern_mdp already scores this way on its own MCTS
    # path (mcts.py) and documents the same argument; these aliases give the
    # survivor family the same treatment.
    #
    # Effect: min >= product, so the value RISES -- sound but looser. The
    # tightness given up is recoverable with MULTI-GOAL patterns (seed Phi with
    # 2+ goals and read the conjunction off the joint), which is the correlation
    # the one-pattern-per-goal builder never models.
    #
    # Aggregation is forced to min for these, ahead of
    # TP_MCTS_HEURISTIC_AGGREGATION -- the product is not offered.
    "survivor_pdb_loop_direct": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "survivor_pdb_loop_direct",
        "label": "survivor_pdb_loop_direct",
    },
    "survivor_pdb_lazy_direct": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "survivor_pdb_lazy_direct",
        "label": "survivor_pdb_lazy_direct",
    },
    "survivor_pdb_pure_direct": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "survivor_pdb_pure_direct",
        "label": "survivor_pdb_pure_direct",
    },
    # Forward baseline DP whose AND/precondition layer is tightened by a
    # horizon-indexed Pattern Database: R_t(a) = P(pre(a) jointly reachable by
    # layer t) from a per-pattern backward DP (max over projected actions),
    # replacing the independence product prod_f P_t(f). Falls back to the product
    # when no pattern covers pre(a); degrades to plain baseline when no PDB is
    # attached. No survival/delete decay, no resolution shrinking. Configured by
    # the --pdb-* CLI knobs (num patterns / max facts per pattern / growth policy).
    "baseline_pdb": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_pdb",
        "label": "baseline_pdb",
    },
    # Delete/survival-aware baseline: forward DP with a per-step survival factor
    # S_t(f) so deletable facts (e.g. free(m)) decay below 1 instead of being
    # pinned at 1. NOT monotone, NOT admissible.
    "baseline_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_survival",
        "label": "baseline_survival",
    },
    # Same survival propagation as baseline_survival, but scored with the
    # variance-aware "meanvar" goal aggregation (mean - alpha*sqrt(k-1)*std over
    # per-goal areas). Kept separate so the two can be compared head-to-head.
    "baseline_survival_meanvar": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_survival_meanvar",
        "label": "baseline_survival_meanvar",
    },
    # Survival propagation with the component-wise AND-layer gamma correction
    # replacing the flat precondition product R(a). Collapses to baseline_survival
    # when no static precondition dependency is detected. NOT a calibrated
    # probability; the goal is better AND-layer ranking direction.
    "baseline_survival_and_gamma": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_survival_and_gamma",
        "label": "baseline_survival_and_gamma",
    },
    # Resolution backtrack (log-spaced / exponential-width layers) with the same
    # component-wise AND-layer gamma correction as baseline_survival_and_gamma.
    # Collapses to atomic_exact_resolution when no precondition dependency exists.
    "atomic_exact_resolution_and_gamma": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution_and_gamma",
        "label": "atomic_exact_resolution_and_gamma",
    },
    # Synonym: internal temporal_heuristic_strategy name.
    "atom_backtrack_exact_resolution_and_gamma": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution_and_gamma",
        "label": "atom_backtrack_exact_resolution_and_gamma",
    },
    # Survival/delete forward DP over log-spaced (exponential-width) resolution
    # layers: P_{t-k} with k = exponential gap. Standalone, and the suffix
    # evaluator of rollout_aligned_resolution_survival.
    "baseline_survival_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_survival_resolution",
        "label": "baseline_survival_resolution",
    },
    # Rollout-aligned common-horizon PTRPG (3 versions). Each aligns a node's
    # remaining horizon to a shared suffix horizon H via real prefix rollouts,
    # then scores the common suffix with the named underlying PTRPG.
    #   v1: baseline (pure testing) | v2: baseline_survival | v3: survival+resolution
    "rollout_aligned_baseline": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "rollout_aligned_baseline",
        "label": "rollout_aligned_baseline",
    },
    "rollout_aligned_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "rollout_aligned_survival",
        "label": "rollout_aligned_survival",
    },
    "rollout_aligned_resolution_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "rollout_aligned_resolution_survival",
        "label": "rollout_aligned_resolution_survival",
    },
    # Option A: frontier-aligned SELECTION. Same per-node aligned value as the
    # rollout-aligned strategies, but used as a frontier selection score (blended
    # with Q via lambda_align) to choose which child to expand; the original node
    # is expanded (no rollout endpoints inserted). Compare against rollout_aligned_*.
    "frontier_aligned_baseline": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_baseline",
        "label": "frontier_aligned_baseline",
    },
    "frontier_aligned_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_survival",
        "label": "frontier_aligned_survival",
    },
    "frontier_aligned_resolution_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_resolution_survival",
        "label": "frontier_aligned_resolution_survival",
    },
    # Fresh global Option A (selection-only aligned value; no lambda blend).
    # First-test CLI: --rollout-aligned-redo 1 --rollout-aligned-boundary-mode wait_no_overshoot
    "frontier_aligned_option_a": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_option_a",
        "label": "frontier_aligned_option_a",
    },
    "frontier_aligned_option_a_survival": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_option_a_survival",
        "label": "frontier_aligned_option_a_survival",
    },
    "frontier_aligned_option_a_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "frontier_aligned_option_a_resolution",
        "label": "frontier_aligned_option_a_resolution",
    },
    "atomic_exact": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact",
        "label": "atomic_exact",
    },
    "atomic_exact_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution",
        "label": "atomic_exact_resolution",
    },
    # Synonym: internal temporal_heuristic_strategy name (same mapping as atomic_exact_resolution).
    "atom_backtrack_exact_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution",
        "label": "atom_backtrack_exact_resolution",
    },
    # Bias-corrected variant: same base scoring as atomic_exact_resolution + structural
    # per-layer correction B(t). Pre-planning is amortized; per-call cost is one lookup.
    "atomic_exact_unbiased": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_unbiased",
        "label": "atomic_exact_unbiased",
    },
    "atom_backtrack_exact_unbiased": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_unbiased",
        "label": "atom_backtrack_exact_unbiased",
    },
    "atomic_exact_cached": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_cached",
        "label": "atomic_exact_cached",
    },
    "fast_atom_cache": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "fast_atom_cache",
        "label": "fast_atom_cache",
    },
    # Correlation-aware DP leaf; same as --heuristic_name baseline_pessimistic on run_domain.
    "baseline_pessimistic": {
        "heuristic_name": "baseline_pessimistic",
        "temporal_heuristic_strategy": "baseline",
        "label": "baseline_pessimistic",
    },
    # Historical typo alias (parser + MCTS accept it).
    "baseline_passmistic": {
        "heuristic_name": "baseline_pessimistic",
        "temporal_heuristic_strategy": "baseline",
        "label": "baseline_pessimistic",
    },
    "ptrpg_old": {
        "heuristic_name": "trpg",
        "temporal_heuristic_strategy": "baseline",  # unused for trpg
        "label": "ptrpg_old (trpg)",
    },
    # Exact small-pattern CoMDP+ MDP. NOT a PTRPG strategy and shares no code with
    # that family: it is DELETE-AWARE (a consumable is genuinely consumed, so the
    # estimate plateaus instead of saturating to 1.0 with the horizon) and it
    # MAXIMISES over a real policy set instead of firing every applicable achiever.
    # State is (pattern facts, in-flight ops, remaining time, sticky goal bit);
    # one backward pass over a DAG -- no value/policy iteration. The only
    # relaxations are the projection onto a small fact set, most-favourable-probe
    # for state-dependent effect probabilities, and dropped at-end conditions
    # (no shipped domain uses EndPreconditionTiming, so that one is vacuous here).
    # Goals combine with MIN, not a product: min of upper bounds is an upper bound,
    # a product of them can fall below the truth. Admissible.
    # Knobs: TP_MCTS_EXACT_PDB_MAX_FACTS (default 4), _MAX_PATTERNS, _MAX_STATES.
    "exact_pattern_mdp": {
        "heuristic_name": "exact_pattern_mdp",
        "temporal_heuristic_strategy": "baseline",  # unused for exact_pattern_mdp
        "label": "exact_pattern_mdp",
    },
    # Part I of the symbolic-STN temporal PDB
    # (comdp_plus_no_deadline/engines/temporal_stn_pdb.py, spec in
    # artifacts/Temporal_STN_PDB_Exact_Draft.docx sections 1-9).
    #
    # It is not a member of ANY existing family and, uniquely here, re-derives
    # nothing: expansion runs through this MDP's own legal_actions and
    # transition_function, so start effects, end effects, adds, deletes,
    # start/end conditions, the inExecution mutex and STATE-DEPENDENT outcome
    # probabilities are the model's, not a reconstruction. That closes by
    # construction the extraction class of bug that once cost machine_shop its
    # only free(machine) re-achiever.
    #
    # State is the spec's annotated STN: X = annotated Z, with (s, R, C)
    # recovered by REPLAY. Time stays SYMBOLIC -- the graph branches on event
    # ORDER, never on a numeric clock -- so one node covers a whole interval of
    # times that a discretising solver would enumerate separately.
    #
    # phi is the IDENTITY. Section 7 requires a proved Markov abstraction
    # ("same legal controls and same abstract transition law", "Do not free
    # omitted preconditions by default") and no pattern builder in this repo
    # supplies that proof -- build_pattern frees exactly what section 7 forbids
    # freeing. So the naive variant abstracts nothing and the value is exact
    # w.r.t. the compiled problem whenever the graph closes.
    #
    # NOT admissible-by-fiat and NOT a scalar: the solver computes an INTERVAL
    # [lo, hi]. A budget-cut leaf is (0, 1) -- section 5's "UNKNOWN is not value
    # zero" -- and lo == hi only when the graph closed. heuristic_score returns
    # hi (an upper bound). Read `last_result.exact` to find out whether that
    # number was the value or merely a bound.
    #
    # Section 6 is where "exact" needs care, so the solver BRACKETS instead of
    # asserting. A symbolic node covers a SET of ground states (one per feasible
    # time assignment) and one scalar max over that set lets two outcome
    # branches each rely on a different value of a time committed BEFORE the
    # branch -- which no real execution can do. Two runs pin that down:
    #   TP_MCTS_STN_PDB_PIN_STARTS=1  -> S_a = C, every time a point, backup IS
    #                                    the MDP backup => EXACT for that policy
    #                                    class => a LOWER bound on the optimum.
    #   TP_MCTS_STN_PDB_PIN_STARTS=0  -> section 4 as written => an UPPER bound.
    # V_pinned <= V* <= V_symbolic, and when the two AGREE the value is certified
    # exact. Measured on prob_match_cellar(1): they agree at every deadline 2..12
    # (0.70, 0.70, 0.91, 0.91, 0.91, 0.91).
    #
    # KNOWN LIMIT, measured: the graph is a TREE (section 5, "One prefix has one
    # parent before merging"; merging is Part II), and without merging it does
    # not answer AT ALL on a real domain. prob_conc(1) at deadline 2:
    #   tree  = 200k nodes / 262s, still CUT at [0.000, 1.000]
    #   merge = 128k nodes / 286s, EXACT at 0.000
    # machine_shop(2) and nasa_rover(2) at deadline 6 do not close either way.
    # The tree re-solves every interleaving of independent starts, so the blow-up
    # is driven by the number of concurrently startable actions, NOT by the
    # horizon. Use it as ground truth on small instances, not as a benchmark
    # leaf, until Part II history compression lands.
    #
    # Knobs: TP_MCTS_STN_PDB_PIN_STARTS (bracket side, default 0 = upper),
    # TP_MCTS_STN_PDB_NODE_BUDGET (default 20000),
    # TP_MCTS_STN_PDB_MAX_PREFIX_EVENTS (default 64),
    # TP_MCTS_STN_PDB_MERGE (Part II merging, UNPROVED, default off),
    # TP_MCTS_STN_PDB_TIE_STARTS (allow a start tied with a pending end).
    "temporal_stn_pdb": {
        "heuristic_name": "temporal_stn_pdb",
        "temporal_heuristic_strategy": "baseline",  # unused for temporal_stn_pdb
        "label": "temporal_stn_pdb",
    },
    "stn_pdb": {
        "heuristic_name": "temporal_stn_pdb",
        "temporal_heuristic_strategy": "baseline",
        "label": "temporal_stn_pdb",
    },
    # ILAO* on the time-left windows MDP (comdp_plus_no_deadline/engines/
    # windows_ilao_pdb.py, spec in artifacts/Windows_ILAO_PDB.docx).
    #
    # State (F, Q, (lo, hi, e) per running action, r): facts, end order, the
    # real-time window of each running action's time left, the time left still
    # guaranteed against the charged deadline, and time in hand. No clock.
    # Guided by the survivor sweep (delete-relaxed occupancy with the phase loop
    # of rpg_exact_states_v18). Offline: ILAO* per goal pattern from the initial
    # state for TP_MCTS_WILAO_OFFLINE_SECONDS; online: budgeted ILAO* from each
    # leaf, reusing the table. Aggregation over patterns: min (sound).
    #
    # Measured (standalone, prob_conc all 4 goals as one pattern): exact 0.7995
    # at D = 8 in 0.44 s / 140 expansions, 0.9426 at D = 12 in 0.70 s; without
    # the sweep the same solve takes 32 s / 231 s.
    #
    # Knobs: TP_MCTS_WILAO_OFFLINE_SECONDS (30), TP_MCTS_WILAO_PATTERN_GOALS
    # (1; 0 = all goals in one pattern), TP_MCTS_WILAO_QUERY_SECONDS (0.05),
    # TP_MCTS_WILAO_QUERY_EXPANSIONS (500), TP_MCTS_WILAO_AGGREGATION (min).
    "windows_ilao_pdb": {
        "heuristic_name": "windows_ilao_pdb",
        "temporal_heuristic_strategy": "baseline",  # unused for windows_ilao_pdb
        "label": "windows_ilao_pdb",
    },
    "windows_ilao": {
        "heuristic_name": "windows_ilao_pdb",
        "temporal_heuristic_strategy": "baseline",
        "label": "windows_ilao_pdb",
    },
    # MCTS leaf: real stochastic rollout to terminal 0/1; PTRPG only guides action choice.
    "ptrpg_guided_rollout_baseline_survival_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "baseline_survival_resolution",
        "value_mode": "ptrpg_guided_terminal_rollout",
        "ptrpg_guided_rollout_policy": "baseline_survival_resolution",
        "label": "ptrpg_guided_rollout_baseline_survival_resolution",
    },
    "ptrpg_guided_rollout_atomic_exact_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution",
        "value_mode": "ptrpg_guided_terminal_rollout",
        "ptrpg_guided_rollout_policy": "atomic_exact_resolution",
        "label": "ptrpg_guided_rollout_atomic_exact_resolution",
    },
    # MCTS leaf: PTRPG-guided prefix to fixed tail horizon, then PTRPG(state, H).
    "fixed_tail_atomic_exact_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution",
        "value_mode": "fixed_tail_mcts_sampled",
        "ptrpg_guided_rollout_policy": "atomic_exact_resolution",
        "label": "fixed_tail_atomic_exact_resolution",
    },
    "fixed_tail_mcts_sampled_atomic_exact_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution",
        "value_mode": "fixed_tail_mcts_sampled",
        "label": "fixed_tail_mcts_sampled_atomic_exact_resolution",
    },
    "fixed_tail_random_rollout_atomic_exact_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution",
        "value_mode": "fixed_tail_random_rollout_eval",
        "label": "fixed_tail_random_rollout_atomic_exact_resolution",
    },
    "fixed_tail_expectimax_atomic_exact_resolution": {
        "heuristic_name": "temporal_probabilistic_rpg",
        "temporal_heuristic_strategy": "atom_backtrack_exact_resolution",
        "value_mode": "fixed_tail_ptrpg_rollout",
        "fixed_tail_prefix_frac": 0.05,
        "fixed_tail_prefix_policy": "expectimax",
        "label": "fixed_tail_expectimax_atomic_exact_resolution",
    },
}

ALL_HEURISTICS = list(HEURISTIC_ALIASES.keys())

DEFAULT_HEURISTICS_MCTS = [
    "ptrpg_old",
    "baseline",
    "baseline_cached",
    "atomic_exact",
    "atomic_exact_resolution",
    "atomic_exact_cached",
    "fast_atom_cache",
]

DEFAULT_HEURISTICS_RUNTIME = [
    "ptrpg_old",
    "baseline",
    "baseline_cached",
    "atomic_exact",
    "atomic_exact_resolution",
    "atomic_exact_cached",
    "fast_atom_cache",
]

# ---------------------------------------------------------------------------
# Repository root detection
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
RUN_DOMAIN_PY = REPO_ROOT / "unified_planning" / "run_domain.py"


def validate_heuristics(names: list[str]) -> None:
    for n in names:
        if n not in HEURISTIC_ALIASES:
            valid = ", ".join(ALL_HEURISTICS)
            raise ValueError(f"Unknown heuristic '{n}'. Valid options: {valid}")


# ---------------------------------------------------------------------------
# Subprocess runner for run_domain.py
# ---------------------------------------------------------------------------

def run_domain_subprocess(
    *,
    domain: str,
    object_amount: int,
    deadline: int,
    runs: int,
    seed: int,
    solver: str,
    domain_type: str = "regular",
    heuristic_name: str,
    temporal_heuristic_strategy: str,
    temporal_heuristic_depth: int,
    search_time: int = 1,
    search_depth: int = 40,
    k: int = 10,
    selection_type: str = "avg",
    exploration_constant: float = 10.0,
    reward_mode: str = "deadline",
    discount_factor: float = 0.95,
    step_penalty: float = -0.05,
    value_mode: str = "tp_mcts",
    final_selection: str = "q",
    ptrpg_guided_rollout_policy: str | None = None,
    ptrpg_guided_rollout_max_steps: int | None = None,
    ptrpg_guided_rollout_epsilon: float | None = None,
    ptrpg_guided_rollout_debug: bool = False,
    fixed_tail_prefix_frac: float | None = None,
    fixed_tail_prefix_policy: str | None = None,
    fixed_tail_debug: bool = False,
    fixed_tail_expectimax_max_nodes: int | None = None,
    fixed_tail_expectimax_max_depth: int | None = None,
    fixed_tail_expectimax_max_time_sec: float | None = None,
    fixed_tail_rollout_samples: int | None = None,
    fixed_tail_rollout_policy: str | None = None,
    garbage_amount: int = 0,
    resolution_alpha: float | None = None,
    resolution_forced_minimum: bool = False,
    resolution_reference_t: int | None = None,
    max_approx_alpha: float | None = None,
    max_approx_num_samples: int | None = None,
    max_approx_seed: int | None = None,
    max_approx_debug: bool = False,
    and_gamma_rollout_calibration: bool = False,
    pdb_num_patterns: int | None = None,
    pdb_max_facts_per_pattern: int | None = None,
    pdb_expansion_policy: str | None = None,
    pdb_seed_per_goal: bool = True,
    pdb_grow_until_covers: bool = False,
    pdb_cover_hard_cap: int | None = None,
    rollout_aligned_h: int | None = None,
    rollout_aligned_redo: int | None = None,
    rollout_aligned_policy: str | None = None,
    rollout_aligned_cache: bool = False,
    rollout_aligned_max_rollouts_per_node: int | None = None,
    rollout_aligned_max_rollouts_per_search: int | None = None,
    rollout_aligned_max_time_per_search: float | None = None,
    rollout_aligned_fallback: str | None = None,
    rollout_aligned_fixed_h: bool = False,
    rollout_aligned_boundary_mode: str | None = None,
    rollout_aligned_min_dynamic_horizon: int | None = None,
    rollout_aligned_fallback_if_small: str | None = None,
    rollout_aligned_lambda_align: float | None = None,
    frontier_option_a_debug: bool = False,
    extra_args: list[str] | None = None,
    verbose: bool = False,
    workers: int = 1,
) -> tuple[str, int]:
    """
    Launch run_domain.py as a subprocess and return (stdout_text, returncode).

    Uses the current Python interpreter so Colab/venv paths are respected.

    workers > 1 splits the runs over that many run_domain.py processes running at
    the same time (see ``_run_parallel``); the returned output is merged so that
    ``parse_run_metrics`` reads the same numbers one process would print.
    """
    cmd = [
        sys.executable,
        str(RUN_DOMAIN_PY),
        "--domain", domain,
        "--object_amount", str(object_amount),
        "--garbage_amount", str(garbage_amount),
        "--deadline", str(deadline),
        "--runs", str(runs),
        "--search_time", str(search_time),
        "--search_depth", str(search_depth),
        "--k", str(k),
        "--selection_type", selection_type,
        "--exploration_constant", str(exploration_constant),
        "--reward_mode", reward_mode,
        "--discount_factor", str(discount_factor),
        "--step_penalty", str(step_penalty),
        "--value_mode", value_mode,
        "--final_selection", final_selection,
        "--seed", str(seed),
        "--solver", solver,
        "--domain_type", domain_type,
        "--heuristic_name", heuristic_name,
        "--temporal_heuristic_depth", str(temporal_heuristic_depth),
        "--temporal_heuristic_strategy", temporal_heuristic_strategy,
    ]
    if resolution_alpha is not None:
        cmd.extend(["--resolution-alpha", str(resolution_alpha)])
    if resolution_forced_minimum:
        cmd.append("--resolution-forced-minimum")
    if resolution_reference_t is not None:
        cmd.extend(["--resolution-reference-t", str(resolution_reference_t)])
    if max_approx_alpha is not None:
        cmd.extend(["--max-approx-alpha", str(max_approx_alpha)])
    if max_approx_num_samples is not None:
        cmd.extend(["--max-approx-samples", str(max_approx_num_samples)])
    if max_approx_seed is not None:
        cmd.extend(["--max-approx-seed", str(max_approx_seed)])
    if max_approx_debug:
        cmd.append("--max-approx-debug")
    if and_gamma_rollout_calibration:
        cmd.append("--and-gamma-rollout-calibration")
    if pdb_num_patterns is not None:
        cmd.extend(["--pdb-num-patterns", str(pdb_num_patterns)])
    if pdb_max_facts_per_pattern is not None:
        cmd.extend(["--pdb-max-facts-per-pattern", str(pdb_max_facts_per_pattern)])
    if pdb_expansion_policy is not None:
        cmd.extend(["--pdb-expansion-policy", str(pdb_expansion_policy)])
    if not pdb_seed_per_goal:
        cmd.append("--pdb-no-seed-per-goal")
    if pdb_grow_until_covers:
        cmd.append("--pdb-grow-until-covers")
    if pdb_cover_hard_cap is not None:
        cmd.extend(["--pdb-cover-hard-cap", str(pdb_cover_hard_cap)])
    if rollout_aligned_h is not None:
        cmd.extend(["--rollout-aligned-h", str(rollout_aligned_h)])
    if rollout_aligned_redo is not None:
        cmd.extend(["--rollout-aligned-redo", str(rollout_aligned_redo)])
    if rollout_aligned_policy is not None:
        cmd.extend(["--rollout-aligned-policy", str(rollout_aligned_policy)])
    if rollout_aligned_cache:
        cmd.append("--rollout-aligned-cache")
    if rollout_aligned_max_rollouts_per_node is not None:
        cmd.extend(["--rollout-aligned-max-rollouts-per-node", str(rollout_aligned_max_rollouts_per_node)])
    if rollout_aligned_max_rollouts_per_search is not None:
        cmd.extend(["--rollout-aligned-max-rollouts-per-search", str(rollout_aligned_max_rollouts_per_search)])
    if rollout_aligned_max_time_per_search is not None:
        cmd.extend(["--rollout-aligned-max-time-per-search", str(rollout_aligned_max_time_per_search)])
    if rollout_aligned_fallback is not None:
        cmd.extend(["--rollout-aligned-fallback", str(rollout_aligned_fallback)])
    if rollout_aligned_fixed_h:
        cmd.append("--rollout-aligned-fixed-h")
    if rollout_aligned_boundary_mode is not None:
        cmd.extend(["--rollout-aligned-boundary-mode", str(rollout_aligned_boundary_mode)])
    if rollout_aligned_min_dynamic_horizon is not None:
        cmd.extend(["--rollout-aligned-min-dynamic-horizon", str(rollout_aligned_min_dynamic_horizon)])
    if rollout_aligned_fallback_if_small is not None:
        cmd.extend(["--rollout-aligned-fallback-if-small", str(rollout_aligned_fallback_if_small)])
    if rollout_aligned_lambda_align is not None:
        cmd.extend(["--rollout-aligned-lambda-align", str(rollout_aligned_lambda_align)])
    if frontier_option_a_debug:
        cmd.append("--frontier-option-a-debug")
    if ptrpg_guided_rollout_policy is not None:
        cmd.extend(["--ptrpg-guided-rollout-policy", str(ptrpg_guided_rollout_policy)])
    if ptrpg_guided_rollout_max_steps is not None:
        cmd.extend(["--ptrpg-guided-rollout-max-steps", str(ptrpg_guided_rollout_max_steps)])
    if ptrpg_guided_rollout_epsilon is not None:
        cmd.extend(["--ptrpg-guided-rollout-epsilon", str(ptrpg_guided_rollout_epsilon)])
    if ptrpg_guided_rollout_debug:
        cmd.append("--ptrpg-guided-rollout-debug")
    if fixed_tail_prefix_frac is not None:
        cmd.extend(["--fixed-tail-prefix-frac", str(fixed_tail_prefix_frac)])
    if fixed_tail_prefix_policy is not None:
        cmd.extend(["--fixed-tail-prefix-policy", str(fixed_tail_prefix_policy)])
    if fixed_tail_debug:
        cmd.append("--fixed-tail-debug")
    if fixed_tail_expectimax_max_nodes is not None:
        cmd.extend(["--fixed-tail-expectimax-max-nodes", str(fixed_tail_expectimax_max_nodes)])
    if fixed_tail_expectimax_max_depth is not None:
        cmd.extend(["--fixed-tail-expectimax-max-depth", str(fixed_tail_expectimax_max_depth)])
    if fixed_tail_expectimax_max_time_sec is not None:
        cmd.extend(["--fixed-tail-expectimax-max-time-sec", str(fixed_tail_expectimax_max_time_sec)])
    if fixed_tail_rollout_samples is not None:
        cmd.extend(["--fixed-tail-rollout-samples", str(fixed_tail_rollout_samples)])
    if fixed_tail_rollout_policy is not None:
        cmd.extend(["--fixed-tail-rollout-policy", str(fixed_tail_rollout_policy)])
    if extra_args:
        cmd.extend(extra_args)

    if verbose:
        print(f"  CMD: {' '.join(cmd)}", flush=True)

    # Prefer the repo's `unified_planning` over any same-named site-packages install.
    env = os.environ.copy()
    repo = str(REPO_ROOT)
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo + os.pathsep + prev_pp if prev_pp else repo

    if workers > 1 and runs > 1:
        return _run_parallel(cmd, env, runs=runs, seed=seed, workers=workers, verbose=verbose)
    return _run_cmd(cmd, env)


def _run_cmd(cmd: list[str], env: dict[str, str]) -> tuple[str, int]:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    output = proc.stdout + proc.stderr
    return output, proc.returncode


# ---------------------------------------------------------------------------
# Parallel runs: one run_domain.py per worker, merged afterwards
# ---------------------------------------------------------------------------

# Worker i runs with seed + i * PARALLEL_SEED_STRIDE. Worker 0 keeps the base seed,
# so its episodes are exactly the first episodes of the one-process run; the others
# get their own random streams (the same seed everywhere would repeat the episodes).
PARALLEL_SEED_STRIDE = 1000


def physical_cores() -> int:
    """Physical cores (MCTS spends wall-clock time per decision, so two workers on
    one core's hyperthreads get fewer iterations each). psutil if installed, else
    the (physical id, core id) pairs of /proc/cpuinfo, else half the logical CPUs."""
    try:
        import psutil  # type: ignore

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    try:
        pairs, phys = set(), None
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    pairs.add((phys, line.split(":", 1)[1].strip()))
        if pairs:
            return len(pairs)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def split_runs(runs: int, workers: int) -> list[int]:
    """Runs per worker, as even as possible; never more workers than runs."""
    n = max(1, min(int(workers), int(runs)))
    base, extra = divmod(int(runs), n)
    return [base + (1 if i < extra else 0) for i in range(n)]


def _set_arg(cmd: list[str], flag: str, value: Any) -> list[str]:
    out = list(cmd)
    out[out.index(flag) + 1] = str(value)
    return out


def _run_parallel(cmd: list[str], env: dict[str, str], *, runs: int, seed: int, workers: int,
                  verbose: bool) -> tuple[str, int]:
    from concurrent.futures import ThreadPoolExecutor

    sizes = split_runs(runs, workers)
    seeds = [int(seed) + i * PARALLEL_SEED_STRIDE for i in range(len(sizes))]
    cmds = []
    for i, (size, worker_seed) in enumerate(zip(sizes, seeds)):
        c = _set_arg(_set_arg(cmd, "--runs", size), "--seed", worker_seed)
        if "--max-approx-seed" in c:
            base = int(c[c.index("--max-approx-seed") + 1])
            c = _set_arg(c, "--max-approx-seed", base + i * PARALLEL_SEED_STRIDE)
        cmds.append(c)
    if verbose:
        print(f"  parallel: {len(sizes)} workers, runs per worker {sizes}, seeds {seeds[0]}.."
              f"{seeds[-1]} (step {PARALLEL_SEED_STRIDE})", flush=True)
    # Threads only wait on the child processes; the work runs in the processes.
    with ThreadPoolExecutor(max_workers=len(cmds)) as pool:
        results = list(pool.map(lambda c: _run_cmd(c, env), cmds))
    returncode = next((rc for _out, rc in results if rc != 0), 0)
    return merge_run_domain_outputs([out for out, _rc in results], seeds=seeds), returncode


_SUMMARY_LINE = re.compile(r"^(Completed|Amount of success|Average success time|STD success time)\s*=\s*(\S+)\s*$")


def merge_run_domain_outputs(outputs: list[str], seeds: list[int] | None = None) -> str:
    """One run_domain.py output per worker -> one output whose four summary lines are
    what ``evaluate.evaluation_loop`` prints for all the episodes together:

        Completed = runs;  Amount of success = S;  Average success time = mean of the
        S success times (-inf if S = 0);  STD success time = stdev(times) / sqrt(S)
        (-1 if S <= 1).

    From worker i (s_i successes, mean m_i, printed std q_i): sample sd_i = q_i sqrt(s_i),
    M = sum s_i m_i / S, SS = sum (s_i - 1) sd_i^2 + s_i (m_i - M)^2, stdev = sqrt(SS/(S-1)).
    Each worker's own summary lines are renamed ("[worker i] ...") so that
    ``parse_run_metrics`` only finds the merged ones, which come last.
    """
    import math

    parts, done = [], []
    for i, text in enumerate(outputs):
        found: dict[str, float] = {}
        lines = []
        for line in text.splitlines():
            m = _SUMMARY_LINE.match(line.strip())
            if m:
                found[m.group(1)] = float(m.group(2))
                lines.append(f"[worker {i}] {m.group(1).lower()}: {m.group(2)}")
            else:
                lines.append(line)
        seed_note = f" seed={seeds[i]}" if seeds else ""
        parts.append(f"===== worker {i}{seed_note} =====\n" + "\n".join(lines))
        if "Completed" in found and "Amount of success" in found:
            done.append((int(found["Completed"]), int(found["Amount of success"]),
                         found.get("Average success time", -math.inf), found.get("STD success time", -1.0)))
        else:
            parts.append(f"[parallel] worker {i} printed no summary (crashed?); its runs are not counted")

    total = sum(n for n, _s, _m, _q in done)
    succ = sum(s for _n, s, _m, _q in done)
    mean = sum(s * m for _n, s, m, _q in done if s > 0) / succ if succ > 0 else -math.inf
    std: float = -1
    if succ > 1:
        ss = sum((s - 1) * (q * math.sqrt(s)) ** 2 + s * (m - mean) ** 2
                 for _n, s, m, q in done if s > 0 and (s == 1 or q >= 0))
        std = math.sqrt(ss / (succ - 1)) / math.sqrt(succ)
    parts.append(f"===== merged over {len(outputs)} workers =====\n"
                 f"Completed = {total}\n"
                 f"Amount of success = {succ}\n"
                 f"Average success time = {mean}\n"
                 f"STD success time = {std}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Metric parsing from run_domain.py stdout
# ---------------------------------------------------------------------------

_METRIC_PATTERNS: dict[str, re.Pattern] = {
    "runs_total": re.compile(r"Completed\s*=\s*(\S+)"),
    "amount_success": re.compile(r"Amount of success\s*=\s*(\S+)"),
    "avg_success_time": re.compile(r"Average success time\s*=\s*(\S+)"),
    "std_success_time": re.compile(r"STD success time\s*=\s*(\S+)"),
}


def summarize_run_domain_output(output: str, *, max_lines: int = 24) -> None:
    """Print high-signal run_domain.py lines (for Script 3 when verbose=False)."""
    keys = (
        "Solver =",
        "Selection Type =",
        "Domain Type =",
        "Temporal Heuristic Strategy =",
        "Max approx",
        "Resolution alpha",
        "Compilation Time",
        "Action amount=",
        "A valid plan",
        "Amount of success",
        "Traceback",
        "Error",
        "ModuleNotFoundError",
        "[windows_ilao_pdb]",
    )
    hits = [ln for ln in output.splitlines() if any(k in ln for k in keys)]
    if hits:
        print("  --- run_domain (key lines) ---")
        for ln in hits[-max_lines:]:
            print("  " + ln)
        return
    tail = [ln for ln in output.splitlines() if ln.strip()][-8:]
    if tail:
        print("  --- run_domain (tail; no key lines matched) ---")
        for ln in tail:
            print("  " + ln)


def parse_run_metrics(output: str) -> dict[str, Any]:
    """Extract key metrics from run_domain.py stdout."""
    result: dict[str, Any] = {}
    for key, pattern in _METRIC_PATTERNS.items():
        m = pattern.search(output)
        if m:
            raw = m.group(1)
            try:
                result[key] = int(raw)
            except ValueError:
                try:
                    result[key] = float(raw)
                except ValueError:
                    result[key] = raw
        else:
            result[key] = None

    # Derived: success rate as float 0..1
    amt = result.get("amount_success")
    tot = result.get("runs_total")
    if amt is not None and tot is not None and tot > 0:
        try:
            result["success_rate"] = round(int(amt) / int(tot), 4)
        except (TypeError, ValueError):
            result["success_rate"] = None
    else:
        result["success_rate"] = None

    return result


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str] | None = None) -> None:
    """Write a list of dicts to a CSV file, auto-detecting columns if not given."""
    if not rows:
        print(f"[warn] No rows to write to {path}")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Pretty terminal table
# ---------------------------------------------------------------------------

def print_summary_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Print a simple fixed-width table to stdout."""
    col_widths = [max(len(str(c)), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in columns]
    sep = "  ".join("-" * w for w in col_widths)
    header = "  ".join(str(c).ljust(w) for c, w in zip(columns, col_widths))
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(columns, col_widths))
        print(line)
