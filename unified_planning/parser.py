import argparse


def _parse_temporal_heuristic_strategy(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "1": "baseline",
        "2": "atom_half_split",
        "3": "atom_backtrack_exact",
        "4": "baseline_cached",
        "5": "atom_backtrack_cached",
        "6": "fast_atom_cache",
        "7": "atom_backtrack_exact_resolution",
        "8": "atom_backtrack_exact_unbiased",
        "9": "baseline_survival",
        "10": "baseline_survival_meanvar",
        "11": "baseline_survival_and_gamma",
        "12": "atom_backtrack_exact_resolution_and_gamma",
        "13": "baseline_survival_resolution",
        "14": "rollout_aligned_baseline",
        "15": "rollout_aligned_survival",
        "16": "rollout_aligned_resolution_survival",
        "17": "frontier_aligned_baseline",
        "18": "frontier_aligned_survival",
        "19": "frontier_aligned_resolution_survival",
        "20": "frontier_aligned_option_a",
        "21": "frontier_aligned_option_a_survival",
        "22": "frontier_aligned_option_a_resolution",
        "23": "baseline_time_to_goal",
        "24": "baseline_pdb",
        "25": "baseline_admissible",
        "26": "baseline_admissible_lp",
        "27": "baseline_admissible_kmutex",
        "28": "baseline_admissible_paths",
        "29": "baseline_admissible_paths_table",
        "30": "baseline_admissible_resolution",
        "31": "baseline_admissible_resolution_forward",
        "32": "baseline_forward",
        "33": "baseline_admissible_survivor_pdb",
        "baseline": "baseline",
        "baseline_forward": "baseline_forward",
        "baseline_admissible": "baseline_admissible",
        "baseline_admissible_resolution": "baseline_admissible_resolution",
        "baseline_admissible_resolution_forward": "baseline_admissible_resolution_forward",
        "baseline_admissible_lp": "baseline_admissible_lp",
        "baseline_admissible_kmutex": "baseline_admissible_kmutex",
        "baseline_admissible_paths": "baseline_admissible_paths",
        "baseline_admissible_paths_table": "baseline_admissible_paths_table",
        "baseline_admissible_survivor_pdb": "baseline_admissible_survivor_pdb",
        "baseline_pdb": "baseline_pdb",
        "atom_half_split": "atom_half_split",
        "atom_backtrack_exact": "atom_backtrack_exact",
        "baseline_cached": "baseline_cached",
        "atom_backtrack_cached": "atom_backtrack_cached",
        "fast_atom_cache": "fast_atom_cache",
        "atom_backtrack_exact_resolution": "atom_backtrack_exact_resolution",
        "atom_backtrack_exact_unbiased": "atom_backtrack_exact_unbiased",
        "baseline_survival": "baseline_survival",
        "baseline_survival_meanvar": "baseline_survival_meanvar",
        "baseline_survival_and_gamma": "baseline_survival_and_gamma",
        "atom_backtrack_exact_resolution_and_gamma": "atom_backtrack_exact_resolution_and_gamma",
        "baseline_survival_resolution": "baseline_survival_resolution",
        "rollout_aligned_baseline": "rollout_aligned_baseline",
        "rollout_aligned_survival": "rollout_aligned_survival",
        "rollout_aligned_resolution_survival": "rollout_aligned_resolution_survival",
        "frontier_aligned_baseline": "frontier_aligned_baseline",
        "frontier_aligned_survival": "frontier_aligned_survival",
        "frontier_aligned_resolution_survival": "frontier_aligned_resolution_survival",
        "frontier_aligned_option_a": "frontier_aligned_option_a",
        "frontier_aligned_option_a_survival": "frontier_aligned_option_a_survival",
        "frontier_aligned_option_a_resolution": "frontier_aligned_option_a_resolution",
        "baseline_time_to_goal": "baseline_time_to_goal",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "temporal_heuristic_strategy must be one of: "
            "1|baseline, 2|atom_half_split, 3|atom_backtrack_exact, "
            "4|baseline_cached, 5|atom_backtrack_cached, 6|fast_atom_cache, "
            "7|atom_backtrack_exact_resolution, "
            "8|atom_backtrack_exact_unbiased, "
            "9|baseline_survival, "
            "10|baseline_survival_meanvar, "
            "11|baseline_survival_and_gamma, "
            "12|atom_backtrack_exact_resolution_and_gamma, "
            "13|baseline_survival_resolution, "
            "14|rollout_aligned_baseline, "
            "15|rollout_aligned_survival, "
            "16|rollout_aligned_resolution_survival, "
            "17|frontier_aligned_baseline, "
            "18|frontier_aligned_survival, "
            "19|frontier_aligned_resolution_survival, "
            "20|frontier_aligned_option_a, "
            "21|frontier_aligned_option_a_survival, "
            "22|frontier_aligned_option_a_resolution, "
            "23|baseline_time_to_goal, "
            "24|baseline_pdb, "
            "25|baseline_admissible, "
            "26|baseline_admissible_lp, "
            "27|baseline_admissible_kmutex, "
            "28|baseline_admissible_paths, "
            "29|baseline_admissible_paths_table, "
            "30|baseline_admissible_resolution, "
            "31|baseline_admissible_resolution_forward, "
            "32|baseline_forward, "
            "33|baseline_admissible_survivor_pdb"
        )
    return aliases[normalized]


parser = argparse.ArgumentParser(description='Description of your script')
parser.add_argument('-d', '--deadline', help='deadline of the problem', nargs='?', default=None, type=int)
parser.add_argument('-st', '--search_time', help='amount of time in each step', nargs='?', default=1, type=int)
parser.add_argument('-sd', '--search_depth', help='search depth of ', nargs='?', default=40, type=int)
parser.add_argument(
    '-se',
    '--selection_type',
    help=(
        'MCTS: avg|rootInterval|max. greedy_parallel: avg (default greedy) or '
        'max_approximation (stochastic parallel action-set picker).'
    ),
    nargs='?',
    default='avg',
)
parser.add_argument(
    '--max-approx-alpha',
    dest='max_approx_alpha',
    default=1.0,
    type=float,
    help='max_approximation: sampling exponent P(a) ∝ score(a)^alpha (greedy_parallel only).',
)
parser.add_argument(
    '--max-approx-samples',
    dest='max_approx_num_samples',
    default=2,
    type=int,
    help='max_approximation: candidate action sets per dispatch (1=deterministic greedy; '
         '+1 stochastic exploration each). Each costs a full greedy build.',
)
parser.add_argument(
    '--max-approx-seed',
    dest='max_approx_seed',
    default=None,
    type=int,
    help='max_approximation: RNG seed for set sampling (default: --seed).',
)
parser.add_argument(
    '--max-approx-debug',
    dest='max_approx_debug',
    action='store_true',
    default=False,
    help='max_approximation: print sampled sets and rejections.',
)
parser.add_argument('-r', '--runs', help='how many runs to run the script', nargs='?', default=1, type=int)
parser.add_argument('-dt', '--domain_type', help='combination, new approach or new approach same as the baseline', nargs='?', default='regular')
parser.add_argument(
    '-s',
    '--solver',
    help='solver: mcts, rtdp, greedy_parallel, or heuristic_tree',
    nargs='?',
    default='mcts',
)
parser.add_argument('-do', '--domain', help='domain')
parser.add_argument('-e', '--exploration_constant', help='the exploration constant for mcts solver', nargs='?', default=10.0, type=float)
parser.add_argument('-ge', '--garbage_amount', help='how many garbage actions to add to the domain', nargs='?', default=0, type=int)
parser.add_argument('-oe', '--object_amount', help='how many different objects in the domain', nargs='?', default=1, type=int)
parser.add_argument('-k', '--k', help='K random actions in the max planner', nargs='?', default=10, type=int)
parser.add_argument('-rm', '--reward_mode', help='reward mode: deadline or terminal', nargs='?', default='deadline')
parser.add_argument(
    '--discount_factor',
    '--gamma',
    dest='discount_factor',
    help='MDP discount factor (gamma) for MCTS discounted backups',
    nargs='?',
    default=0.95,
    type=float,
)
parser.add_argument(
    '--step_penalty',
    dest='step_penalty',
    help='Per-step reward shaping penalty added on each MDP transition',
    nargs='?',
    default=-0.05,
    type=float,
)
parser.add_argument('--seed', help='random seed for reproducibility', nargs='?', default=None, type=int)
parser.add_argument(
    '--heuristic_name',
    help=(
        'leaf heuristic: trpg | temporal_probabilistic_rpg | '
        'baseline_pessimistic | baseline_passmistic | baseline_optimistic | baseline_optimstic'
    ),
    nargs='?',
    default='trpg',
)
parser.add_argument(
    '--temporal_heuristic_depth',
    help=(
        'Temporal lookahead depth for temporal_probabilistic_rpg / TP-MCTS. '
        'Default: same as --deadline (or 25 if deadline is unset).'
    ),
    nargs='?',
    default=None,
    type=int,
)
parser.add_argument(
    '--temporal_heuristic_strategy',
    help=(
        'strategy for temporal_probabilistic_rpg: '
        '1|baseline (layered), '
        '2|atom_half_split (approximate half-interval on eligible atoms), '
        '3|atom_backtrack_exact (memoized exact atom backtrack vs horizon), '
        '4|baseline_cached (incremental dirty-fact table updates), '
        '5|atom_backtrack_cached (persistent formula memos with selective invalidation), '
        '6|fast_atom_cache (schedule-ordered backtrack + cross-call lazy fact/action memos), '
        '7|atom_backtrack_exact_resolution (anchor-based backtrack with reorganized deltas), '
        '8|atom_backtrack_exact_unbiased (resolution + structural per-layer bias correction; '
        'NOT admissible), '
        '9|baseline_survival (layered baseline + delete/survival decay so P_t can drop; '
        'NOT monotone), '
        '10|baseline_survival_meanvar (baseline_survival + variance-aware goal '
        'aggregation mean-alpha*sqrt(k-1)*std over per-goal areas), '
        '11|baseline_survival_and_gamma (baseline_survival + component-wise AND-layer '
        'gamma correction on the precondition support; reduces to baseline_survival '
        'when no dependency is detected; NOT calibrated probability), '
        '12|atom_backtrack_exact_resolution_and_gamma (resolution backtrack with '
        'log-spaced/exponential-width layers + the same AND-layer gamma correction; '
        'reduces to atom_backtrack_exact_resolution when no dependency is detected), '
        '25|baseline_admissible (layered baseline with Frechet-min AND and union-bound '
        'OR instead of the independence product / noisy-OR; very optimistic but '
        'ADMISSIBLE upper bound), '
        '26|baseline_admissible_lp (baseline_admissible with the OR layer replaced by '
        'the marginal-consistent LP bound: per arrival fact, max probability of the '
        'OR-of-AND achiever formula over all local joints consistent with the stored '
        'marginals; still ADMISSIBLE but tighter than the union bound when achievers '
        'share preconditions; falls back to the union bound when |local facts| > '
        'TP_MCTS_ADMISSIBLE_LP_MAX_LOCAL_FACTS), '
        '27|baseline_admissible_kmutex (baseline_admissible with the OR layer '
        'tightened by a mutex-aware K-bounded bound: per arrival fact keep <= K '
        'rows by certified action-mutex and aggregate sum(free rows)+max(mutex '
        'clique); always <= the union bound, reduces to baseline_admissible when '
        'no achievers are mutex; K via TP_MCTS_KMUTEX_K), '
        '28|baseline_admissible_paths (temporal path-mutex tightening: carry <= K '
        'timed achiever-paths per fact and combine by segment-overlap mutex -- '
        'alternative paths sharing a mutex action in overlapping windows (incl. an '
        'action mutex with itself when it occupies a resource) collapse via max '
        'instead of summing as independent retries; an AND path is dropped when its '
        'achievers cannot run in parallel; K via TP_MCTS_KMUTEX_K), '
        '29|baseline_admissible_paths_table (table-flowing paths: K-row achiever-path '
        'tables flow through the RPG so the AND layer and goal score can apply '
        'cross-fact temporal-mutex bounds via kernelized aggregation; ADMISSIBLE; '
        'K via TP_MCTS_KMUTEX_K, aggregation via TP_MCTS_HEURISTIC_AGGREGATION), '
        '30|baseline_admissible_resolution (baseline_admissible operators -- Frechet-min '
        'AND + union-bound OR -- computed as a goal-directed BACKWARD recursion that '
        'evaluates achievers only at the 2^(k/2) resolution anchors; the skipped '
        'completion layers are NOT dropped but charged n_b times at the block latest '
        'time, so it stays an ADMISSIBLE upper bound (looser than dense '
        'baseline_admissible); resolution knobs via --resolution-alpha etc.), '
        '31|baseline_admissible_resolution_forward (FORWARD anchor-jump analogue of 30: '
        'incremental forward sweep over only the 2^(k/2) anchors, jumping P with the '
        'block closed form P=1-(1-P)(1-H)^n_b where H is the union hazard copied over '
        'the n_b skipped earlier layers -- end-of-block anchors keep it ADMISSIBLE and '
        'tighter than 30; computes all facts (no goal scoping); resolution knobs via '
        '--resolution-alpha etc.), '
        '32|baseline_forward (EVENT-DRIVEN forward expansion with independence '
        'operators: at each anchor delta_t expand all applicable actions, land each '
        'effect at delta_t+d(a), then jump to the NEXT scheduled arrival time -- steps '
        'by the action durations, NOT unit layers. Retry recursion carries the previous '
        'anchor P(delta_t)=P(delta_{t-1})+(1-P(delta_{t-1}))H_t with noisy-AND product '
        'R(a)=prod_f P(f) and noisy-OR H_t(f)=1-prod_e(1-B_e). Coarse grid re-fires '
        'actions once per anchor (serialized), so everywhere <= baseline/baseline_admissible '
        '-- NOT admissible)'
    ),
    nargs='?',
    default='baseline',
    type=_parse_temporal_heuristic_strategy,
)

parser.add_argument(
    '--resolution-alpha',
    dest='resolution_alpha',
    default=2.0,
    type=float,
    help='atom_backtrack_exact_resolution: base alpha in raw width alpha^floor(k/2) (default 2).',
)
parser.add_argument(
    '--resolution-forced-minimum',
    dest='resolution_forced_minimum',
    action='store_true',
    default=False,
    help=(
        'atom_backtrack_exact_resolution: use normalized raw widths '
        '(alpha^floor(k/2) * remaining / T) with T from --resolution-reference-t '
        'or remaining; layer count is derived from the schedule loop.'
    ),
)
parser.add_argument(
    '--resolution-reference-t',
    dest='resolution_reference_t',
    default=None,
    type=int,
    metavar='T',
    help=(
        'atom_backtrack_exact_resolution: reference horizon T for raw scaling '
        '(default: same as remaining depth for that evaluation).'
    ),
)

parser.add_argument(
    '--and-gamma-rollout-calibration',
    dest='and_gamma_rollout_calibration',
    action='store_true',
    default=False,
    help=(
        'baseline_survival_and_gamma: enable lazy rollout calibration of the '
        'AND-layer gamma factors (default off = static gamma table).'
    ),
)

# baseline_pdb: horizon-indexed Pattern Database AND-layer correction knobs.
parser.add_argument(
    '--pdb-num-patterns',
    dest='pdb_num_patterns',
    default=4,
    type=int,
    help='baseline_pdb: number of goal-directed patterns to build (default 4).',
)
parser.add_argument(
    '--pdb-max-facts-per-pattern',
    dest='pdb_max_facts_per_pattern',
    default=4,
    type=int,
    help='baseline_pdb: max facts per pattern; PDB DP cost grows with this (default 4).',
)
parser.add_argument(
    '--pdb-expansion-policy',
    dest='pdb_expansion_policy',
    default='max_prob',
    choices=['max_prob', 'random'],
    help='baseline_pdb: pattern-growth achiever choice (default max_prob).',
)
parser.add_argument(
    '--pdb-no-seed-per-goal',
    dest='pdb_seed_per_goal',
    action='store_false',
    default=True,
    help='baseline_pdb: seed each pattern from the whole goal set instead of one '
         'goal each (default: one per goal — required for coverage on multi-goal problems).',
)
parser.add_argument(
    '--pdb-grow-until-covers',
    dest='pdb_grow_until_covers',
    action='store_true',
    default=False,
    help='baseline_pdb: keep growing each pattern past max-facts until it fully '
         'covers at least one action (guarantees the pattern is actually used).',
)
parser.add_argument(
    '--pdb-cover-hard-cap',
    dest='pdb_cover_hard_cap',
    default=None,
    type=int,
    help='baseline_pdb: max pattern size when --pdb-grow-until-covers is set '
         '(default: max-facts + 12).',
)

# Rollout-aligned common-horizon PTRPG (strategies 14/15/16). These wrap the
# underlying PTRPG suffix with a real prefix rollout up to the common horizon H.
parser.add_argument(
    '--rollout-aligned-h',
    dest='rollout_aligned_h',
    default=15,
    type=int,
    metavar='H',
    help='rollout_aligned_*: common suffix horizon H (default 15). Try 5/10/15/20.',
)
parser.add_argument(
    '--rollout-aligned-redo',
    dest='rollout_aligned_redo',
    default=1,
    type=int,
    help='rollout_aligned_*: number of prefix rollouts averaged per node (default 1). Try 1/5/10/20.',
)
parser.add_argument(
    '--rollout-aligned-policy',
    dest='rollout_aligned_policy',
    default='random',
    help='rollout_aligned_*: prefix rollout policy (default random).',
)
parser.add_argument(
    '--rollout-aligned-cache',
    dest='rollout_aligned_cache',
    action='store_true',
    default=False,
    help='rollout_aligned_*: cache aligned values (sample estimates; default off).',
)
parser.add_argument(
    '--rollout-aligned-max-rollouts-per-node',
    dest='rollout_aligned_max_rollouts_per_node',
    default=0,
    type=int,
    help='rollout_aligned_*: cap prefix rollouts per node (0 = unlimited).',
)
parser.add_argument(
    '--rollout-aligned-max-rollouts-per-search',
    dest='rollout_aligned_max_rollouts_per_search',
    default=0,
    type=int,
    help='rollout_aligned_*: cap prefix rollouts per search (0 = unlimited).',
)
parser.add_argument(
    '--rollout-aligned-max-time-per-search',
    dest='rollout_aligned_max_time_per_search',
    default=0.0,
    type=float,
    help='rollout_aligned_*: cap prefix-rollout seconds per search (0 = unlimited).',
)
parser.add_argument(
    '--rollout-aligned-fallback',
    dest='rollout_aligned_fallback',
    default='horizon_capped',
    choices=('horizon_capped', 'raw'),
    help='rollout_aligned_*: budget-exhausted fallback (default horizon_capped).',
)
parser.add_argument(
    '--rollout-aligned-fixed-h',
    dest='rollout_aligned_fixed_h',
    action='store_true',
    default=False,
    help=(
        'rollout_aligned_*: disable the dynamic parent-local horizon and use the '
        'fixed --rollout-aligned-h instead (old fixed-H behavior).'
    ),
)
parser.add_argument(
    '--rollout-aligned-boundary-mode',
    dest='rollout_aligned_boundary_mode',
    default='overshoot',
    choices=('overshoot', 'wait_no_overshoot', 'expected_stochastic_rounding'),
    help='rollout_aligned_*: prefix boundary handling (default overshoot).',
)
parser.add_argument(
    '--rollout-aligned-min-dynamic-horizon',
    dest='rollout_aligned_min_dynamic_horizon',
    default=None,
    type=int,
    help='rollout_aligned_*: safety floor on the dynamic horizon H_p (default off).',
)
parser.add_argument(
    '--rollout-aligned-fallback-if-small',
    dest='rollout_aligned_fallback_if_small',
    default='use_anyway',
    choices=('use_anyway', 'fixed', 'raw'),
    help='rollout_aligned_*: behavior when H_p < min-dynamic-horizon (default use_anyway).',
)
parser.add_argument(
    '--rollout-aligned-lambda-align',
    dest='rollout_aligned_lambda_align',
    default=None,
    type=float,
    help='rollout_aligned_*: blended-selection weight lambda_align in [0,1] (advisory).',
)
parser.add_argument(
    '--frontier-aligned-debug',
    dest='frontier_aligned_debug',
    action='store_true',
    default=False,
    help='frontier_aligned_*: print a per-candidate trace for the first few MCTS decisions.',
)
parser.add_argument(
    '--frontier-option-a-debug',
    dest='frontier_option_a_debug',
    action='store_true',
    default=False,
    help='frontier_aligned_option_a_*: print global frontier trace for first 3 selections.',
)

parser.add_argument(
    '--tree_depth',
    help='lookahead depth for heuristic_tree solver (separate from temporal_heuristic_depth)',
    nargs='?',
    default=3,
    type=int,
)
parser.add_argument(
    '--value_mode',
    help='MCTS leaf/backup target mode: tp_mcts (default), greedy_matched, '
         'ptrpg_guided_terminal_rollout, fixed_tail_mcts_sampled (in-tree UCT prefix), '
         'fixed_tail_random_rollout_eval (ephemeral K-rollout leaf eval), or '
         'fixed_tail_ptrpg_rollout (deprecated alias for mcts_sampled / expectimax)',
    nargs='?',
    default='tp_mcts',
    choices=(
        'tp_mcts',
        'greedy_matched',
        'ptrpg_guided_terminal_rollout',
        'fixed_tail_ptrpg_rollout',
        'fixed_tail_mcts_sampled',
        'fixed_tail_random_rollout_eval',
        'max_approximation',
    ),
)
parser.add_argument(
    '--fixed-tail-prefix-frac',
    dest='fixed_tail_prefix_frac',
    type=float,
    default=0.10,
    help='fixed_tail_ptrpg_rollout: prefix time budget as fraction of root remaining (default 0.10).',
)
parser.add_argument(
    '--fixed-tail-debug',
    dest='fixed_tail_debug',
    action='store_true',
    default=False,
    help='Log the first 5 fixed-tail bootstrap evaluations per MCTS search.',
)
parser.add_argument(
    '--fixed-tail-prefix-policy',
    dest='fixed_tail_prefix_policy',
    default='mcts_sampled',
    choices=('mcts_sampled', 'expectimax'),
    help='fixed_tail_ptrpg_rollout: prefix evaluation before cutoff — '
         'mcts_sampled (default) or expectimax (max/expectation, no sampling in prefix).',
)
parser.add_argument(
    '--fixed-tail-expectimax-max-nodes',
    dest='fixed_tail_expectimax_max_nodes',
    type=int,
    default=5000,
    help='expectimax prefix: max V/Q evaluations per MCTS search (guard fallback to PTRPG).',
)
parser.add_argument(
    '--fixed-tail-expectimax-max-depth',
    dest='fixed_tail_expectimax_max_depth',
    type=int,
    default=64,
    help='expectimax prefix: max recursion depth (guard fallback to PTRPG).',
)
parser.add_argument(
    '--fixed-tail-expectimax-max-time-sec',
    dest='fixed_tail_expectimax_max_time_sec',
    type=float,
    default=0.0,
    help='expectimax prefix: wall-time budget per search in seconds (0 = disabled).',
)
parser.add_argument(
    '--fixed-tail-rollout-samples',
    dest='fixed_tail_rollout_samples',
    type=int,
    default=1,
    help='fixed_tail_random_rollout_eval: ephemeral rollout samples K per leaf (default 1).',
)
parser.add_argument(
    '--fixed-tail-rollout-policy',
    dest='fixed_tail_rollout_policy',
    default='random_legal_fitting',
    choices=('random_legal_fitting', 'first_legal_fitting'),
    help='fixed_tail_random_rollout_eval: action choice during ephemeral prefix rollout.',
)
parser.add_argument(
    '--ptrpg-guided-rollout-policy',
    dest='ptrpg_guided_rollout_policy',
    default='baseline_survival_resolution',
    choices=(
        'baseline_survival_resolution',
        'atomic_exact_resolution',
        'atom_backtrack_exact_resolution',
    ),
    help='PTRPG strategy used only for greedy action choice inside ptrpg_guided_terminal_rollout.',
)
parser.add_argument(
    '--ptrpg-guided-rollout-max-steps',
    dest='ptrpg_guided_rollout_max_steps',
    type=int,
    default=None,
    help='Safety cap on MDP transitions in ptrpg_guided_terminal_rollout '
         '(default: 32 parallel slots × 90 time slices).',
)
parser.add_argument(
    '--ptrpg-guided-rollout-epsilon',
    dest='ptrpg_guided_rollout_epsilon',
    type=float,
    default=0.0,
    help='Reserved for future epsilon-greedy rollout policy (0 = pure greedy).',
)
parser.add_argument(
    '--ptrpg-guided-rollout-debug',
    dest='ptrpg_guided_rollout_debug',
    action='store_true',
    default=False,
    help='Log the first ptrpg_guided_terminal_rollout per MCTS search.',
)
parser.add_argument(
    '--final_selection',
    choices=('q', 'robust'),
    default='q',
    help='Final action selection after MCTS search: q=argmax Q-value (default), robust=most-visited (argmax N)',
)

# parse_known_args: importing unified_planning from tests/tools must not fail on
# unrelated argv tokens (e.g. pytest). run_domain.py is still invoked with known flags only.
args, __up_unknown_cli_args = parser.parse_known_args()
