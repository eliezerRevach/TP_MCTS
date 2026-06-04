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
        "baseline": "baseline",
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
            "23|baseline_time_to_goal"
        )
    return aliases[normalized]


parser = argparse.ArgumentParser(description='Description of your script')
parser.add_argument('-d', '--deadline', help='deadline of the problem', nargs='?', default=None, type=int)
parser.add_argument('-st', '--search_time', help='amount of time in each step', nargs='?', default=1, type=int)
parser.add_argument('-sd', '--search_depth', help='search depth of ', nargs='?', default=40, type=int)
parser.add_argument('-se', '--selection_type', help='selection type in MCTS algorithm', nargs='?', default='avg')
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
        'reduces to atom_backtrack_exact_resolution when no dependency is detected)'
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
