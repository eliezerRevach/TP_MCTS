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
        "baseline": "baseline",
        "atom_half_split": "atom_half_split",
        "atom_backtrack_exact": "atom_backtrack_exact",
        "baseline_cached": "baseline_cached",
        "atom_backtrack_cached": "atom_backtrack_cached",
        "fast_atom_cache": "fast_atom_cache",
        "atom_backtrack_exact_resolution": "atom_backtrack_exact_resolution",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "temporal_heuristic_strategy must be one of: "
            "1|baseline, 2|atom_half_split, 3|atom_backtrack_exact, "
            "4|baseline_cached, 5|atom_backtrack_cached, 6|fast_atom_cache, "
            "7|atom_backtrack_exact_resolution"
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
        '7|atom_backtrack_exact_resolution (anchor-based backtrack with reorganized deltas)'
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
        'atom_backtrack_exact_resolution: derive K from min(K_target, reference_T) '
        'then cap by remaining (see temporal_probabilistic_rpg.build_resolution_delta_schedule).'
    ),
)
parser.add_argument(
    '--resolution-k-target',
    dest='resolution_k_target',
    default=8,
    type=int,
    help='atom_backtrack_exact_resolution: target layer count K_target (default 8).',
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
    '--tree_depth',
    help='lookahead depth for heuristic_tree solver (separate from temporal_heuristic_depth)',
    nargs='?',
    default=3,
    type=int,
)
parser.add_argument(
    '--value_mode',
    help='MCTS leaf/backup target mode: tp_mcts (default) or greedy_matched',
    nargs='?',
    default='tp_mcts',
    choices=('tp_mcts', 'greedy_matched'),
)

# parse_known_args: importing unified_planning from tests/tools must not fail on
# unrelated argv tokens (e.g. pytest). run_domain.py is still invoked with known flags only.
args, __up_unknown_cli_args = parser.parse_known_args()
