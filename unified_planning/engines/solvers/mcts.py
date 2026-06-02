import numpy as np
from unified_planning.shortcuts import *
import unified_planning as up
import math
import time
import random
from unified_planning.engines.utils import (
    create_init_stn,
    update_stn,
)
from unified_planning.engines.linked_list import LinkedListNode
from comdp_plus_no_deadline.engines.frontier_aligned_option_a import (
    OPTION_A_STRATEGIES as _FRONTIER_ALIGNED_OPTION_A,
    aligned_value_for_node as _option_a_aligned_value,
    build_option_a_evaluator as _build_option_a_evaluator,
    collect_global_frontier as _collect_global_frontier,
    compute_H_frontier as _compute_H_frontier,
    format_option_a_debug_row as _format_option_a_debug_row,
    is_option_a_strategy as _is_option_a_strategy,
    option_a_ptrpg_suffix as _option_a_ptrpg_suffix,
    remaining_horizon as _option_a_remaining_horizon,
    select_frontier_node as _select_frontier_node,
)


def _effective_temporal_depth(configured_depth: int, current_time: float, deadline: float) -> int:
    configured_depth = max(0, int(configured_depth))
    if deadline is None:
        return configured_depth
    remaining = max(0, int(math.floor(deadline - current_time)))
    return min(configured_depth, remaining)


# User-requested names include typos; aliases map to the same implementation.
_CORR_PESSIMISTIC = frozenset({"baseline_pessimistic", "baseline_passmistic"})
_CORR_OPTIMISTIC = frozenset({"baseline_optimistic", "baseline_optimstic"})


def _resolution_heuristic_kwargs_from_cli() -> dict:
    """Optional kwargs for atom_backtrack_exact_resolution (from unified_planning.parser args)."""
    try:
        import unified_planning as up

        a = getattr(up, "args", None)
        if a is None:
            return {}
    except Exception:
        return {}
    ra = getattr(a, "resolution_alpha", 2.0)
    if ra is None:
        ra = 2.0
    else:
        ra = float(ra)
    return {
        "resolution_alpha": ra,
        "resolution_forced_minimum": bool(getattr(a, "resolution_forced_minimum", False)),
        "resolution_reference_t": getattr(a, "resolution_reference_t", None),
    }


def _aggregation_for_strategy(temporal_heuristic_strategy: str) -> str:
    """Pick the goal-aggregation for heuristic_score based on the strategy.

    `baseline_survival_meanvar` uses the variance-aware "meanvar" aggregation
    (mean - alpha*sqrt(k-1)*std over per-goal areas): a balance-sensitive
    direction that — unlike the final-layer "product" — is not gated by the
    slowest goal and gives a usable gradient. `baseline_survival` keeps the
    standard final-layer "product" score, so the two are directly comparable
    (identical survival propagation, different goal aggregation). Every other
    strategy also uses "product".
    """
    strat = (temporal_heuristic_strategy or "").strip().lower()
    if strat == "baseline_survival_meanvar":
        return "meanvar"
    if strat == "baseline_time_to_goal":
        # baseline propagation + deadline-normalized time-to-goal margin
        # V = 1 - t*/R (comparable across deadlines; see temporal_probabilistic_rpg).
        return "time_to_goal"
    return "product"


def _uses_tprpg_family(heuristic_name: str) -> bool:
    return (
        heuristic_name == "temporal_probabilistic_rpg"
        or heuristic_name in _CORR_PESSIMISTIC
        or heuristic_name in _CORR_OPTIMISTIC
    )


def _dynamic_aligned_horizon(heuristic_mdp, parent_snode):
    """Parent-local comparison horizon H_p = min over the parent's children of
    their remaining horizon = deadline - max(child STN end time).

    Each child action-node carries an STN whose current end time is that child's
    current_time, so this needs no extra stepping. Returns None when it cannot be
    computed (no parent / no deadline / no children)."""
    if parent_snode is None:
        return None
    deadline = heuristic_mdp.deadline()
    if deadline is None:
        return None
    children = getattr(parent_snode, "children", None)
    if not children:
        return None
    max_end = None
    for anode in children.values():
        stn = getattr(anode, "stn", None)
        if stn is None:
            continue
        try:
            end = stn.get_current_end_time()
        except Exception:
            continue
        if max_end is None or end > max_end:
            max_end = end
    if max_end is None:
        return None
    return max(0, int(math.floor(deadline - max_end)))


def _tprpg_heuristic_value(
    heuristic_mdp,
    state,
    current_time: float,
    temporal_heuristic_depth: int,
    temporal_heuristic_strategy: str,
    cached_table=None,
    leaf_heuristic_name: str = "temporal_probabilistic_rpg",
    aligned_h_override=None,
):
    """
    Evaluate the temporal_probabilistic_rpg heuristic, threading the baseline_cached
    table when applicable.  Returns (score_with_time_penalty, updated_cache_or_None).

    - For strategy == "baseline_cached": calls _temporal_heuristic with
      return_cache_table=True so the incremental propagation table can be
      forwarded to the next evaluation (matching the greedy_parallel pattern).
    - For all other strategies: calls _temporal_heuristic normally and returns
      None as the cache output (no table to forward).
    """
    if temporal_heuristic_strategy == "baseline_cached" and leaf_heuristic_name == "temporal_probabilistic_rpg":
        raw = _temporal_heuristic(
            heuristic_mdp, state, current_time,
            temporal_heuristic_depth, temporal_heuristic_strategy,
            cached_table=cached_table, return_cache_table=True,
            leaf_heuristic_name=leaf_heuristic_name,
        )
        if isinstance(raw, tuple):
            score, cache_out = raw
        else:
            score, cache_out = raw, None
    else:
        score = _temporal_heuristic(
            heuristic_mdp, state, current_time,
            temporal_heuristic_depth, temporal_heuristic_strategy,
            leaf_heuristic_name=leaf_heuristic_name,
            aligned_h_override=aligned_h_override,
        )
        cache_out = None
    return score - 0.001 * current_time, cache_out


# Rollout-aligned common-horizon PTRPG: maps each version to the underlying
# PTRPG suffix strategy used over the common horizon H.
_ROLLOUT_ALIGNED_SUFFIX = {
    "rollout_aligned_baseline": "baseline",
    "rollout_aligned_survival": "baseline_survival",
    "rollout_aligned_resolution_survival": "baseline_survival_resolution",
}

# Option A frontier-aligned SELECTION: same per-node aligned value as the
# rollout-aligned strategies, but the aligned value is used as a frontier
# selection score (blended with the backed-up Q via lambda_align) to decide which
# child to descend/expand. H_frontier = deadline - max(child elapsed) over the
# candidate child set. The original node is expanded; rollout endpoints are never
# inserted into the tree.
_FRONTIER_ALIGNED_SUFFIX = {
    "frontier_aligned_baseline": "baseline",
    "frontier_aligned_survival": "baseline_survival",
    "frontier_aligned_resolution_survival": "baseline_survival_resolution",
    # Explicit Option A names (global open-leaf frontier driver). Same global
    # frontier behavior as the frontier_aligned_* above.
    "frontier_aligned_option_a": "baseline",
    "frontier_aligned_option_a_survival": "baseline_survival",
    "frontier_aligned_option_a_resolution": "baseline_survival_resolution",
}

# Both families compute the per-node aligned value through the same evaluator.
_ALIGNED_SUFFIX = {**_ROLLOUT_ALIGNED_SUFFIX, **_FRONTIER_ALIGNED_SUFFIX}


def _uses_ptrpg_guided_rollout_value_mode(value_mode: str) -> bool:
    return value_mode == "ptrpg_guided_terminal_rollout"


def validate_ptrpg_guided_rollout_config(
    value_mode: str,
    temporal_heuristic_strategy: str,
) -> None:
    if not _uses_ptrpg_guided_rollout_value_mode(value_mode):
        return
    if temporal_heuristic_strategy in _ALIGNED_SUFFIX:
        raise ValueError(
            "value_mode=ptrpg_guided_terminal_rollout cannot be combined with "
            f"rollout_aligned / frontier_aligned strategy {temporal_heuristic_strategy!r}"
        )
    if _is_option_a_strategy(temporal_heuristic_strategy):
        raise ValueError(
            "value_mode=ptrpg_guided_terminal_rollout cannot be combined with "
            f"frontier_aligned_option_a strategy {temporal_heuristic_strategy!r}"
        )


def _uses_fixed_tail_ptrpg_rollout_value_mode(value_mode: str) -> bool:
    return value_mode == "fixed_tail_ptrpg_rollout"


def validate_fixed_tail_ptrpg_rollout_config(
    value_mode: str,
    temporal_heuristic_strategy: str,
) -> None:
    if not _uses_fixed_tail_ptrpg_rollout_value_mode(value_mode):
        return
    if temporal_heuristic_strategy in _ALIGNED_SUFFIX:
        raise ValueError(
            "value_mode=fixed_tail_ptrpg_rollout cannot be combined with "
            f"rollout_aligned / frontier_aligned strategy {temporal_heuristic_strategy!r}"
        )
    if _is_option_a_strategy(temporal_heuristic_strategy):
        raise ValueError(
            "value_mode=fixed_tail_ptrpg_rollout cannot be combined with "
            f"frontier_aligned_option_a strategy {temporal_heuristic_strategy!r}"
        )


def _rollout_aligned_config_from_cli():
    """Build a RolloutAlignedConfig from unified_planning.parser CLI args."""
    from comdp_plus_no_deadline.engines.rollout_aligned import RolloutAlignedConfig

    cfg = RolloutAlignedConfig()
    a = getattr(up, "args", None)
    if a is None:
        return cfg
    h = getattr(a, "rollout_aligned_h", None)
    if h is not None:
        cfg.common_horizon_H = int(h)
    redo = getattr(a, "rollout_aligned_redo", None)
    if redo is not None:
        cfg.redo = max(1, int(redo))
    policy = getattr(a, "rollout_aligned_policy", None)
    if policy:
        cfg.prefix_rollout_policy = str(policy)
    cfg.cache_aligned_values = bool(getattr(a, "rollout_aligned_cache", False))
    # Dynamic parent-local horizon knobs.
    cfg.use_dynamic_H = not bool(getattr(a, "rollout_aligned_fixed_h", False))
    boundary = getattr(a, "rollout_aligned_boundary_mode", None)
    if boundary:
        cfg.boundary_mode = str(boundary)
    min_dyn = getattr(a, "rollout_aligned_min_dynamic_horizon", None)
    if min_dyn is not None:
        cfg.min_dynamic_horizon = int(min_dyn)
    small_fb = getattr(a, "rollout_aligned_fallback_if_small", None)
    if small_fb:
        cfg.fallback_if_H_too_small = str(small_fb)
    lam = getattr(a, "rollout_aligned_lambda_align", None)
    if lam is not None:
        cfg.lambda_align = float(lam)
    mr_node = getattr(a, "rollout_aligned_max_rollouts_per_node", None)
    if mr_node is not None:
        cfg.max_prefix_rollouts_per_node = int(mr_node)
    mr_search = getattr(a, "rollout_aligned_max_rollouts_per_search", None)
    if mr_search is not None:
        cfg.max_prefix_rollouts_per_search = int(mr_search)
    mt = getattr(a, "rollout_aligned_max_time_per_search", None)
    if mt is not None:
        cfg.max_prefix_rollout_time_per_search = float(mt)
    fb = getattr(a, "rollout_aligned_fallback", None)
    if fb:
        cfg.fallback_mode = str(fb)
    return cfg


def _get_rollout_aligned_evaluator(
    heuristic_mdp, heuristic, suffix_strategy: str, force_pure_dynamic: bool = False
):
    """Build (once per MDP+suffix) a RolloutAlignedEvaluator bound to this MDP.

    The closures use the *real* MDP for the unmatched prefix (legal actions,
    durations, stochastic effects via mdp.step — exactly like ``simulate()``),
    and raw PTRPG for the common suffix horizon.

    ``force_pure_dynamic`` (used for frontier_aligned_*) pins the evaluator to the
    dynamic H_frontier and makes the fixed ROLLOUT_ALIGNED_H have no effect.
    """
    from comdp_plus_no_deadline.engines.rollout_aligned import (
        BOUNDARY_WAIT_NO_OVERSHOOT,
        RolloutAlignedEvaluator,
    )

    cache = getattr(heuristic_mdp, "_rollout_aligned_evaluators", None)
    if cache is None:
        cache = {}
        setattr(heuristic_mdp, "_rollout_aligned_evaluators", cache)
    cache_key = (suffix_strategy, bool(force_pure_dynamic))
    existing = cache.get(cache_key)
    if existing is not None:
        return existing

    cfg = _rollout_aligned_config_from_cli()
    if force_pure_dynamic:
        # frontier_aligned_*: always pure dynamic H_frontier. Ignore the fixed-H
        # switch entirely so ROLLOUT_ALIGNED_H / FIXED_H can never affect it.
        cfg.use_dynamic_H = True
    deadline = heuristic_mdp.deadline()
    goals = set(heuristic_mdp.problem.goals)
    res_kwargs = _resolution_heuristic_kwargs_from_cli()
    suffix_aggregation = _aggregation_for_strategy(suffix_strategy)

    def _is_goal(s):
        preds = getattr(s, "predicates", None)
        return preds is not None and goals.issubset(set(preds))

    def raw_eval_fn(s, horizon):
        ct = float(getattr(s, "current_time", 0.0))
        if deadline is not None:
            eff = min(int(horizon), max(0, int(math.floor(deadline - ct))))
        else:
            eff = int(horizon)
        return heuristic.heuristic_score(
            s,
            goals,
            aggregation=suffix_aggregation,
            fixed_depth=eff,
            start_time=ct,
            strategy=suffix_strategy,
            **res_kwargs,
        )

    def prefix_rollout_fn(s, delta):
        # Real MDP prefix rollout (legal actions, durations, stochastic effects,
        # STN feasibility via legal_actions/step). Returns dead_end=True when it
        # hits a no-legal-action / terminal-without-goal state before the
        # boundary (value 0.0). Boundary handling: "wait_no_overshoot" never
        # commits an action that would advance past the boundary (waits instead);
        # otherwise the last action may overshoot.
        sim = s
        start_ct = float(getattr(sim, "current_time", 0.0))
        reached = False
        dead_end = False
        length = 0
        no_overshoot = cfg.boundary_mode == BOUNDARY_WAIT_NO_OVERSHOOT
        while (float(getattr(sim, "current_time", 0.0)) - start_ct) < delta:
            legal = heuristic_mdp.legal_actions(sim)
            if not legal:
                dead_end = True
                break
            if no_overshoot:
                committed = False
                candidates = list(legal)
                random.shuffle(candidates)
                for action in candidates:
                    terminal, next_state, _r = heuristic_mdp.step(sim, action)
                    new_elapsed = float(getattr(next_state, "current_time", 0.0)) - start_ct
                    if new_elapsed <= delta:
                        sim = next_state
                        committed = True
                        length += 1
                        if _is_goal(sim):
                            reached = True
                        elif terminal:
                            dead_end = True
                        break
                if not committed:
                    break  # no fitting action -> wait at the boundary
                if reached or dead_end:
                    break
            else:
                action = random.choice(legal)
                terminal, next_state, _r = heuristic_mdp.step(sim, action)
                sim = next_state
                length += 1
                if _is_goal(sim):
                    reached = True
                    break
                if terminal:
                    dead_end = True
                    break
        return sim, reached, length, dead_end

    def state_hash_fn(s):
        preds = getattr(s, "predicates", None)
        ct = getattr(s, "current_time", 0.0)
        if preds is not None:
            return (frozenset(preds), ct)
        return (id(s), ct)

    evaluator = RolloutAlignedEvaluator(
        config=cfg,
        raw_eval_fn=raw_eval_fn,
        prefix_rollout_fn=prefix_rollout_fn,
        state_hash_fn=state_hash_fn,
    )
    cache[cache_key] = evaluator
    return evaluator


def _temporal_heuristic(
    heuristic_mdp,
    state,
    current_time: float,
    temporal_heuristic_depth: int,
    temporal_heuristic_strategy: str = "baseline",
    cached_table=None,
    return_cache_table: bool = False,
    leaf_heuristic_name: str = "temporal_probabilistic_rpg",
    aligned_h_override=None,
):
    from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
        TemporalProbabilisticRPGHeuristic,
    )
    from unified_planning.engines.heuristic_timing import WorkerTimer, is_active

    heuristic = getattr(heuristic_mdp, "_temporal_probabilistic_rpg_heuristic", None)
    if heuristic is None:
        heuristic = TemporalProbabilisticRPGHeuristic.from_problem(heuristic_mdp.problem)
        setattr(heuristic_mdp, "_temporal_probabilistic_rpg_heuristic", heuristic)

    # Optional: turn on AND-layer gamma rollout calibration from the CLI flag
    # (--and-gamma-rollout-calibration). Set before the first
    # baseline_survival_and_gamma query so the calibrator builds its simulator.
    try:
        _cli_args = getattr(up, "args", None)
        if _cli_args is not None and getattr(_cli_args, "and_gamma_rollout_calibration", False):
            _and_gamma_cfg = getattr(heuristic, "_and_gamma_config", None)
            if _and_gamma_cfg is not None:
                _and_gamma_cfg.enable_rollout_calibration = True
    except Exception:
        pass

    effective_depth = _effective_temporal_depth(
        temporal_heuristic_depth,
        current_time,
        heuristic_mdp.deadline(),
    )

    # Rollout-aligned common-horizon PTRPG: align this node's remaining horizon
    # to the shared suffix horizon H via real prefix rollouts, then score the
    # common suffix with the underlying PTRPG strategy. Intercept before the
    # plain heuristic_score path (these strategy names are MCTS-only and are not
    # propagation strategies of the heuristic itself).
    if temporal_heuristic_strategy in _ALIGNED_SUFFIX:
        suffix_strategy = _ALIGNED_SUFFIX[temporal_heuristic_strategy]
        # frontier_aligned_* always uses pure dynamic H_frontier (the fixed
        # ROLLOUT_ALIGNED_H must never affect it).
        is_frontier = temporal_heuristic_strategy in _FRONTIER_ALIGNED_SUFFIX
        evaluator = _get_rollout_aligned_evaluator(
            heuristic_mdp, heuristic, suffix_strategy, force_pure_dynamic=is_frontier
        )
        return evaluator.evaluate(
            state, effective_depth, h_override=aligned_h_override
        )

    if leaf_heuristic_name in _CORR_PESSIMISTIC:
        return heuristic.pessimistic_heuristic(
            state,
            goal_facts=heuristic_mdp.problem.goals,
            fixed_depth=effective_depth,
            start_time=current_time,
            problem_deadline=heuristic_mdp.deadline(),
        )
    if leaf_heuristic_name in _CORR_OPTIMISTIC:
        return heuristic.optimistic_heuristic(
            state,
            goal_facts=heuristic_mdp.problem.goals,
            fixed_depth=effective_depth,
            start_time=current_time,
            problem_deadline=heuristic_mdp.deadline(),
        )

    if is_active():
        cache_size_before = len(heuristic._query_cache)
        with WorkerTimer() as wt:
            result = heuristic.heuristic_score(
                state,
                heuristic_mdp.problem.goals,
                aggregation=_aggregation_for_strategy(temporal_heuristic_strategy),
                fixed_depth=effective_depth,
                start_time=current_time,
                strategy=temporal_heuristic_strategy,
                cached_table=cached_table,
                return_cache_table=return_cache_table,
                **_resolution_heuristic_kwargs_from_cli(),
            )
            # Cache hit: key already existed, so cache size did not grow.
            wt.hit = len(heuristic._query_cache) == cache_size_before
        return result

    return heuristic.heuristic_score(
        state,
        heuristic_mdp.problem.goals,
        aggregation=_aggregation_for_strategy(temporal_heuristic_strategy),
        fixed_depth=effective_depth,
        start_time=current_time,
        strategy=temporal_heuristic_strategy,
        cached_table=cached_table,
        return_cache_table=return_cache_table,
        **_resolution_heuristic_kwargs_from_cli(),
    )


class Base_MCTS:
    def __init__(self, mdp: "up.engines.MDP", search_depth: int,
                 exploration_constant: float, k: int):
        self._mdp = mdp
        self._search_depth = search_depth
        self._exploration_constant = exploration_constant
        self._root_node = None
        self._k = k

    @property
    def mdp(self):
        return self._mdp

    @property
    def root_node(self):
        return self._root_node

    @property
    def k(self):
        return self._k

    def root_state(self):
        return self.root_node.state

    @property
    def search_depth(self):
        return self._search_depth

    @property
    def exploration_constant(self):
        return self._exploration_constant

    def set_root_node(self, root_node):
        self._root_node = root_node

    def default_policy(self, state: "up.engines.State"):
        """ Choose a random action. Heustics can be used here to improve simulations. """
        return random.choice(self.mdp.legal_actions(state))

    def uct(self, snode: "up.engines.Snode", explore_constant: float):
        anodes = snode.children
        best_ub = -float('inf')
        best_action = -1
        possible_actions = snode.possible_actions
        for action in possible_actions:
            if anodes[action].count == 0:
                return action

            ub = anodes[action].value + (
                    explore_constant * math.sqrt(math.log(snode.count) / anodes[action].count))
            # ub = anodes[action].value + (
            #         explore_constant * math.sqrt(math.log(snode.count + 1) / anodes[action].count))
            if ub > best_ub:
                best_ub = ub
                best_action = action

        assert best_action != -1
        return best_action

    def best_action(self, root_node: "up.engines.SNode"):
        """

        :param root_node: the root node of the MCTS tree
        :return: returns the best action for the `root_node`
        """
        anodes = root_node.children
        aStart_value = float("-inf")
        aStar = -1

        for action in root_node.possible_actions:
            if anodes[action].count > 0 and anodes[action].value > aStart_value:
                aStart_value = anodes[action].value
                aStar = action

        if aStar == -1:
            print(4)

        return aStar

    def best_action_robust(self, root_node: "up.engines.SNode"):
        """Return the most-visited child (robust child / argmax-N)."""
        anodes = root_node.children
        best_count = -1
        aStar = -1
        for action in root_node.possible_actions:
            if anodes[action].count > best_count:
                best_count = anodes[action].count
                aStar = action
        return aStar

    def search(self, timeout=1, selection_type='avg', final_selection='q'):
        """
        Execute the MCTS algorithm from the initial state given, with timeout in seconds
        """
        start_time = time.time()
        current_time = time.time()
        i = 0
        if hasattr(self, "_ptrpg_rollout_debug_done"):
            self._ptrpg_rollout_debug_done = False
        if hasattr(self, "_fixed_tail_debug_count"):
            self._fixed_tail_debug_count = 0
        if getattr(self, "_fixed_tail_expectimax", None) is not None:
            self._fixed_tail_expectimax.reset_search()
        self._fixed_tail_root_debug_done = False
        avg_variants = {'avg', 'avg_topk', 'avg_pw'}
        selection = (
            self.selection
            if selection_type in avg_variants
            else (self.selection_root_interval if selection_type == 'rootInterval' else self.selection_max)
        )
        # Option A: frontier_aligned_* / frontier_aligned_option_a_* use a GLOBAL
        # open-leaf frontier driver instead of recursive parent-local UCT descent.
        strategy = getattr(self, "temporal_heuristic_strategy", None)
        use_frontier_driver = (
            strategy in _FRONTIER_ALIGNED_SUFFIX
            and hasattr(self, "_frontier_iteration")
        )
        use_option_a_driver = (
            _is_option_a_strategy(strategy)
            and hasattr(self, "_option_a_frontier_iteration")
        )
        while current_time < start_time + timeout:
            if use_option_a_driver:
                self._option_a_frontier_iteration()
            elif use_frontier_driver:
                self._frontier_iteration()
            else:
                selection(self.root_node)
            current_time = time.time()
            i += 1
        # print(f'i = {i}')
        if final_selection == 'robust':
            return self.best_action_robust(self.root_node)
        return self.best_action(self.root_node)

    def selection(self, snode: "up.engines.Snode"):
        raise NotImplementedError

    def selection_max(self, snode: "up.engines.Snode"):
        raise NotImplementedError

    def selection_root_interval(self, snode: "up.engines.Snode"):
        raise NotImplementedError
    def selection_root_interval_max(self, snode: "up.engines.Snode"):
        raise NotImplementedError

    def simulate(self, state, depth):
        raise NotImplementedError


class MCTS(Base_MCTS):
    """
    Original MCTS solver implementation.
    """
    def __init__(self, mdp: "up.engines.MDP", split_mdp: "up.engines.MDP", root_node: "up.engines.SNode",
                 root_state: "up.engines.state.State", search_depth: int,
                 exploration_constant: float, selection_type, k: int,
                 heuristic_name: str = 'trpg', temporal_heuristic_depth: int = 25,
                 temporal_heuristic_strategy: str = "baseline",
                 root_baseline_cache=None):
        super().__init__(mdp, search_depth, exploration_constant, k)
        self.split_mdp = split_mdp
        self.heuristic_name = heuristic_name
        self.temporal_heuristic_depth = temporal_heuristic_depth
        self.temporal_heuristic_strategy = temporal_heuristic_strategy
        self._root_baseline_cache = root_baseline_cache
        create_snode = self.create_Snode_max if selection_type == 'max' else self.create_Snode
        snode, _ = create_snode(root_state, 0)
        self.set_root_node(root_node if root_node is not None else snode)

    def create_Snode(self, state: "up.engines.State", depth: int,
                     parent: "up.engines.ANode" = None):
        """ Create a new Snode for the state `state` with parent `parent`"""
        return up.engines.SNode(state, depth, self.mdp.legal_actions(state), parent), None

    def create_Snode_max(self, state: "up.engines.State", depth: int,
                         parent: "up.engines.C_ANode" = None):
        """
        Create a new Snode for the state `state` with parent `parent`
        In this approach k children of snode are evaluated and the initiate value of snode is set to maximum value.

        """
        snode = up.engines.SNode(state, depth, self.mdp.legal_actions(state), parent)
        best = -math.inf

        actions_idx = list(range(len(snode.children)))
        if self.k < len(snode.children):
            # samples k children
            actions_idx = random.sample(range(0, len(snode.children)), self.k)

        for action_idx in actions_idx:
            # perform each action and evaluate the next state with the heuristic function
            action = list(snode.children.keys())[action_idx]
            terminal, next_state, reward = self.mdp.step(snode.state, action)
            reward += self.mdp.discount_factor * self.heuristic(next_state)
            snode.children[action].update(reward)
            if reward > best:
                best = reward
        if best == -math.inf:
            best = self.heuristic(state)

        snode.update(best)
        return snode, best

    def heuristic(self, state: "up.engines.State"):
        current_time = 0
        if isinstance(state, up.engines.CombinationState):
            current_time = state.current_time
        if _uses_tprpg_family(self.heuristic_name):
            score, _ = _tprpg_heuristic_value(
                self.split_mdp,
                state,
                current_time,
                self.temporal_heuristic_depth,
                self.temporal_heuristic_strategy,
                cached_table=self._root_baseline_cache,
                leaf_heuristic_name=self.heuristic_name,
            )
            return score
        h = up.engines.heuristics.TRPG(self.split_mdp, state, current_time)
        return h.get_heuristic()

    def selection(self, snode: "up.engines.Snode"):
        """
        Traverse the tree until reaching a leaf node.
        """
        if len(snode.possible_actions) == 0 or snode.state.current_time > self.mdp.deadline():
            # Stop when there are no possible actions to take so the plan remains consistent
            return -100

        if snode.depth > self.search_depth:
            return self.heuristic(snode.state)

        explore_constant = self.exploration_constant

        # Choose a consistent action
        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        anode = snode.children[action]
        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                reward += self.mdp.discount_factor * self.selection(snodes[next_state])

            else: # leaf
                next_snode, _ = self.create_Snode(next_state, snode.depth + 1, anode)
                reward += self.mdp.discount_factor * self.heuristic(next_state)
                anode.add_child(next_snode)

        snode.update(reward)
        anode.update(reward)

        return reward

    def selection_max(self, snode: "up.engines.Snode"):
        """
        Traverse the tree until reaching a leaf node.
        Selection with max logic -
        average between states and maximum between possible actions
        """
        if len(snode.possible_actions) == 0 or snode.state.current_time > self.mdp.deadline():
            # Stop when there are no possible actions to take so the plan remains consistent
            return -100

        if snode.depth > self.search_depth:
            # Stop if the search depth is reached
            return self.heuristic(snode.state)
        explore_constant = self.exploration_constant

        # Choose a consistent action
        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        anode = snode.children[action]
        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                reward += self.mdp.discount_factor * self.selection_max(snodes[next_state])

            else: # leaf
                next_snode, snode_reward = self.create_Snode_max(next_state, snode.depth + 1, anode)
                reward += snode_reward
                anode.add_child(next_snode)

        anode.update(reward)
        max_v = snode.max_update()

        return max_v

    def simulate(self, state, depth):
        """ Simulate until a terminal state """
        cumulative_reward = 0.0
        terminal = False
        deadline = self.mdp.deadline()
        while not terminal and depth < self.search_depth and len(self.mdp.legal_actions(state)) > 0:
            # Choose an action to execute
            action = self.default_policy(state)

            # Execute the action
            (terminal, next_state, reward) = self.mdp.step(state, action)

            # Discount the reward
            cumulative_reward += pow(self.mdp.discount_factor, depth) * reward
            depth += 1

            state = next_state

        return cumulative_reward


class C_MCTS(Base_MCTS):
    """
    TP MCTS solver implementation.
    Contains STNs in each node
    """
    def __init__(self, mdp, root_node: "up.engines.C_SNode", root_state: "up.engines.state.State", search_depth: int,
                 exploration_constant: float, stn: "up.plans.stn.STNPlan", selection_type, k: int,
                 previous_chosen_action_node: "up.plans.stn.STNPlanNode" = None,
                 heuristic_name: str = 'trpg', temporal_heuristic_depth: int = 25,
                 temporal_heuristic_strategy: str = "baseline",
                 root_baseline_cache=None, value_mode: str = "tp_mcts",
                 uct_initial_k: int = 3):
        super().__init__(mdp, search_depth, exploration_constant, k)
        self._previous_chosen_action_node = previous_chosen_action_node
        self.heuristic_name = heuristic_name
        self.temporal_heuristic_depth = temporal_heuristic_depth
        self.temporal_heuristic_strategy = temporal_heuristic_strategy
        self._root_baseline_cache = root_baseline_cache
        self.value_mode = value_mode
        validate_ptrpg_guided_rollout_config(value_mode, temporal_heuristic_strategy)
        validate_fixed_tail_ptrpg_rollout_config(value_mode, temporal_heuristic_strategy)
        self._uct_initial_k = max(1, int(uct_initial_k))
        self._uct_filter_mode = {
            'avg_topk': 'topk',
            'avg_pw': 'pw',
        }.get(selection_type)
        self._ptrpg_rollout_config = None
        self._ptrpg_rollout_debug_done = False
        if _uses_ptrpg_guided_rollout_value_mode(value_mode):
            from unified_planning.engines.solvers.ptrpg_guided_rollout import (
                rollout_config_from_args,
                resolve_rollout_policy,
            )

            self._ptrpg_rollout_config = rollout_config_from_args(mdp)
            try:
                self._ptrpg_rollout_config.policy_strategy = resolve_rollout_policy(
                    temporal_heuristic_strategy
                )
            except ValueError:
                pass

        self._fixed_tail_config = None
        self._fixed_tail_ctx = None
        self._fixed_tail_expectimax = None
        self._fixed_tail_debug_count = 0
        self._fixed_tail_root_debug_done = False
        if _uses_fixed_tail_ptrpg_rollout_value_mode(value_mode):
            from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
                build_fixed_tail_search_context,
                fixed_tail_config_from_args,
            )
            from unified_planning.engines.solvers.fixed_tail_expectimax import (
                FixedTailExpectimaxEvaluator,
                FixedTailExpectimaxGuards,
                uses_expectimax_prefix,
            )

            self._fixed_tail_config = fixed_tail_config_from_args()
            self._fixed_tail_config.tail_strategy = temporal_heuristic_strategy
            self._fixed_tail_ctx = build_fixed_tail_search_context(
                mdp, root_state, stn, self._fixed_tail_config
            )
            if uses_expectimax_prefix(self._fixed_tail_config):
                self._fixed_tail_expectimax = FixedTailExpectimaxEvaluator(
                    mdp=mdp,
                    ctx=self._fixed_tail_ctx,
                    strategy=temporal_heuristic_strategy,
                    guards=FixedTailExpectimaxGuards(
                        max_nodes=self._fixed_tail_config.max_expectimax_nodes,
                        max_depth=self._fixed_tail_config.max_expectimax_depth,
                        max_time_sec=self._fixed_tail_config.max_expectimax_time_sec,
                    ),
                )

        self._next_option_a_node_id = 1
        create_snode = self.create_Snode_max if selection_type == 'max' else (self.create_Snode_root_interval if selection_type == 'rootInterval' else self.create_Snode)
        snode, _ = create_snode(root_state, 0, stn,
                                previous_chosen_action_node=previous_chosen_action_node)
        self.set_root_node(root_node if root_node is not None else snode)
        self._stn = stn

    def _assign_option_a_node_id(self, snode, parent_snode=None):
        if not hasattr(self, "_next_option_a_node_id"):
            self._next_option_a_node_id = 1
        snode._option_a_node_id = self._next_option_a_node_id
        snode._option_a_parent_id = (
            None if parent_snode is None else getattr(parent_snode, "_option_a_node_id", None)
        )
        self._next_option_a_node_id += 1

    @property
    def previous_chosen_action_node(self):
        return self._previous_chosen_action_node

    @property
    def stn(self):
        return self._stn

    def _terminal_backup_reward(self, terminal: bool, next_state: "up.engines.State", sampled_reward: float) -> float:
        if self.value_mode == "greedy_matched" and terminal:
            return self.mdp.terminal_reward(True, next_state)
        return sampled_reward

    def _greedy_matched_action_target(self, snode: "up.engines.C_Snode", action: "up.engines.Action") -> float:
        from unified_planning.engines.solvers.greedy_parallel import greedy_matched_value_target

        current_time = snode.children[action].stn.get_current_end_time()
        return greedy_matched_value_target(
            mdp=self.mdp,
            state=snode.state,
            action=action,
            current_time=current_time,
            heuristic_name=self.heuristic_name,
            temporal_heuristic_depth=self.temporal_heuristic_depth,
            temporal_heuristic_strategy=self.temporal_heuristic_strategy,
        )

    def _uses_ptrpg_guided_rollout(self) -> bool:
        return _uses_ptrpg_guided_rollout_value_mode(self.value_mode)

    def _uses_fixed_tail_ptrpg_rollout(self) -> bool:
        return _uses_fixed_tail_ptrpg_rollout_value_mode(self.value_mode)

    def _uses_fixed_tail_expectimax(self) -> bool:
        from unified_planning.engines.solvers.fixed_tail_expectimax import (
            uses_expectimax_prefix,
        )

        return (
            self._uses_fixed_tail_ptrpg_rollout()
            and self._fixed_tail_config is not None
            and uses_expectimax_prefix(self._fixed_tail_config)
            and self._fixed_tail_expectimax is not None
        )

    def _fixed_tail_previous_stn_node(
        self, snode: "up.engines.C_SNode"
    ) -> "up.plans.stn.STNPlanNode":
        if snode.parent is not None:
            return snode.parent.STNNode
        return self._previous_chosen_action_node

    def _fixed_tail_expectimax_v_at_snode(self, snode: "up.engines.C_SNode") -> float:
        stn = self._fixed_tail_snode_stn(snode)
        prev = self._fixed_tail_previous_stn_node(snode)
        actions = list(snode.children.keys())
        return self._fixed_tail_expectimax.value_for_feasible_actions(
            snode.state,
            stn,
            prev,
            actions,
        )

    def _fixed_tail_seed_expectimax_q(self, snode: "up.engines.C_SNode") -> None:
        if getattr(snode, "_expectimax_seeded", False):
            return
        stn = self._fixed_tail_snode_stn(snode)
        prev = self._fixed_tail_previous_stn_node(snode)
        evaluator = self._fixed_tail_expectimax
        q_by_action = {}
        for action, anode in snode.children.items():
            q_val, outcomes = evaluator.q_value_with_outcomes(
                snode.state, stn, prev, action
            )
            anode._expectimax_q = q_val
            anode._expectimax_outcomes = outcomes
            q_by_action[action] = q_val
            if anode.count == 0:
                anode.update(q_val)
        snode._expectimax_seeded = True
        if snode.depth == 0:
            self._fixed_tail_maybe_debug_root(snode, q_by_action)

    def _fixed_tail_maybe_debug_root(
        self,
        snode: "up.engines.C_SNode",
        q_by_action: dict,
    ) -> None:
        cli = getattr(up, "args", None)
        if cli is None or not getattr(cli, "fixed_tail_debug", False):
            return
        if getattr(self, "_fixed_tail_root_debug_done", False):
            return
        self._fixed_tail_root_debug_done = True

        from unified_planning.engines.solvers.greedy_parallel import pick_best_action

        stn = self._fixed_tail_snode_stn(snode)
        prev = self._fixed_tail_previous_stn_node(snode)
        legal = list(snode.children.keys())
        greedy_action = pick_best_action(
            mdp=self.mdp,
            state=snode.state,
            stn=stn,
            previous_action_node=prev,
            legal_actions=legal,
            heuristic_name=self.heuristic_name,
            temporal_heuristic_depth=self.temporal_heuristic_depth,
            temporal_heuristic_strategy=self.temporal_heuristic_strategy,
        )
        best_q = -float("inf")
        best_actions = []
        for action, q_val in q_by_action.items():
            if q_val > best_q:
                best_q = q_val
                best_actions = [action]
            elif q_val == best_q:
                best_actions.append(action)
        expectimax_action = best_actions[0] if best_actions else None

        ctx = self._fixed_tail_ctx
        print(
            f"[fixed_tail expectimax root] prefix_frac={ctx.prefix_frac} "
            f"prefix_budget={ctx.prefix_budget} root_remaining={ctx.root_remaining}",
            flush=True,
        )
        for action in legal:
            anode = snode.children[action]
            q_val = getattr(anode, "_expectimax_q", q_by_action.get(action, 0.0))
            outcomes = getattr(anode, "_expectimax_outcomes", [])
            oc_str = ",".join(f"p={p:.3f}:v={v:.4f}" for p, v in outcomes)
            selected = "yes" if action in best_actions else "no"
            name = getattr(action, "name", action)
            print(
                f"  action={name} Q_expectimax={q_val:.6f} outcomes=[{oc_str}] "
                f"selected={selected}",
                flush=True,
            )
        gname = getattr(greedy_action, "name", greedy_action) if greedy_action else "none"
        ename = getattr(expectimax_action, "name", expectimax_action) if expectimax_action else "none"
        match = greedy_action == expectimax_action
        print(
            f"  greedy_first={gname} expectimax_first={ename} match={'yes' if match else 'no'}",
            flush=True,
        )

    def _fixed_tail_snode_stn(self, snode: "up.engines.C_SNode") -> "up.plans.stn.STNPlan":
        if snode.parent is not None:
            return snode.parent.stn
        return self._stn

    def _fixed_tail_should_debug(self) -> bool:
        cli = getattr(up, "args", None)
        if cli is None or not getattr(cli, "fixed_tail_debug", False):
            return False
        if self._fixed_tail_debug_count >= 5:
            return False
        self._fixed_tail_debug_count += 1
        return True

    def _fixed_tail_bootstrap_at_snode(
        self,
        snode: "up.engines.C_SNode",
        *,
        label: str = "bootstrap",
        action_duration: float | None = None,
        child_remaining: int | None = None,
        child_elapsed: int | None = None,
        crossed: bool | None = None,
        cache: bool = True,
    ) -> float:
        from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
            fixed_tail_bootstrap_value,
        )

        if cache and getattr(snode, "_fixed_tail_bootstrap", False):
            return float(getattr(snode, "_fixed_tail_value", 0.0))

        stn = self._fixed_tail_snode_stn(snode)
        debug = self._fixed_tail_should_debug()
        value = fixed_tail_bootstrap_value(
            self.mdp,
            snode.state,
            stn,
            self.temporal_heuristic_strategy,
            ctx=self._fixed_tail_ctx,
            debug_emit=debug,
            debug_label=label,
            action_duration=action_duration,
            child_remaining=child_remaining,
            child_elapsed=child_elapsed,
            crossed=crossed,
        )
        if cache:
            snode._fixed_tail_bootstrap = True
            snode._fixed_tail_value = value
        return value

    def _fixed_tail_elapsed_at_snode(self, snode: "up.engines.C_SNode") -> int:
        from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
            elapsed_from_root,
            node_remaining,
        )

        rem = node_remaining(self.mdp, snode.state, self._fixed_tail_snode_stn(snode))
        return elapsed_from_root(self._fixed_tail_ctx, rem)

    def _fixed_tail_at_cutoff(self, snode: "up.engines.C_SNode") -> bool:
        from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import crossed_cutoff

        return crossed_cutoff(self._fixed_tail_ctx, self._fixed_tail_elapsed_at_snode(snode))

    def _leaf_fixed_tail_value(self, snode: "up.engines.C_SNode") -> float:
        if self._uses_fixed_tail_expectimax() and not self._fixed_tail_at_cutoff(snode):
            self._fixed_tail_seed_expectimax_q(snode)
            return self._fixed_tail_expectimax_v_at_snode(snode)
        return self._fixed_tail_bootstrap_at_snode(snode, label="depth_or_leaf")

    def _leaf_rollout_value_from_anode(
        self,
        state: "up.engines.State",
        anode: "up.engines.C_ANode",
    ) -> float:
        from unified_planning.engines.solvers.ptrpg_guided_rollout import (
            ptrpg_guided_terminal_rollout,
        )

        debug_emit = False
        if (
            self._ptrpg_rollout_config is not None
            and self._ptrpg_rollout_config.debug_first_rollout
            and not self._ptrpg_rollout_debug_done
        ):
            debug_emit = True
            self._ptrpg_rollout_debug_done = True
        return ptrpg_guided_terminal_rollout(
            mdp=self.mdp,
            state=state,
            stn=anode.stn,
            previous_action_node=anode.STNNode,
            config=self._ptrpg_rollout_config,
            heuristic_name=self.heuristic_name,
            temporal_heuristic_depth=self.temporal_heuristic_depth,
            debug_emit=debug_emit,
        )

    def _leaf_rollout_value(self, snode: "up.engines.C_SNode") -> float:
        if snode.parent is None:
            return 0.0
        return self._leaf_rollout_value_from_anode(snode.state, snode.parent)

    def _depth_cutoff_value(self, snode: "up.engines.C_Snode"):
        if self._uses_ptrpg_guided_rollout():
            return self._leaf_rollout_value(snode)
        if self._uses_fixed_tail_ptrpg_rollout():
            return self._leaf_fixed_tail_value(snode)
        if self.value_mode != "greedy_matched":
            return self.heuristic(snode)

        best = -math.inf
        for action in snode.possible_actions:
            target = self._greedy_matched_action_target(snode, action)
            if target > best:
                best = target
        return self.heuristic(snode) if best == -math.inf else best

    def _action_rank_key(self, action):
        action_name = getattr(action, "name", None)
        return action_name if action_name is not None else str(action)

    def _rank_actions_by_expected_target(self, snode: "up.engines.C_Snode"):
        ranked = []
        for action in snode.possible_actions:
            ranked.append((self._greedy_matched_action_target(snode, action), action))
        ranked.sort(key=lambda item: (-item[0], self._action_rank_key(item[1])))
        return [action for _, action in ranked]

    def _allowed_actions_for_uct(self, snode: "up.engines.C_Snode"):
        ranked_actions = self._rank_actions_by_expected_target(snode)
        if not ranked_actions:
            return ranked_actions

        if self._uct_filter_mode == 'topk':
            allowed = min(len(ranked_actions), self._uct_initial_k)
        else:
            allowed = min(len(ranked_actions), self._uct_initial_k + int(math.sqrt(snode.count)))

        return ranked_actions[:allowed]

    def _frontier_lambda_align(self) -> float:
        a = getattr(up, "args", None)
        lam = getattr(a, "rollout_aligned_lambda_align", None) if a is not None else None
        return 1.0 if lam is None else max(0.0, min(1.0, float(lam)))

    def _frontier_debug_enabled(self) -> bool:
        import os
        if os.environ.get("FRONTIER_ALIGNED_DEBUG"):
            return True
        a = getattr(up, "args", None)
        return bool(getattr(a, "frontier_aligned_debug", False)) if a is not None else False

    def _frontier_aligned_value(self, snode: "up.engines.C_SNode") -> float:
        """Frontier-aligned value of a node, used ONLY for selection — never
        backpropagated. Routes through the alignment evaluator (prefix rollout to
        the comparison horizon + raw PTRPG suffix; goal->1, dead-end->0)."""
        if not _uses_tprpg_family(self.heuristic_name):
            return self.heuristic(snode)
        current_time = 0
        if snode.parent:
            current_time = snode.parent.stn.get_current_end_time()
        aligned_override = None
        if snode.parent is not None:
            aligned_override = _dynamic_aligned_horizon(self.mdp, snode.parent.parent)
        score, _ = _tprpg_heuristic_value(
            self.mdp,
            snode.state,
            current_time,
            self.temporal_heuristic_depth,
            self.temporal_heuristic_strategy,
            cached_table=self._root_baseline_cache,
            leaf_heuristic_name=self.heuristic_name,
            aligned_h_override=aligned_override,
        )
        return score

    # ---- Option A: global open-leaf frontier driver -----------------------

    def _node_elapsed(self, snode):
        if snode.parent is not None:
            try:
                return float(snode.parent.stn.get_current_end_time())
            except Exception:
                pass
        try:
            return float(self._stn.get_current_end_time())
        except Exception:
            return float(getattr(snode.state, "current_time", 0.0))

    def _collect_frontier(self):
        """Global open/expandable leaf nodes across the tree (spanning depths)."""
        frontier = []
        stack = [self.root_node]
        seen = set()
        while stack:
            snode = stack.pop()
            if id(snode) in seen:
                continue
            seen.add(id(snode))
            if not snode.possible_actions:
                continue  # terminal / dead-end -> not expandable
            if snode.depth > self.search_depth:
                continue
            expandable = any(
                (anode.count == 0 or not anode.children)
                for anode in snode.children.values()
            )
            if expandable:
                frontier.append(snode)
            for anode in snode.children.values():
                for child in anode.children.values():
                    stack.append(child)
        return frontier

    def _frontier_aligned_value_global(self, snode, elapsed, H_frontier):
        """aligned_value(n): prefix-roll delta then PTRPG at the GLOBAL H_frontier
        (goal->1, dead-end->0), via the evaluator. Cached per node until
        H_frontier changes (lazy). Never parent-local; ROLLOUT_ALIGNED_H inert."""
        key = H_frontier if H_frontier is not None else -1.0
        cache = getattr(snode, "_aligned_cache", None)
        if cache is not None and abs(cache[0] - key) < 1e-9:
            return cache[1]
        if not _uses_tprpg_family(self.heuristic_name):
            value = snode.value
        else:
            score, _ = _tprpg_heuristic_value(
                self.mdp,
                snode.state,
                elapsed,
                self.temporal_heuristic_depth,
                self.temporal_heuristic_strategy,
                cached_table=self._root_baseline_cache,
                leaf_heuristic_name=self.heuristic_name,
                aligned_h_override=H_frontier,
            )
            value = score
        snode._aligned_cache = (key, value)
        return value

    def _pick_expand_action(self, snode):
        for action, anode in snode.children.items():
            if anode.count == 0:
                return action
        best, best_count = None, float("inf")
        for action, anode in snode.children.items():
            if anode.count < best_count:
                best_count, best = anode.count, action
        return best

    def _backprop_to_root(self, snode, reward):
        """Standard backprop of a freshly expanded value up to the root."""
        node = snode
        r = reward
        while node.parent is not None:  # node.parent is a C_ANode
            anode = node.parent
            parent_snode = anode.parent
            r = self.mdp.discount_factor * r  # edge reward ~0 in terminal mode
            anode.update(r)
            if parent_snode is None:
                break
            parent_snode.update(r)
            node = parent_snode

    def _expand_and_backprop(self, snode):
        """Expand ONE child of the selected ORIGINAL node, standard backprop.
        The aligned score is selection-only; the child is scored by the normal
        heuristic and rollout endpoints are never inserted."""
        action = self._pick_expand_action(snode)
        if action is None:
            snode.update(snode.value)
            return
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        reward = self._terminal_backup_reward(terminal, next_state, reward)
        anode = snode.children[action]
        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                reward += self.mdp.discount_factor * snodes[next_state].value
            else:
                next_snode, _ = self.create_Snode(next_state, snode.depth + 1, anode.stn, anode)
                h_val = self.heuristic(next_snode)  # standard PTRPG (raw for frontier)
                reward += self.mdp.discount_factor * h_val
                anode.add_child(next_snode)
                next_snode.update(reward)
        snode.update(reward)
        anode.update(reward)
        self._backprop_to_root(snode, reward)

    def _frontier_iteration(self):
        """One Option A iteration: pick the globally best open-leaf node by
        frontier-aligned score, then expand THAT original node."""
        from comdp_plus_no_deadline.engines.rollout_aligned import frontier_score

        frontier = self._collect_frontier()
        if not frontier:
            return
        deadline = self.mdp.deadline()
        elapsed_map = {id(n): self._node_elapsed(n) for n in frontier}
        deepest_elapsed = max(elapsed_map.values())
        H_frontier = (deadline - deepest_elapsed) if deadline is not None else None
        if deadline is not None:
            assert H_frontier == deadline - deepest_elapsed, "H_frontier mismatch"
        lam = self._frontier_lambda_align()
        same_elapsed = all(abs(e - deepest_elapsed) < 1e-9 for e in elapsed_map.values())

        best, best_score, rows = None, -float("inf"), []
        for n in frontier:
            elapsed = elapsed_map[id(n)]
            remaining = (deadline - elapsed) if deadline is not None else self.temporal_heuristic_depth
            delta = deepest_elapsed - elapsed
            assert abs(delta - (deepest_elapsed - elapsed)) < 1e-9, "delta mismatch"
            aligned = self._frontier_aligned_value_global(n, elapsed, H_frontier)
            existing = n.value
            priority = 1e-6 / (1.0 + n.count)  # tie-break / progress; spec score is (1-l)Q + l*aligned
            score = frontier_score(existing, aligned, lam) + priority
            if score > best_score:
                best_score, best = score, n
            rows.append((n, elapsed, remaining, delta, aligned, existing, score))

        if self._frontier_debug_enabled():
            # Spend the trace budget on NON-degenerate frontiers (delta>0 on some
            # node) so we actually observe alignment engaging; degenerate frontiers
            # (all elapsed equal -> no alignment) only get a single compact note so
            # the early all-elapsed-0 phase does not exhaust the budget. Override
            # the old "first N iterations" behavior with FRONTIER_ALIGNED_DEBUG_ALL=1.
            import os
            debug_all = bool(os.environ.get("FRONTIER_ALIGNED_DEBUG_ALL"))
            budget = getattr(self.mdp, "_frontier_debug_remaining", 5)
            should_print = budget > 0 and (debug_all or not same_elapsed)
            if same_elapsed and not debug_all:
                if not getattr(self.mdp, "_frontier_debug_degen_noted", False):
                    setattr(self.mdp, "_frontier_debug_degen_noted", True)
                    print(f"\n[frontier_aligned/global] DEGENERATE frontier "
                          f"(all elapsed={deepest_elapsed}, size={len(frontier)}); "
                          f"suppressing further degenerate traces, waiting for delta>0.",
                          flush=True)
                should_print = False
            if should_print:
                setattr(self.mdp, "_frontier_debug_remaining", budget - 1)
                note = ("  (DEGENERATE: all frontier nodes share elapsed -> raw PTRPG, no alignment needed)"
                        if same_elapsed else "")
                print(f"\n[frontier_aligned/global] frontier_size={len(frontier)} "
                      f"deadline={deadline} deepest_elapsed={deepest_elapsed} "
                      f"H_frontier={H_frontier} lambda={lam}{note}", flush=True)
                for n, elapsed, remaining, delta, aligned, existing, score in rows:
                    print(f"  depth={n.depth:<2} elapsed={elapsed:<6} remaining={remaining:<6} "
                          f"delta={delta:<6} aligned={aligned:.4f} existing={existing:.4f} "
                          f"frontier_score={score:.4f} selected={'YES' if n is best else 'no'}",
                          flush=True)
                # Surface evaluator diagnostics so we can tell WHY delta>0 inflates:
                # goal-reaching during the prefix (-> 1.0) vs a higher suffix PTRPG.
                evals = getattr(self.mdp, "_rollout_aligned_evaluators", None) or {}
                for ekey, ev in evals.items():
                    d = ev.diagnostics.as_dict()
                    print(f"  [evaluator {ekey}] prefix_rollouts={d['prefix_rollouts']} "
                          f"reached_goal={d['prefix_rollouts_reached_goal']} "
                          f"dead_end={d['prefix_rollouts_dead_end']} "
                          f"goal_rate={d['prefix_goal_rate']:.3f} "
                          f"avg_suffix_ptrpg={d['avg_suffix_ptrpg_value']:.4f} "
                          f"avg_prefix_len={d['avg_prefix_rollout_length']:.2f}",
                          flush=True)

        if best is not None:
            self._expand_and_backprop(best)

    # ---- frontier_aligned_option_a: clean global Option A driver ------------

    def _option_a_debug_enabled(self) -> bool:
        import os
        if os.environ.get("FRONTIER_OPTION_A_DEBUG"):
            return True
        a = getattr(up, "args", None)
        return bool(getattr(a, "frontier_option_a_debug", False)) if a is not None else False

    def _get_option_a_evaluator(self):
        from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
            TemporalProbabilisticRPGHeuristic,
        )

        heuristic = getattr(self.mdp, "_temporal_probabilistic_rpg_heuristic", None)
        if heuristic is None:
            heuristic = TemporalProbabilisticRPGHeuristic.from_problem(self.mdp.problem)
            setattr(self.mdp, "_temporal_probabilistic_rpg_heuristic", heuristic)
        suffix = _option_a_ptrpg_suffix(self.temporal_heuristic_strategy)
        return _build_option_a_evaluator(
            self.mdp,
            heuristic,
            suffix,
            aggregation=_aggregation_for_strategy(suffix),
            resolution_kwargs=_resolution_heuristic_kwargs_from_cli(),
        )

    def _option_a_raw_ptrpg(self, snode, elapsed: float, remaining: int) -> float:
        suffix = _option_a_ptrpg_suffix(self.temporal_heuristic_strategy)
        score, _ = _tprpg_heuristic_value(
            self.mdp,
            snode.state,
            elapsed,
            self.temporal_heuristic_depth,
            suffix,
            cached_table=self._root_baseline_cache,
            leaf_heuristic_name=self.heuristic_name,
        )
        return score

    def _option_a_frontier_iteration(self):
        """Global frontier Option A: argmax aligned_value, expand selected node only."""
        deadline = self.mdp.deadline()
        frontier = _collect_global_frontier(
            self.root_node,
            search_depth=self.search_depth,
            deadline=deadline,
            root_stn=self._stn,
        )
        if not frontier:
            return
        deepest_elapsed, H_frontier, elapsed_map = _compute_H_frontier(
            frontier, deadline, root_stn=self._stn
        )
        evaluator = self._get_option_a_evaluator()
        scores: dict = {}
        rows = []
        for n in frontier:
            elapsed = elapsed_map[id(n)]
            remaining = _option_a_remaining_horizon(
                deadline, elapsed, self.temporal_heuristic_depth
            )
            delta = deepest_elapsed - elapsed
            raw = self._option_a_raw_ptrpg(n, elapsed, remaining)
            aligned = _option_a_aligned_value(
                evaluator,
                n.state,
                remaining=remaining,
                H_frontier=H_frontier,
                deepest_elapsed=deepest_elapsed,
                elapsed=elapsed,
            )
            scores[id(n)] = aligned
            rows.append((n, elapsed, remaining, delta, raw, aligned))

        best = _select_frontier_node(frontier, scores)
        same_elapsed = all(abs(e - deepest_elapsed) < 1e-9 for e in elapsed_map.values())

        if self._option_a_debug_enabled():
            budget = getattr(self.mdp, "_option_a_debug_remaining", 3)
            if budget > 0:
                setattr(self.mdp, "_option_a_debug_remaining", budget - 1)
                note = (
                    "  (DEGENERATE: all frontier nodes share elapsed)"
                    if same_elapsed
                    else ""
                )
                print(
                    f"\n[frontier_aligned_option_a] frontier_size={len(frontier)} "
                    f"deadline={deadline} deepest_elapsed={deepest_elapsed} "
                    f"H_frontier={H_frontier}{note}",
                    flush=True,
                )
                for n, elapsed, remaining, delta, raw, aligned in rows:
                    print(
                        "  "
                        + _format_option_a_debug_row(
                            node_id=getattr(n, "_option_a_node_id", id(n)),
                            parent_id=getattr(n, "_option_a_parent_id", None),
                            depth=n.depth,
                            elapsed=elapsed,
                            remaining=float(remaining),
                            deepest_elapsed=deepest_elapsed,
                            H_frontier=H_frontier,
                            delta=delta,
                            raw_ptrpg=raw,
                            aligned_value=aligned,
                            selected=(n is best),
                        ),
                        flush=True,
                    )

        if best is not None:
            self._expand_and_backprop(best)

    def _uct_frontier_aligned(self, snode: "up.engines.Snode", explore_constant: float):
        """Option A: pick the child to descend/expand by a frontier-aligned score.

        Over the candidate child set (the frontier F), every child's aligned value
        was computed against H_frontier = deadline - max(child elapsed) and stashed
        on its action node. The selection score blends that aligned value with the
        backed-up action value Q via lambda_align, plus a UCT exploration term.
        Unvisited children are explored first. The ORIGINAL chosen child is then
        expanded by the normal selection flow (rollout endpoints are never
        inserted)."""
        from comdp_plus_no_deadline.engines.rollout_aligned import frontier_score

        anodes = snode.children
        if not anodes:
            return super().uct(snode, explore_constant)

        # Explore unvisited children first (standard UCT behavior).
        for action, anode in anodes.items():
            if anode.count == 0:
                return action

        # Frontier diagnostics (deepest elapsed / H_frontier).
        diag = getattr(self.mdp, "_frontier_aligned_diag", None)
        if diag is None:
            diag = {"selections": 0, "h_frontier_sum": 0.0, "frontier_size_sum": 0}
            setattr(self.mdp, "_frontier_aligned_diag", diag)
        deadline = self.mdp.deadline()
        deepest_elapsed = None
        for anode in anodes.values():
            stn = getattr(anode, "stn", None)
            if stn is None:
                continue
            try:
                end = stn.get_current_end_time()
            except Exception:
                continue
            if deepest_elapsed is None or end > deepest_elapsed:
                deepest_elapsed = end
        if deadline is not None and deepest_elapsed is not None:
            diag["h_frontier_sum"] += max(0.0, float(deadline) - float(deepest_elapsed))
        diag["frontier_size_sum"] += len(anodes)
        diag["selections"] += 1

        lam = self._frontier_lambda_align()
        log_n = math.log(snode.count) if snode.count > 0 else 0.0
        best_score = -float("inf")
        best_action = None
        rows = []
        for action, anode in anodes.items():
            q = anode.value
            aligned = getattr(anode, "_aligned_seed", q)
            exploration = explore_constant * math.sqrt(log_n / anode.count)
            score = frontier_score(q, aligned, lam, exploration)
            if score > best_score:
                best_score = score
                best_action = action
            rows.append((action, anode, q, aligned, score))

        # Debug trace for the first few decisions (FRONTIER_ALIGNED_DEBUG=1).
        if self._frontier_debug_enabled():
            budget = getattr(self.mdp, "_frontier_debug_remaining", None)
            if budget is None:
                budget = 3
            if budget > 0:
                setattr(self.mdp, "_frontier_debug_remaining", budget - 1)
                deadline_v = deadline if deadline is not None else float("nan")
                print(f"\n[frontier_aligned debug] node=snode(depth={snode.depth}) "
                      f"deadline={deadline_v} deepest_elapsed={deepest_elapsed} "
                      f"H_frontier={(deadline_v - deepest_elapsed) if deepest_elapsed is not None else None} "
                      f"lambda={lam} frontier_size={len(anodes)}", flush=True)
                # Assertions from the spec.
                if deadline is not None and deepest_elapsed is not None:
                    assert abs((deadline - deepest_elapsed) - (deadline - deepest_elapsed)) < 1e-9
                for action, anode, q, aligned, score in rows:
                    try:
                        elapsed = anode.stn.get_current_end_time()
                    except Exception:
                        elapsed = None
                    remaining = (deadline - elapsed) if (deadline is not None and elapsed is not None) else None
                    delta = (deepest_elapsed - elapsed) if (deepest_elapsed is not None and elapsed is not None) else None
                    if delta is not None:
                        assert delta >= -1e-9, f"delta<0: {delta}"
                    print(f"  action={getattr(action, 'name', action)!s:35.35} "
                          f"elapsed={elapsed} remaining={remaining} delta={delta} "
                          f"aligned={aligned:.4f} Q={q:.4f} frontier_score={score:.4f} "
                          f"selected={'YES' if action is best_action else 'no'}", flush=True)

        return best_action if best_action is not None else super().uct(snode, explore_constant)

    def uct(self, snode: "up.engines.Snode", explore_constant: float):
        if _is_option_a_strategy(self.temporal_heuristic_strategy):
            return super().uct(snode, explore_constant)
        if self.temporal_heuristic_strategy in _FRONTIER_ALIGNED_SUFFIX:
            return self._uct_frontier_aligned(snode, explore_constant)
        if self._uct_filter_mode is None:
            return super().uct(snode, explore_constant)

        anodes = snode.children
        candidate_actions = self._allowed_actions_for_uct(snode)
        if not candidate_actions:
            return super().uct(snode, explore_constant)

        best_ub = -float('inf')
        best_action = None
        for action in candidate_actions:
            if anodes[action].count == 0:
                return action

            ub = anodes[action].value + (
                    explore_constant * math.sqrt(math.log(snode.count) / anodes[action].count))
            if ub > best_ub:
                best_ub = ub
                best_action = action

        if best_action is not None:
            return best_action
        return super().uct(snode, explore_constant)

    def create_Snode(self, state: "up.engines.State", depth: int, stn: "up.plans.stn.STNPlan",
                     parent: "up.engines.C_ANode" = None,
                     previous_chosen_action_node: "up.plans.stn.STNPlanNode" = None, isInterval=False):
        """ Create a new Snode for the state `state` with parent `parent`"""
        snode = up.engines.C_SNode(
            state, depth, self.mdp.legal_actions(state), stn, parent,
            previous_chosen_action_node, isInterval,
        )
        if _is_option_a_strategy(self.temporal_heuristic_strategy):
            parent_snode = parent.parent if parent is not None else None
            self._assign_option_a_node_id(snode, parent_snode=parent_snode)
        return snode, None

    def create_Snode_root_interval(self, state: "up.engines.State", depth: int, stn: "up.plans.stn.STNPlan",
                     parent: "up.engines.C_ANode" = None,
                     previous_chosen_action_node: "up.plans.stn.STNPlanNode" = None, isInterval=True):
        """ Create a new Snode for the state `state` with parent `parent`
        RootInterval approach """
        return up.engines.C_SNode(state, depth, self.mdp.legal_actions(state), stn, parent,
                                  previous_chosen_action_node, isInterval), None

    def create_Snode_max(self, state: "up.engines.State", depth: int, stn: "up.plans.stn.STNPlan",
                         parent: "up.engines.C_ANode" = None,
                         previous_chosen_action_node: "up.plans.stn.STNPlanNode" = None):
        """ Create a new Snode for the state `state` with parent `parent`
         In this approach k children of snode are evaluated and the initiate value of snode is set to maximum value."""
        snode = up.engines.C_SNode(state, depth, self.mdp.legal_actions(state), stn, parent,
                                   previous_chosen_action_node)
        best = -math.inf

        actions_idx = list(range(len(snode.children)))
        if self.k < len(snode.children):
            actions_idx = random.sample(range(0, len(snode.children)), self.k)

        for action_idx in actions_idx:
            action = list(snode.children.keys())[action_idx]
            terminal, next_state, reward = self.mdp.step(snode.state, action)
            reward = self._terminal_backup_reward(terminal, next_state, reward)
            if self.value_mode == "greedy_matched":
                if not terminal:
                    reward = self._greedy_matched_action_target(snode, action)
            elif self._uses_ptrpg_guided_rollout():
                if not terminal:
                    reward += self.mdp.discount_factor * self._leaf_rollout_value_from_anode(
                        next_state, snode.children[action]
                    )
            elif self._uses_fixed_tail_ptrpg_rollout():
                if not terminal:
                    anode_child = snode.children[action]
                    child_snode, _ = self.create_Snode(
                        next_state, snode.depth + 1, anode_child.stn, anode_child
                    )
                    from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
                        crossed_cutoff,
                        elapsed_from_root,
                        node_remaining,
                    )

                    child_rem = node_remaining(self.mdp, next_state, anode_child.stn)
                    child_elapsed = elapsed_from_root(self._fixed_tail_ctx, child_rem)
                    anode_child.add_child(child_snode)
                    if crossed_cutoff(self._fixed_tail_ctx, child_elapsed):
                        leaf_val = self._fixed_tail_post_step_child_value(
                            snode, child_snode, anode_child
                        )
                    elif self._uses_fixed_tail_expectimax():
                        leaf_val = self._fixed_tail_expectimax_v_at_snode(child_snode)
                    else:
                        leaf_val = self.selection_max(child_snode)
                    reward += self.mdp.discount_factor * leaf_val
            else:
                reward += self.mdp.discount_factor * self.heuristic_init(
                    next_state, snode.children[action].stn, parent_snode=snode
                )
            snode.children[action].update(reward)
            if reward > best:
                best = reward
        if best == -math.inf:
            best = self._depth_cutoff_value(snode)

        snode.update(best)
        return snode, best


    def _fixed_tail_post_step_child_value(
        self,
        parent_snode: "up.engines.C_SNode",
        child_snode: "up.engines.C_SNode",
        anode: "up.engines.C_ANode",
    ) -> float:
        from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
            crossed_cutoff,
            elapsed_from_root,
            node_remaining,
        )

        child_rem = node_remaining(self.mdp, child_snode.state, anode.stn)
        child_elapsed = elapsed_from_root(self._fixed_tail_ctx, child_rem)
        parent_rem = node_remaining(
            self.mdp, parent_snode.state, self._fixed_tail_snode_stn(parent_snode)
        )
        action_duration = float(max(0, parent_rem - child_rem))
        crossed = crossed_cutoff(self._fixed_tail_ctx, child_elapsed)
        label = "overshoot_child" if crossed else "expand_child"
        return self._fixed_tail_bootstrap_at_snode(
            child_snode,
            label=label,
            action_duration=action_duration,
            child_remaining=child_rem,
            child_elapsed=child_elapsed,
            crossed=crossed,
            cache=crossed,
        )

    def selection(self, snode: "up.engines.C_Snode"):
        """
                Traverse the tree until reaching a leaf node.
         """
        if self._uses_fixed_tail_ptrpg_rollout():
            if getattr(snode, "_fixed_tail_bootstrap", False):
                return float(getattr(snode, "_fixed_tail_value", 0.0))

        if len(snode.possible_actions) == 0:
            if self._uses_fixed_tail_ptrpg_rollout():
                return 0.0
            return -100

        if snode.depth > self.search_depth:
            return self._depth_cutoff_value(snode)

        if self._uses_fixed_tail_ptrpg_rollout() and self._fixed_tail_at_cutoff(snode):
            return self._fixed_tail_bootstrap_at_snode(snode, label="cutoff_node", crossed=True)

        if self._uses_fixed_tail_expectimax() and not self._fixed_tail_at_cutoff(snode):
            self._fixed_tail_seed_expectimax_q(snode)

        explore_constant = self.exploration_constant

        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        reward = self._terminal_backup_reward(terminal, next_state, reward)
        anode = snode.children[action]
        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                reward += self.mdp.discount_factor * self.selection(snodes[next_state])

            else: # leaf
                next_snode, _ = self.create_Snode(next_state, snode.depth + 1, anode.stn, anode)
                if self.value_mode == "greedy_matched":
                    reward = self._greedy_matched_action_target(snode, action)
                elif self._uses_ptrpg_guided_rollout():
                    reward += self.mdp.discount_factor * self._leaf_rollout_value(next_snode)
                elif self._uses_fixed_tail_ptrpg_rollout():
                    from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
                        crossed_cutoff,
                        elapsed_from_root,
                        node_remaining,
                    )

                    child_rem = node_remaining(self.mdp, next_state, anode.stn)
                    child_elapsed = elapsed_from_root(self._fixed_tail_ctx, child_rem)
                    if crossed_cutoff(self._fixed_tail_ctx, child_elapsed):
                        leaf_val = self._fixed_tail_post_step_child_value(
                            snode, next_snode, anode
                        )
                        reward += self.mdp.discount_factor * leaf_val
                        anode.add_child(next_snode)
                    elif self._uses_fixed_tail_expectimax():
                        anode.add_child(next_snode)
                        leaf_val = self._fixed_tail_expectimax_v_at_snode(next_snode)
                        reward += self.mdp.discount_factor * leaf_val
                    else:
                        anode.add_child(next_snode)
                        reward += self.mdp.discount_factor * self.selection(next_snode)
                        next_snode.update(reward)
                else:
                    h_val = self.heuristic(next_snode)
                    if self.temporal_heuristic_strategy in _FRONTIER_ALIGNED_SUFFIX:
                        anode._aligned_seed = self._frontier_aligned_value(next_snode)
                    reward += self.mdp.discount_factor * h_val
                    anode.add_child(next_snode)
                    next_snode.update(reward)

        snode.update(reward)
        anode.update(reward)

        return reward

    def selection_max(self, snode: "up.engines.C_Snode"):
        """
        Traverse the tree until reaching a leaf node.
        Selection with max logic -
        average between states and maximum between possible actions
        """
        if self._uses_fixed_tail_ptrpg_rollout():
            if getattr(snode, "_fixed_tail_bootstrap", False):
                return float(getattr(snode, "_fixed_tail_value", 0.0))

        if len(snode.possible_actions) == 0:
            if self._uses_fixed_tail_ptrpg_rollout():
                return 0.0
            return -100

        if snode.depth > self.search_depth:
            return self._depth_cutoff_value(snode)

        if self._uses_fixed_tail_ptrpg_rollout() and self._fixed_tail_at_cutoff(snode):
            return self._fixed_tail_bootstrap_at_snode(snode, label="cutoff_node", crossed=True)

        if self._uses_fixed_tail_expectimax() and not self._fixed_tail_at_cutoff(snode):
            self._fixed_tail_seed_expectimax_q(snode)

        explore_constant = self.exploration_constant

        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        reward = self._terminal_backup_reward(terminal, next_state, reward)
        anode = snode.children[action]
        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                reward += self.mdp.discount_factor * self.selection_max(snodes[next_state])

            else: #leaf
                if self.value_mode == "greedy_matched":
                    next_snode, _ = self.create_Snode(next_state, snode.depth + 1, anode.stn, anode)
                    reward = self._greedy_matched_action_target(snode, action)
                elif self._uses_ptrpg_guided_rollout():
                    next_snode, _ = self.create_Snode(next_state, snode.depth + 1, anode.stn, anode)
                    reward += self.mdp.discount_factor * self._leaf_rollout_value(next_snode)
                elif self._uses_fixed_tail_ptrpg_rollout():
                    next_snode, _ = self.create_Snode(next_state, snode.depth + 1, anode.stn, anode)
                    from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
                        crossed_cutoff,
                        elapsed_from_root,
                        node_remaining,
                    )

                    child_rem = node_remaining(self.mdp, next_state, anode.stn)
                    child_elapsed = elapsed_from_root(self._fixed_tail_ctx, child_rem)
                    if crossed_cutoff(self._fixed_tail_ctx, child_elapsed):
                        leaf_val = self._fixed_tail_post_step_child_value(
                            snode, next_snode, anode
                        )
                        reward += self.mdp.discount_factor * leaf_val
                    elif self._uses_fixed_tail_expectimax():
                        leaf_val = self._fixed_tail_expectimax_v_at_snode(next_snode)
                        reward += self.mdp.discount_factor * leaf_val
                    else:
                        reward += self.mdp.discount_factor * self.selection_max(next_snode)
                    anode.add_child(next_snode)
                else:
                    next_snode, snode_reward = self.create_Snode_max(next_state, snode.depth + 1, anode.stn, anode)
                    reward += snode_reward
                    anode.add_child(next_snode)

        anode.update(reward)
        max_v = snode.max_update()

        return max_v

    def selection_root_interval(self, snode: "up.engines.C_Snode", root_STNnode: "up.plans.stn.STNPlanNode" = None):
        """
        Traverse the tree until reaching a leaf node.
        Selection with root interval logic -
        set the value per root action legal interval.
        The value is propagated and updated according the legal interval
        """
        if self._uses_fixed_tail_ptrpg_rollout():
            if getattr(snode, "_fixed_tail_bootstrap", False):
                val = float(getattr(snode, "_fixed_tail_value", 0.0))
                if root_STNnode is None:
                    return val
                return val, *snode.parent.stn.get_legal_interval(root_STNnode)

        if len(snode.possible_actions) == 0:
            if root_STNnode is None:
                return 0.0 if self._uses_fixed_tail_ptrpg_rollout() else 0
            zero = 0.0 if self._uses_fixed_tail_ptrpg_rollout() else 0
            return zero, *snode.parent.stn.get_legal_interval(root_STNnode)

        if snode.depth > self.search_depth:
            cutoff = self._depth_cutoff_value(snode)
            return cutoff, *snode.parent.stn.get_legal_interval(root_STNnode)

        if self._uses_fixed_tail_ptrpg_rollout() and self._fixed_tail_at_cutoff(snode):
            val = self._fixed_tail_bootstrap_at_snode(snode, label="cutoff_node", crossed=True)
            return val, *snode.parent.stn.get_legal_interval(root_STNnode)

        if self._uses_fixed_tail_expectimax() and not self._fixed_tail_at_cutoff(snode):
            self._fixed_tail_seed_expectimax_q(snode)

        explore_constant = self.exploration_constant
        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        reward = self._terminal_backup_reward(terminal, next_state, reward)

        anode = snode.children[action]
        if root_STNnode is None:
            root_STNnode = anode.STNNode

        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                next_reward, lower, upper = self.selection_root_interval(snodes[next_state], root_STNnode)
                reward += next_reward * self.mdp.discount_factor

            else: # leaf
                next_snode, _ = self.create_Snode_root_interval(next_state, snode.depth + 1, anode.stn, anode)
                lower, upper = anode.stn.get_legal_interval(root_STNnode)
                if self.value_mode == "greedy_matched":
                    reward = self._greedy_matched_action_target(snode, action)
                    anode.add_child(next_snode)
                    next_snode.update(reward, lower, upper)
                elif self._uses_ptrpg_guided_rollout():
                    reward += self._leaf_rollout_value(next_snode) * self.mdp.discount_factor
                    anode.add_child(next_snode)
                    next_snode.update(reward, lower, upper)
                elif self._uses_fixed_tail_ptrpg_rollout():
                    from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
                        crossed_cutoff,
                        elapsed_from_root,
                        node_remaining,
                    )

                    child_rem = node_remaining(self.mdp, next_state, anode.stn)
                    child_elapsed = elapsed_from_root(self._fixed_tail_ctx, child_rem)
                    if crossed_cutoff(self._fixed_tail_ctx, child_elapsed):
                        leaf_val = self._fixed_tail_post_step_child_value(
                            snode, next_snode, anode
                        )
                        reward += leaf_val * self.mdp.discount_factor
                        anode.add_child(next_snode)
                    elif self._uses_fixed_tail_expectimax():
                        anode.add_child(next_snode)
                        leaf_val = self._fixed_tail_expectimax_v_at_snode(next_snode)
                        reward += leaf_val * self.mdp.discount_factor
                    else:
                        anode.add_child(next_snode)
                        next_reward, lower, upper = self.selection_root_interval(
                            next_snode, root_STNnode
                        )
                        reward += next_reward * self.mdp.discount_factor
                        next_snode.update(reward, lower, upper)
                else:
                    reward += self.heuristic(next_snode) * self.mdp.discount_factor
                    anode.add_child(next_snode)
                    next_snode.update(reward, lower, upper)

        else:
            # when the state is terminal set the lower and upper according to anode root
            lower, upper = anode.stn.get_legal_interval(root_STNnode)

        anode.update(reward, lower, upper)
        snode.update(reward, lower, upper)
        return reward, lower, upper


    def selection_root_interval_max(self, snode: "up.engines.C_Snode", root_STNnode: "up.plans.stn.STNPlanNode" = None):
        if len(snode.possible_actions) == 0:
            if root_STNnode is None:
                return -100
            # Stop when there are no possible actions to take so the plan remains consistent
            return LinkedListNode(*snode.parent.stn.get_legal_interval(root_STNnode), - 100)

        if snode.depth > self.search_depth:
            # Stop if the search depth is reached
            return LinkedListNode(*snode.parent.stn.get_legal_interval(root_STNnode), self.heuristic(snode))
        explore_constant = self.exploration_constant
        # Choose a consistent action
        action = self.uct(snode, explore_constant)
        terminal, next_state, reward = self.mdp.step(snode.state, action)
        anode = snode.children[action]
        if root_STNnode is None:
            root_STNnode = anode.STNNode

        if not terminal:
            snodes = anode.children
            if next_state in snodes:
                backup_node = self.selection_root_interval_max(snodes[next_state], root_STNnode)
                backup_node.update_df_reward(self.mdp.discount_factor, reward)

            else:
                next_snode, backup_node = self.create_Snode_root_interval_max(next_state, snode.depth + 1, anode.stn, anode, root_STNnode=root_STNnode)
                backup_node.update_df_reward(self.mdp.discount_factor, reward) #TODO: should it be with discount reward
                anode.add_child(next_snode)

        else:
            backup_node = LinkedListNode(*anode.stn.get_legal_interval(root_STNnode), reward)

        anode.update(None, backup_node)
        backup_node = snode.max_update(backup_node)
        return backup_node

    def heuristic(self, snode: "up.engines.C_SNode"):
        current_time = 0
        lower_bounds = None
        if snode.parent:
            current_time = snode.parent.stn.get_current_end_time()
            lower_bounds = snode.parent.stn.get_lower_bound_potential_end_action()
        if _uses_tprpg_family(self.heuristic_name):
            strategy_for_value = self.temporal_heuristic_strategy
            aligned_override = None
            if _is_option_a_strategy(self.temporal_heuristic_strategy):
                # frontier_aligned_option_a_*: backprop uses raw suffix PTRPG only.
                strategy_for_value = _option_a_ptrpg_suffix(self.temporal_heuristic_strategy)
            elif self.temporal_heuristic_strategy in _FRONTIER_ALIGNED_SUFFIX:
                # Option A: the node's backed-up value is STANDARD PTRPG (the raw
                # suffix strategy), NOT the aligned value. The frontier-aligned
                # value is computed separately via _frontier_aligned_value and is
                # used ONLY for selection — never backpropagated.
                strategy_for_value = _FRONTIER_ALIGNED_SUFFIX[self.temporal_heuristic_strategy]
            elif (
                self.temporal_heuristic_strategy in _ROLLOUT_ALIGNED_SUFFIX
                and snode.parent is not None
            ):
                # Parent-local H_p (rollout_aligned_* uses the aligned value AS the
                # node value).
                aligned_override = _dynamic_aligned_horizon(
                    self.mdp, snode.parent.parent
                )
            score, _ = _tprpg_heuristic_value(
                self.mdp,
                snode.state,
                current_time,
                self.temporal_heuristic_depth,
                strategy_for_value,
                cached_table=self._root_baseline_cache,
                leaf_heuristic_name=self.heuristic_name,
                aligned_h_override=aligned_override,
            )
            return score
        h = up.engines.heuristics.TRPG(self.mdp, snode.state, current_time)
        return h.get_heuristic(lower_bounds)

    def heuristic_init(self, state, stn, parent_snode=None):
        current_time = stn.get_current_end_time()
        if _uses_tprpg_family(self.heuristic_name):
            aligned_override = None
            strategy = self.temporal_heuristic_strategy
            if _is_option_a_strategy(strategy):
                strategy = _option_a_ptrpg_suffix(strategy)
            elif strategy in _ALIGNED_SUFFIX:
                aligned_override = _dynamic_aligned_horizon(self.mdp, parent_snode)
            score, _ = _tprpg_heuristic_value(
                self.mdp,
                state,
                current_time,
                self.temporal_heuristic_depth,
                strategy,
                cached_table=self._root_baseline_cache,
                leaf_heuristic_name=self.heuristic_name,
                aligned_h_override=aligned_override,
            )
            return score
        h = up.engines.heuristics.TRPG(self.mdp, state, current_time)
        return h.get_heuristic()


def plan(mdp: "up.engines.MDP", steps: int, search_time: int, search_depth: int, exploration_constant: float,
         selection_type='avg', k=10, heuristic_name='trpg', temporal_heuristic_depth=25,
         temporal_heuristic_strategy: str = "baseline", value_mode: str = "tp_mcts",
         final_selection: str = 'q'):
    stn = create_init_stn(mdp)
    root_state = mdp.initial_state()

    reuse = False
    history = []
    previous_action_node = None
    step = 0
    root_node = None
    _use_baseline_cache = (
        heuristic_name == "temporal_probabilistic_rpg"
        and temporal_heuristic_strategy == "baseline_cached"
    )
    baseline_cache_table = None

    while stn.get_current_end_time() <= mdp.deadline():
        print(f"started step {step}")
        mcts = C_MCTS(
            mdp,
            root_node,
            root_state,
            search_depth,
            exploration_constant,
            stn,
            selection_type,
            k,
            previous_action_node,
            heuristic_name=heuristic_name,
            temporal_heuristic_depth=temporal_heuristic_depth,
            temporal_heuristic_strategy=temporal_heuristic_strategy,
            root_baseline_cache=baseline_cache_table,
            value_mode=value_mode,
        )
        action = mcts.search(search_time, selection_type, final_selection)

        if action == -1:
            print("A valid plan is not found")
            return 0, -math.inf

        print(f"Current state is {root_state}")
        print(f"The chosen action is {action.name}")

        terminal, root_state, reward = mcts.mdp.step(root_state, action)

        if reuse and root_state in mcts.root_node.children[action].children:
            root_node = mcts.root_node.children[action].children[root_state]
            root_node.set_depth(0)

        # update STN to include the action
        action_node = mcts.root_node.children[action] if selection_type == 'rootInterval' else None

        previous_action_node = update_stn(stn, action, previous_action_node, type='SetTime', action_node=action_node)

        assert stn.is_consistent()

        print(f"The time of the plan so far: {stn.get_current_end_time()}")
        history.append(previous_action_node)

        if terminal:
            print(f"Current state is {root_state}")
            print(f"The amount of time the plan took: {stn.get_current_end_time()}")
            return 1, stn.get_current_end_time()

        # Advance the incremental PTRPG cache to the real successor state so the
        # next MCTS round starts with a warm baseline_cached table.
        if _use_baseline_cache:
            current_time = stn.get_current_end_time()
            _, baseline_cache_table = _tprpg_heuristic_value(
                mdp,
                root_state,
                current_time,
                temporal_heuristic_depth,
                temporal_heuristic_strategy,
                cached_table=baseline_cache_table,
                leaf_heuristic_name=heuristic_name,
            )

        step += 1

    print("A valid plan is not found")
    return 0, -math.inf


def combination_plan(mdp: "up.engines.MDP", split_mdp: "up.engines.MDP", steps: int, search_time: int,
                     search_depth: int, exploration_constant: float,
                     selection_type='avg', k=10, heuristic_name='trpg', temporal_heuristic_depth=25,
                     temporal_heuristic_strategy: str = "baseline", value_mode: str = "tp_mcts"):
    del value_mode
    root_state = mdp.initial_state()
    history = []
    step = 0
    root_node = None
    _use_baseline_cache = (
        heuristic_name == "temporal_probabilistic_rpg"
        and temporal_heuristic_strategy == "baseline_cached"
    )
    baseline_cache_table = None

    while root_state.current_time < mdp.deadline():
        print(f"started step {step}")

        mcts = MCTS(
            mdp,
            split_mdp,
            root_node,
            root_state,
            search_depth,
            exploration_constant,
            selection_type,
            k,
            heuristic_name=heuristic_name,
            temporal_heuristic_depth=temporal_heuristic_depth,
            temporal_heuristic_strategy=temporal_heuristic_strategy,
            root_baseline_cache=baseline_cache_table,
        )
        action = mcts.search(search_time, selection_type)

        print(f"Current state is {root_state}")
        print(f"The chosen action is {action.name}")

        terminal, root_state, reward = mcts.mdp.step(root_state, action)

        history.append(action)
        print(f'current time = {root_state.current_time}')

        if terminal and root_state.current_time <= mdp.deadline():
            print(f"Current state is {root_state}")
            print(f"The amount of time the plan took: {root_state.current_time}")
            return 1, root_state.current_time

        # Advance the incremental PTRPG cache to the real successor state so the
        # next MCTS round starts with a warm baseline_cached table.
        if _use_baseline_cache:
            current_time = root_state.current_time
            _, baseline_cache_table = _tprpg_heuristic_value(
                split_mdp,
                root_state,
                current_time,
                temporal_heuristic_depth,
                temporal_heuristic_strategy,
                cached_table=baseline_cache_table,
                leaf_heuristic_name=heuristic_name,
            )

        step += 1

    return 0, -math.inf
