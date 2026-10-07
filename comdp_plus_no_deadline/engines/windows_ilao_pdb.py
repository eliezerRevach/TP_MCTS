"""``windows_ilao_pdb``: MCTS leaf heuristic = a table of the time-left windows MDP
per goal pattern: the whole pattern at every r where it fits, ILAO* (guided by the
survivor sweep, the RPG PDB of rpg_exact_states_v18) where it does not.

Spec: ``artifacts/Windows_ILAO_PDB.docx``.

    offline : one pattern per goal group, TP_MCTS_WILAO_OFFLINE_SECONDS in total.
              0. The pattern's facts (cegar_pattern.py): start from the goal, build
                 its table, replay the table's policy in the real model from the
                 initial state, add the fact whose precondition fails there (positive
                 or negative), repeat until nothing fails or PATTERN_FACTS is reached.
                 The last table that finished is the pattern's table.
              1. The full table (``WindowsTable``) of every pattern: every state
                 reachable from the initial state, built forward once without r, then
                 V(s, r) for every r <= deadline backward from the goals, starting at 0
                 (no heuristic, so no loop can hold a made-up value). Value iteration,
                 no ILAO*. Each pattern gets an equal share of 90% of the budget.
              2. Only for a pattern whose table did not finish: ILAO* from the initial
                 state until its best policy is solved, then (time left) the branches
                 that policy did not take.
              Budget <= 0: no limit -- every table is built to the end (8 facts can
              take far longer than minutes).
              TP_MCTS_WILAO_FULL_TABLE = 0 skips step 1 (ILAO* only, for comparison).
    online  : per MCTS leaf, per pattern, a LOOKUP:
                table_exact  the leaf's state is in the full table
                table_cover  a table state with the same facts and end order that covers
                             the leaf some time d after its event (see WindowsTable):
                             V(s, r + d_max), an upper bound
                exact/cover  the same in an ILAO* table (patterns without a full table)
                miss         TP_MCTS_WILAO_MISS = one  -> 1.0 (optimistic)
                                                  lazy -> ILAO* from the leaf that stops
                                                          at solved states, within the
                                                          query budget
              running actions take their remaining time from the search's STN. With
              TP_MCTS_WILAO_NO_STN_PASS = 1 it is ignored: every running action gets the
              window [0, d] with a free end (e = 0) and every end order -- a looser upper
              bound, but starting a pattern action raises the value at once (with exact
              times it cannot: the value already counts starting it now). The table has
              no state that loose, so these lookups go to the miss path.
    value   : two settings
                TP_MCTS_WILAO_AGG       how patterns are combined: min | avg | same
                                        (all sound: each value >= P(its goal) >= P(all));
                                        same = goals that can influence each other share ONE
                                        pattern (fact_pattern.independent_goal_groups), so their
                                        shared hands / stores / time are inside one table; the
                                        groups are then combined by product (PATTERN_GOALS ignored);
                                        min is tightest, avg has more gradient
                TP_MCTS_WILAO_GROUPING  0: AGG over all patterns
                                        1: the goals are split into groups that share no
                                           action and no changed fact in the real model
                                           (fact_pattern.independent_goal_groups); AGG
                                           inside each group, PRODUCT across groups --
                                           sound because the groups cannot influence each
                                           other: P(all) <= product of P(each group)
              So: min | avg | product of group mins | product of group avgs.
              Legacy TP_MCTS_WILAO_AGGREGATION (only when both are unset): min, avg, and
              groups (= avg + grouping).

A pattern is a group of goal facts plus every action that can achieve or threaten
them (``goal_relevance_closure``). The relaxed action table of the sweep is built
once per pattern, at the initial state.

Knobs (environment, set from experiments.ipynb):
    TP_MCTS_WILAO_OFFLINE_SECONDS   total offline budget; <= 0 = no limit   (30)
    TP_MCTS_WILAO_FULL_TABLE        full table per pattern; 0 = ILAO* only  (1)
    TP_MCTS_WILAO_OFFLINE_EXTEND    ILAO* extend for patterns w/o a table   (1)
    TP_MCTS_WILAO_PATTERN_GOALS     goal facts per pattern; 0 = all goals   (1)
    TP_MCTS_WILAO_PATTERN_FACTS     at most this many facts per pattern; 0 = no
                                    cap: every relevant action on full states (8)
    TP_MCTS_WILAO_PATTERN_GROWTH    cegar (counterexamples, cegar_pattern.py) |
                                    static (backward ranking, fact_pattern.py) |
                                    logic (untimed counterexamples from the initial
                                    state + rollout states, logic_cegar.py) |
                                    pairs (a COLLECTION per goal group: the group's goals
                                    + one shared resource each, fact_pattern.
                                    resource_patterns; min inside a group) |
                                    llm_by_hand_nasa_two (facts picked by hand for
                                    nasa_rover obj 2, hand_patterns.py)      (cegar)
    TP_MCTS_WILAO_LOGIC_SOURCES     logic only: rollouts random | greedy     (random)
    TP_MCTS_WILAO_LOGIC_ROLLOUTS    logic only: rollouts per round           (20)
    TP_MCTS_WILAO_LOGIC_DEPTH       logic only: steps per rollout            (60)
    TP_MCTS_WILAO_LOGIC_RANK        logic only: frequency | restrict (facts most
                                    actions need first)                      (frequency)
    TP_MCTS_WILAO_MISS              one | lazy                              (lazy)
    TP_MCTS_WILAO_QUERY_SECONDS     lazy budget per pattern per leaf        (0.05)
    TP_MCTS_WILAO_QUERY_EXPANSIONS  lazy expansions per pattern per leaf    (200)
    TP_MCTS_WILAO_AGG               min | avg | same                        (min)
    TP_MCTS_WILAO_GROUPING          1 = product across independent groups   (1)
    TP_MCTS_WILAO_NO_STN_PASS       1 = ignore the STN's remaining times    (0)
    TP_MCTS_WILAO_EXACT_STATICS     static facts (changed by no action) keep their
                                    initial value in every pattern; 0 = old
                                    relaxation (h1 counted as a good hand)   (1)
    TP_MCTS_WILAO_TRANSITION_READS  facts an outcome probability reads (push reads
                                    rock_under_car, no precondition) count as read
                                    facts: grouping, pairs resources and growth add
                                    them first; when one is still outside a pattern,
                                    every value of it is its own option and VI takes
                                    the best (fact_pattern.outcome_variants), so h
                                    stays an upper bound; 0 = old (outside facts at
                                    their initial value -- can go BELOW V*)  (1)
    TP_MCTS_WILAO_BLUR              window grid eps: lo down / hi up to eps, e exact
                                    (fewer states, still an upper bound;
                                    windows_lao.py "Blur"); 0 = off          (0)
    TP_MCTS_WILAO_TABLE_CACHE       folder for built tables; empty = off     ('')
                                    The first process with these settings builds
                                    and saves; every other one waits for the file
                                    (a lock), then LOADS its own private copy --
                                    parallel workers build once, not once each.
                                    Lookups never touch the file or the lock.
    TP_MCTS_WILAO_CACHE_STALE       seconds after which a lock whose builder
                                    left no file is taken over               (21600)
    TP_MCTS_WILAO_REPORT            print a summary at exit: 1 | 0          (1)
"""

from __future__ import annotations

import atexit
import hashlib
import os
import pickle
import time
from fractions import Fraction
from collections import Counter
from typing import Dict, List, Optional

from comdp_plus_no_deadline.engines.cegar_pattern import cegar_pattern
from comdp_plus_no_deadline.engines.hand_patterns import HAND_PATTERNS, hand_patterns
from comdp_plus_no_deadline.engines.logic_cegar import logic_cegar_patterns
from comdp_plus_no_deadline.engines.fact_pattern import (OPAQUE_EVENTS, FactPatternModel, exact_statics_default,
                                                         grow_pattern, independent_goal_groups, resource_patterns,
                                                         transition_reads_default)
from comdp_plus_no_deadline.engines.survivor_sweep import relaxed_actions_from_engine
from comdp_plus_no_deadline.engines.temporal_stn_pdb import EngineModel, goal_relevance_closure
from comdp_plus_no_deadline.engines.windows_lao import WindowsLAO, WindowsTable, blur_default


# Legacy single setting -> (AGG, GROUPING).
_LEGACY_AGGREGATION = {"min": ("min", False), "avg": ("avg", False), "groups": ("avg", True)}


def _combine(values, how: str) -> float:
    if how == "avg":
        return sum(values) / len(values)
    return min(values)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _compact_number(x):
    """A whole Fraction as an int: equal value and hash, and small ints are shared by Python,
    so a loaded table's windows cost almost nothing."""
    if isinstance(x, Fraction) and x.denominator == 1:
        return int(x)
    return x


def _compact(windows):
    return tuple(tuple(_compact_number(v) for v in w) for w in windows)


def _take_lock(lock: str, stale_seconds: float) -> bool:
    """Create the lock file; True if this process now holds it. A lock older than
    ``stale_seconds`` (its builder died) is removed and taken."""
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock) > stale_seconds:
                os.remove(lock)
                return _take_lock(lock, stale_seconds)
        except OSError:
            pass
        return False
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{os.getpid()} {time.time()}\n")
    return True


def _release_lock(lock: str) -> None:
    try:
        os.remove(lock)
    except OSError:
        pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return default


class WindowsILAOPDBHeuristic:
    """One ILAO* table per pattern, solved offline, read (and lazily extended) online."""

    def __init__(self, mdp):
        self._mdp = mdp
        self.offline_seconds = _env_float("TP_MCTS_WILAO_OFFLINE_SECONDS", 30.0)
        self.pattern_goals = _env_int("TP_MCTS_WILAO_PATTERN_GOALS", 1)
        self.miss_mode = (os.environ.get("TP_MCTS_WILAO_MISS") or "lazy").strip().lower()
        self.query_seconds = _env_float("TP_MCTS_WILAO_QUERY_SECONDS", 0.05)
        self.query_expansions = _env_int("TP_MCTS_WILAO_QUERY_EXPANSIONS", 200)
        agg = (os.environ.get("TP_MCTS_WILAO_AGG") or "").strip().lower()
        grouping = os.environ.get("TP_MCTS_WILAO_GROUPING")
        if not agg and grouping is None:
            legacy = (os.environ.get("TP_MCTS_WILAO_AGGREGATION") or "min").strip().lower()
            if legacy not in _LEGACY_AGGREGATION:
                raise ValueError(f"TP_MCTS_WILAO_AGGREGATION={legacy!r}: use TP_MCTS_WILAO_AGG (min | avg) "
                                 f"and TP_MCTS_WILAO_GROUPING (0 | 1)")
            agg, grouping = _LEGACY_AGGREGATION[legacy]
        self.agg = agg or "min"
        if self.agg not in ("min", "avg", "same"):
            raise ValueError(f"TP_MCTS_WILAO_AGG={self.agg!r}: use min | avg | same")
        text = str(grouping if grouping is not None else "1").strip().lower()
        if text not in ("0", "1", "true", "false"):
            raise ValueError(f"TP_MCTS_WILAO_GROUPING={grouping!r}: use 0 | 1")
        self.grouping = text in ("1", "true") or self.agg == "same"     # same: product across the groups
        self.extend_offline = bool(_env_int("TP_MCTS_WILAO_OFFLINE_EXTEND", 1))
        self.full_table = bool(_env_int("TP_MCTS_WILAO_FULL_TABLE", 1))
        self.growth = (os.environ.get("TP_MCTS_WILAO_PATTERN_GROWTH") or "cegar").strip().lower()
        if self.growth not in ("cegar", "static", "logic", "pairs") and self.growth not in HAND_PATTERNS:
            raise ValueError(f"TP_MCTS_WILAO_PATTERN_GROWTH={self.growth!r}: use cegar | static | logic | pairs | "
                             + " | ".join(HAND_PATTERNS))
        self.logic_sources = (os.environ.get("TP_MCTS_WILAO_LOGIC_SOURCES") or "random").strip().lower()
        if self.logic_sources not in ("random", "greedy"):
            raise ValueError(f"TP_MCTS_WILAO_LOGIC_SOURCES={self.logic_sources!r}: use random | greedy")
        self.logic_rollouts = _env_int("TP_MCTS_WILAO_LOGIC_ROLLOUTS", 20)
        self.logic_depth = _env_int("TP_MCTS_WILAO_LOGIC_DEPTH", 60)
        self.logic_rank = (os.environ.get("TP_MCTS_WILAO_LOGIC_RANK") or "frequency").strip().lower()
        if self.logic_rank not in ("frequency", "restrict"):
            raise ValueError(f"TP_MCTS_WILAO_LOGIC_RANK={self.logic_rank!r}: use frequency | restrict")
        self.pattern_facts = _env_int("TP_MCTS_WILAO_PATTERN_FACTS", 8)
        self.no_stn_pass = bool(_env_int("TP_MCTS_WILAO_NO_STN_PASS", 0))
        self.patterns: List[dict] = []
        self.groups: List[List[int]] = []
        self.counts: Counter = Counter()
        self.query_seconds_total = 0.0
        self.offline_report: Dict[str, object] = {}
        self._built = False
        if _env_int("TP_MCTS_WILAO_REPORT", 1):
            atexit.register(self.print_report)

    @classmethod
    def from_mdp(cls, mdp) -> "WindowsILAOPDBHeuristic":
        return cls(mdp)

    # -- offline -------------------------------------------------------------
    def build(self) -> None:
        """Build the tables once per process; with TP_MCTS_WILAO_TABLE_CACHE, once per setting:
        load the saved tables if they exist, else build under a lock (or wait for the
        process holding it) and save. Online lookups only ever read this process's copy."""
        if self._built:
            return
        self._built = True
        cache_dir = (os.environ.get("TP_MCTS_WILAO_TABLE_CACHE") or "").strip()
        if not cache_dir:
            self._build_fresh()
            self._drop_build_data()
            return
        os.makedirs(cache_dir, exist_ok=True)
        key = self._cache_key()
        path = os.path.join(cache_dir, f"wilao_{key}.pkl")
        lock = path + ".lock"
        waited = 0.0
        if not os.path.exists(path):
            stale = _env_float("TP_MCTS_WILAO_CACHE_STALE", 21600.0)
            t0 = time.perf_counter()
            while not os.path.exists(path):
                if _take_lock(lock, stale):
                    try:
                        if not os.path.exists(path):          # another process may have finished meanwhile
                            self._build_fresh()
                            self._drop_build_data()
                            saved = self._save_cache(path)
                            self.offline_report["cache"] = {"file": path, "hit": False, "saved": saved,
                                                            "waited_seconds": round(time.perf_counter() - t0, 1)}
                            return
                    finally:
                        _release_lock(lock)
                    break
                time.sleep(2.0)
            waited = time.perf_counter() - t0
        if self._load_cache(path):
            self.offline_report["cache"] = {"file": path, "hit": True, "waited_seconds": round(waited, 1)}
            return
        # unreadable or incompatible file: build here, without saving over it
        self._build_fresh()
        self.offline_report["cache"] = {"file": path, "hit": False, "saved": "no: the cached file did not load"}
        self._drop_build_data()

    def _build_fresh(self) -> None:
        started = time.perf_counter()
        base = EngineModel(self._mdp)
        goals = sorted(self._mdp.problem.goals, key=str)
        size = len(goals) if self.pattern_goals <= 0 else self.pattern_goals
        if self.agg == "same" or self.growth == "pairs":
            # one pattern per group of goals that can influence each other; independent groups apart
            groups = [sorted(g, key=str) for g in independent_goal_groups(base.ops(), goals)] or [goals]
        else:
            groups = [goals[i:i + size] for i in range(0, len(goals), size)] or [goals]
        deadline = int(float(self._mdp.deadline()))
        facts = frozenset(self._mdp.initial_state().predicates)
        unlimited = self.offline_seconds <= 0
        offline_end = None if unlimited else started + self.offline_seconds
        # The tables share 90% of the budget; the rest (and whatever they leave) is
        # kept for ILAO* on the patterns whose table did not finish.
        table_end = None if unlimited else started + 0.9 * self.offline_seconds

        def share(patterns_left: int, end) -> Optional[float]:
            """An equal share of the time still left: a quick pattern leaves its
            unused time to the slower ones after it."""
            if unlimited:
                return None
            return max(0.0, end - time.perf_counter()) / patterns_left

        cegar = self.growth == "cegar" and self.pattern_facts > 0 and self.full_table
        logic = None
        if self.growth == "logic" and self.pattern_facts > 0:
            # Facts by untimed counterexamples (logic_cegar.py), all groups together; tables come in phase 1.
            t0 = time.perf_counter()
            logic = logic_cegar_patterns(self._mdp, base, groups, facts, max(self.pattern_facts, max(map(len, groups))),
                                         sources=self.logic_sources, rollouts=self.logic_rollouts,
                                         depth=self.logic_depth, rank=self.logic_rank)
            self.logic_seconds = round(time.perf_counter() - t0, 2)
        # pairs: several patterns per group (one per shared resource); otherwise one per group
        if self.growth in HAND_PATTERNS:
            entries = hand_patterns(self.growth, self._facts_by_name(base), goals)
        elif self.growth == "pairs":
            entries = [(g, f, r) for g, f, r in resource_patterns(base.ops(), groups, facts)]
        else:
            entries = [(group, None, None) for group in groups]
        for index, (group, fixed_phi, resource) in enumerate(entries):
            grown = None
            if fixed_phi is not None:
                phi = list(fixed_phi)
                model = FactPatternModel(base, phi, group, facts)
            elif cegar:
                # Facts added by counterexamples (cegar_pattern.py); the table comes with them.
                grown = cegar_pattern(base, group, facts, max(self.pattern_facts, len(group)), deadline,
                                      time_budget=share(len(entries) - index, table_end))
                phi = grown["facts"]
                model = grown["model"] or FactPatternModel(base, phi, group, facts)
            elif logic is not None:
                phi = logic[index]["facts"]
                model = FactPatternModel(base, phi, group, facts)
            elif self.pattern_facts > 0:
                # Fact-capped pattern grown backwards from the goals (fact_pattern.py).
                phi = grow_pattern(base.ops(), group, facts, max(self.pattern_facts, len(group)))
                model = FactPatternModel(base, phi, group, facts)
            else:
                phi = None
                ops = frozenset().union(*(goal_relevance_closure(base, g) for g in group))
                model = EngineModel(self._mdp, allowed_ops=ops, goals=set(group))
            root_facts = model.project(facts) if phi is not None else facts
            solver = WindowsLAO(model, sweep_horizon=deadline,
                                relaxed_actions=relaxed_actions_from_engine(model, root_facts))
            self.patterns.append({
                "goals": [str(g) for g in group],
                "facts": None if phi is None else [str(f) for f in phi],
                "actions": len(model.ops()),
                "model": model,
                "root_facts": root_facts,
                "solver": solver,
                "table": None,
                "table_stats": None,
                "offline_value": None,
                "offline_optimal": None,
                "offline_seconds": None,
                "solved_at_optimal": 0,
                "cegar_log": None if grown is None else grown["log"],
                "logic_stop": None if logic is None else logic[index]["stop"],
                "resource": resource,
                "cegar_seconds": None if grown is None else grown["seconds"],
            })
            if grown is not None and grown["table"] is not None:
                table = grown["table"]
                self.patterns[-1].update(table=table, table_stats=dict(table.stats, built=True),
                                         offline_value=table.lookup(root_facts, deadline)[0],
                                         offline_optimal=True)
        # Phase 1: the full table of every pattern that fits: every state, every r.
        if self.full_table:
            todo = [p for p in self.patterns if p["table"] is None and p["cegar_log"] is None]
            for index, pattern in enumerate(todo):
                table = WindowsTable(pattern["solver"], deadline)
                if table.build(pattern["root_facts"], time_budget=share(len(todo) - index, table_end)):
                    pattern["table"] = table
                    pattern["offline_value"] = table.lookup(pattern["root_facts"], deadline)[0]
                    pattern["offline_optimal"] = True
                pattern["table_stats"] = dict(table.stats, built=table.complete)
        # Phase 2: ILAO* only where there is no table: the optimal policy, then (time
        # left) the branches it did not take, round-robin.
        fallback = [p for p in self.patterns if p["table"] is None]
        for index, pattern in enumerate(fallback):
            solver = pattern["solver"]
            pattern["offline_value"] = solver.solve(pattern["root_facts"], deadline,
                                                    time_budget=share(len(fallback) - index, offline_end))
            pattern["offline_optimal"] = solver.complete
            pattern["offline_seconds"] = solver.stats["seconds"]
            pattern["solved_at_optimal"] = len(solver.solved)
        if self.extend_offline and not unlimited:
            active = [p for p in fallback if p["solver"]._candidates]
            while active and time.perf_counter() < offline_end:
                slice_seconds = (offline_end - time.perf_counter()) / len(active)
                active = [p for p in active if p["solver"].extend(slice_seconds)]
        # Groups of goals that cannot influence each other (used when GROUPING is on).
        group_of = {}
        for gi, members in enumerate(independent_goal_groups(base.ops(), goals)):
            for g in members:
                group_of[str(g)] = gi
        merged: Dict[int, List[int]] = {}
        for index, pattern in enumerate(self.patterns):
            key = min(group_of.get(g, -1 - index) for g in pattern["goals"])
            merged.setdefault(key, []).append(index)
        self.groups = list(merged.values())
        tables = [p["table_stats"] or {} for p in self.patterns]
        self.offline_report = {
            "patterns": len(self.patterns),
            "pattern_facts": [len(p["facts"]) if p["facts"] is not None else "all" for p in self.patterns],
            "growth": self.growth if self.growth in HAND_PATTERNS else (
                "cegar" if cegar else ("logic" if logic is not None else
                                       ("pairs" if self.growth == "pairs" else "static"))),
            "resources": [p["resource"] for p in self.patterns],
            "logic": None if logic is None else {"sources": self.logic_sources, "rollouts": self.logic_rollouts,
                                                 "depth": self.logic_depth, "rank": self.logic_rank, "seconds": self.logic_seconds,
                                                 "stop": [p["logic_stop"] for p in self.patterns]},
            "cegar_stop": [p["cegar_log"][-1].get("stop") if p["cegar_log"] else None for p in self.patterns],
            "cegar_seconds": [p["cegar_seconds"] for p in self.patterns],
            "pattern_fact_names": [p["facts"] for p in self.patterns],
            "pattern_actions": [p["actions"] for p in self.patterns],
            "seconds": round(time.perf_counter() - started, 2),
            "budget": "none" if unlimited else self.offline_seconds,
            "agg": self.agg,
            "grouping": self.grouping,
            "no_stn_pass": self.no_stn_pass,
            "exact_statics": exact_statics_default(),
            "transition_reads": transition_reads_default(),
            "hidden_read_events": [sorted(getattr(p["model"], "variant_events", ())) if p.get("model") is not None
                                   else [] for p in self.patterns],
            "opaque_events": sorted(OPAQUE_EVENTS),
            "blur": str(blur_default()),
            "groups": [[self.patterns[i]["goals"][0] for i in members] for members in self.groups],
            "values": [None if p["offline_value"] is None else round(p["offline_value"], 6)
                       for p in self.patterns],
            "optimal": [p["offline_optimal"] for p in self.patterns],
            "ilao_seconds_to_optimal": [p["offline_seconds"] for p in self.patterns],
            "table_built": [t.get("built", False) for t in tables],
            "table_states": [t.get("states", t.get("states_when_cut")) for t in tables],
            "table_seconds": [t.get("seconds", t.get("forward_seconds")) for t in tables],
            "solved_after_extend": [len(p["solver"].solved) for p in self.patterns],
            "branches_left": [len(p["solver"]._candidates) for p in self.patterns],
        }

    # -- table cache -----------------------------------------------------------
    _CACHE_VERSION = 3

    def _cache_key(self) -> str:
        """The problem (actions, initial state, goals, deadline) and every setting that
        changes the tables: a different setting is a different file."""
        h = hashlib.sha1()
        base = EngineModel(self._mdp)
        for key in sorted(base.ops()):
            op = base.ops()[key]
            h.update(f"{key}|{getattr(op, 'duration', '')}".encode())
            for ev in (op.start_action, op.end_action):
                if ev is None:
                    h.update(b"-")
                    continue
                for part in (ev.pos_preconditions, getattr(ev, "neg_preconditions", ()), ev.add_effects,
                             ev.del_effects):
                    h.update(("|" + ",".join(sorted(str(f) for f in part))).encode())
                for pe in getattr(ev, "probabilistic_effects", []) or []:
                    h.update(("|" + repr(pe)).encode())
        h.update(("|init:" + ",".join(sorted(str(f) for f in self._mdp.initial_state().predicates))).encode())
        h.update(("|goals:" + ",".join(sorted(str(g) for g in self._mdp.problem.goals))).encode())
        h.update(f"|deadline:{self._mdp.deadline()}".encode())
        settings = (self._CACHE_VERSION, self.growth, self.pattern_facts, self.pattern_goals, self.agg,
                    self.grouping, self.full_table, self.offline_seconds, self.extend_offline,
                    exact_statics_default(), transition_reads_default(), str(blur_default()),
                    self.logic_sources, self.logic_rollouts,
                    self.logic_depth, self.logic_rank)
        h.update(repr(settings).encode())
        return h.hexdigest()[:16]

    def _save_cache(self, path: str):
        """Lookup data only (index, values, tmin, grid); every pattern needs a full table."""
        if not self.patterns or any(p["table"] is None or not p["table"].complete or p["facts"] is None
                                    for p in self.patterns):
            return "no: a pattern has no full table"
        names: Dict[object, frozenset] = {}

        def name_of(fs):
            out = names.get(fs)
            if out is None:
                out = names[fs] = frozenset(str(f) for f in fs)
            return out

        head = {"version": self._CACHE_VERSION, "groups": self.groups,
                "offline_report": self.offline_report, "patterns": len(self.patterns)}
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(head, fh, protocol=pickle.HIGHEST_PROTOCOL)
            for p in self.patterns:                 # one record per pattern: one renamed index at a time
                t = p["table"]
                pickle.dump({
                    "goals": p["goals"], "facts": p["facts"], "resource": p.get("resource"),
                    "actions": p["actions"], "table_stats": p["table_stats"], "offline_value": p["offline_value"],
                    "offline_optimal": p["offline_optimal"], "logic_stop": p.get("logic_stop"),
                    "cegar_log": p.get("cegar_log"), "cegar_seconds": p.get("cegar_seconds"),
                    "index": {(name_of(f), q, _compact(w)): i for (f, q, w), i in t.index.items()},
                    "V": t.V, "tmin": [_compact_number(x) for x in t.tmin], "grid": t.grid,
                    "horizon": t.horizon, "stats": t.stats}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)                       # atomic: a reader never sees half a file
        return True

    def _facts_by_name(self, base) -> Dict[str, object]:
        """Every fact of the problem (initial state, goals, anything an action reads or writes) by name."""
        initial = frozenset(self._mdp.initial_state().predicates)
        fact = {str(f): f for f in initial | set(self._mdp.problem.goals)}
        for op in base.ops().values():
            for ev in (op.start_action, op.end_action):
                if ev is None:
                    continue
                for f in (set(ev.pos_preconditions) | set(getattr(ev, "neg_preconditions", ())) |
                          set(ev.add_effects) | set(ev.del_effects)):
                    fact.setdefault(str(f), f)
                for pe in getattr(ev, "probabilistic_effects", []) or []:
                    for f in getattr(pe, "fluents", []) or []:
                        fact.setdefault(str(f), f)
        return fact

    def _load_cache(self, path: str) -> bool:
        try:
            with open(path, "rb") as fh:
                head = pickle.load(fh)
                if not isinstance(head, dict) or head.get("version") != self._CACHE_VERSION:
                    return False
                records = [pickle.load(fh) for _ in range(head["patterns"])]
        except Exception:
            return False
        base = EngineModel(self._mdp)
        deadline = int(float(self._mdp.deadline()))
        initial = frozenset(self._mdp.initial_state().predicates)
        fact = self._facts_by_name(base)
        patterns = []
        for rec in records:
            try:
                phi = [fact[x] for x in rec["facts"]]
                goal_objs = [fact[x] for x in rec["goals"]]
            except KeyError:
                return False
            model = FactPatternModel(base, phi, goal_objs, initial)
            root_facts = model.project(initial)
            solver = WindowsLAO(model, sweep_horizon=deadline,
                                relaxed_actions=relaxed_actions_from_engine(model, root_facts))
            table = WindowsTable(solver, deadline)
            facts_of: Dict[frozenset, frozenset] = {}
            shared: Dict[object, object] = {}
            index = {}
            for (fs, q, w), i in rec["index"].items():
                f2 = facts_of.get(fs)
                if f2 is None:
                    f2 = facts_of[fs] = frozenset(fact[x] for x in fs)
                index[(f2, shared.setdefault(q, q), shared.setdefault(w, w))] = i
            table.index, table.V, table.tmin = index, rec["V"], rec["tmin"]
            table.grid, table.horizon, table.stats = rec["grid"], rec["horizon"], rec["stats"]
            for (f2, q, w), k in index.items():
                table._by_fq.setdefault((f2, q), []).append((w, k))
            table.complete = True
            patterns.append({
                "goals": rec["goals"], "facts": rec["facts"], "actions": rec["actions"], "model": model,
                "root_facts": root_facts, "solver": solver, "table": table, "table_stats": rec["table_stats"],
                "offline_value": rec["offline_value"], "offline_optimal": rec["offline_optimal"],
                "offline_seconds": None, "solved_at_optimal": 0, "cegar_log": rec["cegar_log"],
                "logic_stop": rec["logic_stop"], "resource": rec["resource"], "cegar_seconds": rec["cegar_seconds"]})
        self.patterns, self.groups = patterns, head["groups"]
        self.offline_report = dict(head["offline_report"])
        return True

    def _drop_build_data(self) -> None:
        """Lookups read only index, V, tmin and the cover index: free the build graph."""
        for p in self.patterns:
            t = p.get("table")
            if t is not None and t.complete:
                t.options = None
                t.states = None

    # -- online --------------------------------------------------------------
    def heuristic_score(self, state, goal_facts=(), fixed_depth: int = 25, start_time: float = 0.0,
                        running_remaining: Optional[Dict[str, float]] = None, **_ignored) -> float:
        self.build()
        started = time.perf_counter()
        self.counts["queries"] += 1
        if self.no_stn_pass:
            running_remaining = None
        preds = getattr(state, "predicates", None)
        facts = frozenset(preds) if preds is not None else frozenset(state)
        r = max(0, int(fixed_depth))
        values = []
        for pattern in self.patterns:
            solver = pattern["solver"]
            running = [(key, None if running_remaining is None else running_remaining.get(key))
                       for key in self._running_keys(facts, solver.ops)]
            local = pattern["model"].project(facts) if pattern["facts"] is not None else facts
            value = None
            if pattern["table"] is not None:
                value, kind = pattern["table"].lookup(local, r, running)
                if value is not None:
                    kind = "table_" + kind
                else:
                    self.counts["table_miss"] += 1
            if value is None:
                value, kind = solver.lookup(local, r, running)
            if value is None:
                if self.miss_mode == "one":
                    value, kind = 1.0, "miss_one"
                else:
                    value = solver.solve(local, r, running, lazy=True, time_budget=self.query_seconds,
                                         max_expansions=self.query_expansions)
                    kind = "lazy_optimal" if solver.complete else "lazy_cut"
            self.counts[kind] += 1
            values.append(value)
        self.query_seconds_total += time.perf_counter() - started
        if not values:
            return 1.0
        if not self.grouping:
            return _combine(values, self.agg)
        out = 1.0
        for members in self.groups:
            out *= _combine([values[i] for i in members], self.agg)
        return out

    def report(self) -> Dict[str, object]:
        lookups = sum(self.counts[k] for k in ("table_exact", "table_cover", "exact", "cover",
                                                "miss_one", "lazy_optimal", "lazy_cut"))
        return {
            "offline": self.offline_report,
            "offline_seconds": self.offline_report.get("seconds"),
            "queries": self.counts["queries"],
            "pattern_lookups": lookups,
            "table_exact": self.counts["table_exact"],
            "table_cover": self.counts["table_cover"],
            "table_miss": self.counts["table_miss"],
            "exact": self.counts["exact"],
            "cover": self.counts["cover"],
            "miss_one": self.counts["miss_one"],
            "lazy_optimal": self.counts["lazy_optimal"],
            "lazy_cut": self.counts["lazy_cut"],
            "ms_per_query": round(1000 * self.query_seconds_total / max(1, self.counts["queries"]), 2),
            "solved_states_now": [len(p["solver"].solved) for p in self.patterns],
        }

    def print_report(self) -> None:
        if self._built:
            print(f"[windows_ilao_pdb] {self.report()}", flush=True)

    @staticmethod
    def _running_keys(facts, ops) -> List[str]:
        """Running actions from the compiled model's inExecution(start-<a>) facts."""
        out = []
        texts = [str(f) for f in facts if "inExecution" in str(f)]
        for key in ops:
            base = key[len("start_"):] if key.startswith("start_") else key
            if any(f"start-{base}" in t for t in texts):
                out.append(key)
        return out
