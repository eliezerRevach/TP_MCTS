"""``windows_ilao_pdb``: MCTS leaf heuristic = a table of the time-left windows MDP
per goal pattern: the whole pattern at every r where it fits, ILAO* (guided by the
survivor sweep, the RPG PDB of rpg_exact_states_v18) where it does not.

Spec: ``artifacts/Windows_ILAO_PDB.docx``.

    offline : one pattern per goal group, TP_MCTS_WILAO_OFFLINE_SECONDS in total.
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
    value   : min over patterns (sound); product only for comparison

A pattern is a group of goal facts plus every action that can achieve or threaten
them (``goal_relevance_closure``). The relaxed action table of the sweep is built
once per pattern, at the initial state.

Knobs (environment, set from experiments.ipynb):
    TP_MCTS_WILAO_OFFLINE_SECONDS   total offline budget; <= 0 = no limit   (30)
    TP_MCTS_WILAO_FULL_TABLE        full table per pattern; 0 = ILAO* only  (1)
    TP_MCTS_WILAO_OFFLINE_EXTEND    ILAO* extend for patterns w/o a table   (1)
    TP_MCTS_WILAO_PATTERN_GOALS     goal facts per pattern; 0 = all goals   (1)
    TP_MCTS_WILAO_PATTERN_FACTS     facts per pattern, grown backwards from
                                    the goals (fact_pattern.py); 0 = no cap:
                                    every relevant action on full states    (8)
    TP_MCTS_WILAO_MISS              one | lazy                              (lazy)
    TP_MCTS_WILAO_QUERY_SECONDS     lazy budget per pattern per leaf        (0.05)
    TP_MCTS_WILAO_QUERY_EXPANSIONS  lazy expansions per pattern per leaf    (200)
    TP_MCTS_WILAO_AGGREGATION       min | product                           (min)
    TP_MCTS_WILAO_REPORT            print a summary at exit: 1 | 0          (1)
"""

from __future__ import annotations

import atexit
import os
import time
from collections import Counter
from typing import Dict, List, Optional

from comdp_plus_no_deadline.engines.fact_pattern import FactPatternModel, grow_pattern
from comdp_plus_no_deadline.engines.survivor_sweep import relaxed_actions_from_engine
from comdp_plus_no_deadline.engines.temporal_stn_pdb import EngineModel, goal_relevance_closure
from comdp_plus_no_deadline.engines.windows_lao import WindowsLAO, WindowsTable


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


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
        self.aggregation = (os.environ.get("TP_MCTS_WILAO_AGGREGATION") or "min").strip().lower()
        self.extend_offline = bool(_env_int("TP_MCTS_WILAO_OFFLINE_EXTEND", 1))
        self.full_table = bool(_env_int("TP_MCTS_WILAO_FULL_TABLE", 1))
        self.pattern_facts = _env_int("TP_MCTS_WILAO_PATTERN_FACTS", 8)
        self.patterns: List[dict] = []
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
        if self._built:
            return
        self._built = True
        started = time.perf_counter()
        base = EngineModel(self._mdp)
        goals = sorted(self._mdp.problem.goals, key=str)
        size = len(goals) if self.pattern_goals <= 0 else self.pattern_goals
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

        for group in groups:
            if self.pattern_facts > 0:
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
            })
        # Phase 1: the full table of every pattern that fits: every state, every r.
        if self.full_table:
            for index, pattern in enumerate(self.patterns):
                table = WindowsTable(pattern["solver"], deadline)
                if table.build(pattern["root_facts"], time_budget=share(len(self.patterns) - index, table_end)):
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
        tables = [p["table_stats"] or {} for p in self.patterns]
        self.offline_report = {
            "patterns": len(self.patterns),
            "pattern_facts": [len(p["facts"]) if p["facts"] is not None else "all" for p in self.patterns],
            "pattern_actions": [p["actions"] for p in self.patterns],
            "seconds": round(time.perf_counter() - started, 2),
            "budget": "none" if unlimited else self.offline_seconds,
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

    # -- online --------------------------------------------------------------
    def heuristic_score(self, state, goal_facts=(), fixed_depth: int = 25, start_time: float = 0.0,
                        running_remaining: Optional[Dict[str, float]] = None, **_ignored) -> float:
        self.build()
        started = time.perf_counter()
        self.counts["queries"] += 1
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
        if self.aggregation == "product":
            out = 1.0
            for v in values:
                out *= v
            return out
        return min(values)

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
