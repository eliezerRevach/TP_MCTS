"""``windows_ilao_pdb``: MCTS leaf heuristic = a table from ILAO* on the time-left
windows MDP, guided by the survivor sweep (the RPG PDB of rpg_exact_states_v18).

Spec: ``artifacts/Windows_ILAO_PDB.docx``.

    offline : one pattern per goal group; ILAO* from the initial state with
              r = deadline until the best policy is solved (optimal). Its states
              are marked solved. Then, until TP_MCTS_WILAO_OFFLINE_SECONDS is used
              up, the branches that policy did not take are solved one by one,
              closest to the root first (higher value first), each by ILAO* that
              stops at solved states -- so MCTS finds more of its leaves solved.
    online  : per MCTS leaf, per pattern, a LOOKUP:
                exact  the leaf's state is solved
                cover  a solved state with the same facts and end order, looser
                       windows and at least as much time: its value is an upper bound
                miss   TP_MCTS_WILAO_MISS = one  -> 1.0 (optimistic: MCTS explores it)
                                           lazy -> ILAO* from the leaf that stops at
                                                   solved states and uses their value,
                                                   within the query budget
    value   : min over patterns (sound); product only for comparison

A pattern is a group of goal facts plus every action that can achieve or threaten
them (``goal_relevance_closure``). The relaxed action table of the sweep is built
once per pattern, at the initial state.

Knobs (environment, set from experiments.ipynb):
    TP_MCTS_WILAO_OFFLINE_SECONDS   total offline budget                    (30)
    TP_MCTS_WILAO_OFFLINE_EXTEND    after optimal, solve other branches     (1)
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
from comdp_plus_no_deadline.engines.windows_lao import WindowsLAO


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
        offline_end = started + self.offline_seconds
        # Phase 1: the optimal policy of every pattern. Each pattern gets an equal
        # share of the time still left, so a quick pattern leaves its unused time
        # to the slower ones after it.
        for index, group in enumerate(groups):
            per_pattern = max(0.0, offline_end - time.perf_counter()) / (len(groups) - index)
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
            value = solver.solve(root_facts, deadline, time_budget=per_pattern)
            self.patterns.append({
                "goals": [str(g) for g in group],
                "facts": None if phi is None else [str(f) for f in phi],
                "actions": len(model.ops()),
                "model": model,
                "solver": solver,
                "offline_value": value,
                "offline_optimal": solver.complete,
                "offline_seconds": solver.stats["seconds"],
                "solved_at_optimal": len(solver.solved),
            })
        # Phase 2: spend what is left solving the branches the optimal policies did
        # not take, closest to the root first, round-robin over the patterns.
        if self.extend_offline:
            active = [p for p in self.patterns if p["solver"]._candidates]
            while active and time.perf_counter() < offline_end:
                slice_seconds = (offline_end - time.perf_counter()) / len(active)
                active = [p for p in active if p["solver"].extend(slice_seconds)]
        self.offline_report = {
            "patterns": len(self.patterns),
            "pattern_facts": [len(p["facts"]) if p["facts"] is not None else "all" for p in self.patterns],
            "pattern_actions": [p["actions"] for p in self.patterns],
            "seconds": round(time.perf_counter() - started, 2),
            "values": [round(p["offline_value"], 6) for p in self.patterns],
            "optimal": [p["offline_optimal"] for p in self.patterns],
            "seconds_to_optimal": [p["offline_seconds"] for p in self.patterns],
            "solved_at_optimal": [p["solved_at_optimal"] for p in self.patterns],
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
        lookups = sum(self.counts[k] for k in ("exact", "cover", "miss_one", "lazy_optimal", "lazy_cut"))
        return {
            "offline": self.offline_report,
            "queries": self.counts["queries"],
            "pattern_lookups": lookups,
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
