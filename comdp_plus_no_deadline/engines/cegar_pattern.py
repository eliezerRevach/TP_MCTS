"""Counterexample-guided pattern growth for the windows PDB (CEGAR).

As in Rovner, Sievers & Helmert (ICAPS 2019) and, for goal probability, Klössner,
Torralba, Steinmetz & Hoffmann (ICAPS 2021): start from the goal, solve the
abstraction, execute its policy in the real problem, and add whatever fact makes
the execution fail -- until nothing fails.

    phi = goals
    repeat:
        table   = the full windows table of the pattern phi (WindowsTable)
        replay  its best policy from the initial state in the REAL model, following
                every outcome the policy reaches (abstract r = deadline):
                  an option whose real precondition fails -- a positive one false or
                  a negative one true -- is a FLAW, weighted by the probability of
                  reaching it; the replay does not go past a flaw
        no flaw -> stop: the pattern's policy runs in the real problem
        flaw    -> add the fact with the most flaw probability to phi
    until |phi| = cap or the time budget is used

In the pattern, a fact outside phi counts as true in a positive precondition and is
ignored in a negative one, so every prefix of the loop is an upper bound (adding a
fact only removes abstract-only behaviour).

Replay details. The abstract and the real state share the end order and windows
(they depend only on the options taken), so only the facts differ. A real outcome
is followed to the abstract state it projects to; an instant action that lands back
on the same abstract state is a retry (folded in the table) and is skipped. A real
outcome the abstraction does not have (its effects read a fact outside phi) is
counted as ``unmatched`` and not followed.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

from comdp_plus_no_deadline.engines.fact_pattern import FactPatternModel, _is_exec
from comdp_plus_no_deadline.engines.windows_lao import WindowsLAO, WindowsTable

_EPS = 1e-12


def replay_flaws(table: WindowsTable, model: FactPatternModel, base, real_facts,
                 horizon, max_nodes: int = 50_000) -> Dict[str, object]:
    """Execute the table's policy in the real model; flaw facts with the
    probability of reaching them."""
    horizon = Fraction(horizon)
    ops = base.ops()
    flaws: Dict[object, float] = defaultdict(float)
    unmatched = 0.0
    root = (0, frozenset(real_facts), horizon)
    frontier = deque([(root, 1.0)])
    seen = set()
    nodes = 0
    while frontier and nodes < max_nodes:
        (i, facts, r), prob = frontier.popleft()
        if (i, facts, r) in seen:
            continue
        seen.add((i, facts, r))
        nodes += 1
        option = table.policy(i, r)
        if option is None:
            continue
        charge, _inert, branches, label = option
        op = ops[label[1]]
        event = op.end_action if label[0] == "end" else op.start_action
        broken = [f for f in event.pos_preconditions if f not in facts and not _is_exec(f)]
        broken += [f for f in event.neg_preconditions if f in facts and not _is_exec(f)]
        if broken:
            for f in broken:
                flaws[f] += prob
            continue
        if not branches:
            continue
        abstract = table.states[i][0]
        _f, queue, windows = table.states[branches[0][1]]
        for nxt, p in base.outcomes(event, facts):
            if p <= _EPS:
                continue
            projected = model.project(nxt)
            if label[0] == "do" and projected == abstract:
                continue                                  # zero-time retry, folded in the table
            j = table.index.get((projected, queue, windows))
            if j is None:
                unmatched += prob * p
                continue
            frontier.append(((j, frozenset(nxt), r - charge), prob * p))
    return {"flaws": dict(flaws), "unmatched": unmatched, "nodes": nodes, "cut": bool(frontier)}


def cegar_pattern(base, goals: Sequence, initial_facts, max_facts: int, deadline,
                  time_budget: Optional[float] = None) -> Dict[str, object]:
    """Grow a pattern for ``goals`` by counterexamples. Returns the last pattern
    whose table was built: ``facts``, ``model``, ``table`` (None if even the first
    table did not fit the budget) and a ``log`` of the refinements."""
    started = time.perf_counter()
    stop_at = None if time_budget is None else started + float(time_budget)
    real = frozenset(initial_facts)
    phi: List = list(dict.fromkeys(goals))
    best = {"facts": list(phi), "model": None, "table": None}
    log = []
    while True:
        model = FactPatternModel(base, phi, goals, real)
        table = WindowsTable(WindowsLAO(model, heuristic="none"), deadline)
        left = None if stop_at is None else max(0.0, stop_at - time.perf_counter())
        if not table.build(model.project(real), time_budget=left):
            log.append({"facts": len(phi), "stop": "budget"})
            break
        best = {"facts": list(phi), "model": model, "table": table}
        result = replay_flaws(table, model, base, real, deadline)
        entry = {"facts": len(phi), "states": len(table.states),
                 "value": round(table.lookup(model.project(real), deadline)[0], 6),
                 "replay_nodes": result["nodes"], "unmatched": round(result["unmatched"], 6)}
        flaws = {f: p for f, p in result["flaws"].items() if f not in phi}
        if not flaws:
            entry["stop"] = "no flaw"
            log.append(entry)
            break
        if len(phi) >= max_facts:
            entry["stop"] = "cap"
            entry["open_flaws"] = len(flaws)
            log.append(entry)
            break
        chosen = max(flaws, key=lambda f: (flaws[f], str(f)))
        entry["added"] = str(chosen)
        entry["flaw_probability"] = round(flaws[chosen], 6)
        log.append(entry)
        phi.append(chosen)
    best["log"] = log
    best["seconds"] = round(time.perf_counter() - started, 2)
    return best
