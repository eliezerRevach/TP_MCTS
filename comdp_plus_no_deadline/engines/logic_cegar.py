"""Logic CEGAR: the pattern's facts by counterexamples, without a timed table per round
(``TP_MCTS_WILAO_PATTERN_GROWTH = logic``).

Same loop as ``cegar_pattern.py`` (Rovner, Sievers & Helmert 2019; for probabilistic tasks
Kloessner et al. 2022), with three changes:

    plan     the pattern's FASTEST EXPECTED ROUTE, untimed: a stochastic shortest path over
             (pattern facts, running pattern actions) where an end costs the action's duration,
             an instant action 0.01 and a start 0. No windows, no deadline: a round is cheap.
    sources  the initial state AND real states visited by rollouts (TP_MCTS_WILAO_LOGIC_SOURCES):
               random  uniformly random legal actions, drawn once
               greedy  every round, the action with the smallest sum over the current patterns of
                       the expected time to their goals (the same untimed values): the visited
                       states follow the patterns' own plans
             A fact like free_h(h0) is a flaw only where the hand is already taken; from the
             initial state alone every hand is free, so the repo's ``cegar`` never adds it.
    flaws    every failing fact of every replay; added up to the cap (all of them in one round
             when they fit), ordered by TP_MCTS_WILAO_LOGIC_RANK:
               frequency  the facts that fail most often first
               restrict   the facts the most actions NEED first (a hand before a store): they cut
                          how many actions can run at once, so the table stays small

    phi_g = goals of group g
    repeat:
        untimed model of each phi_g, its fastest-route values V_g
        sources = initial state + rollout states (greedy: guided by sum_g V_g)
        replay each pattern's route from each source in the REAL model: a real precondition
          (start or end, positive or negative) that fails is a flaw -> its fact
        add the flaw facts (most frequent first) up to max_facts
    until no pattern gains a fact

The rollouts save and restore the global random state, so a run's own episodes are unchanged.
"""

from __future__ import annotations

import collections
import math
import random
from typing import Dict, List, Optional, Sequence

from comdp_plus_no_deadline.engines.fact_pattern import FactPatternModel, _is_exec, op_reads, op_touches

_BIG = 1e6
_INSTANT_COST = 0.01


def _changeable(ops) -> set:
    out = set()
    for op in ops.values():
        out |= op_touches(op)
    return out


def running_of(facts, ops) -> frozenset:
    """Pattern actions running in a real fact set (from the inExecution markers)."""
    texts = [str(f) for f in facts if _is_exec(f)]
    out = []
    for key in ops:
        base = key[len("start_"):] if key.startswith("start_") else key
        if any(f"start-{base}" in t for t in texts):
            out.append(key)
    return frozenset(out)


def local_state(model, facts):
    return model.project(facts), running_of(facts, model.ops())


def solve_untimed(model, roots, cap: int = 300_000):
    """Fastest expected route: SSP values over the untimed pattern states reachable from roots.
    States that cannot reach the goal get inf; value iteration starts from 0 (an inf start
    never moves through retry loops)."""
    ops = model.ops()
    succ: Dict[tuple, list] = {}
    stack, seen = list(roots), set(roots)
    while stack and len(seen) < cap:
        s = stack.pop()
        facts, running = s
        opts = []
        if not (model.goals <= facts):
            for k in model.legal_action_names(facts):
                if k in running:
                    continue
                op = ops[k]
                cost = _INSTANT_COST if op.end_action is None else 0.0
                branches = []
                for f2, p in model.outcomes(op.start_action, facts):
                    if p > 1e-12:
                        branches.append((p, (frozenset(f2), running if op.end_action is None else running | {k})))
                if branches:
                    opts.append((("start", k), cost, branches))
            for k in running:
                op = ops[k]
                if not model.end_legal_op(op, facts):
                    continue
                branches = [(p, (frozenset(f2), running - {k})) for f2, p in model.outcomes(op.end_action, facts)
                            if p > 1e-12]
                if branches:
                    opts.append((("end", k), float(op.duration), branches))
        succ[s] = opts
        for _l, _c, br in opts:
            for _p, s2 in br:
                if s2 not in seen:
                    seen.add(s2)
                    stack.append(s2)
    preds = collections.defaultdict(set)
    for s, opts in succ.items():
        for _l, _c, br in opts:
            for _p, s2 in br:
                preds[s2].add(s)
    alive = {s for s in seen if model.goals <= s[0]}
    stack = list(alive)
    while stack:
        s2 = stack.pop()
        for s in preds[s2]:
            if s not in alive:
                alive.add(s)
                stack.append(s)
    V = {s: (0.0 if s in alive else _BIG) for s in seen}
    for _it in range(5000):
        delta = 0.0
        for s, opts in succ.items():
            if model.goals <= s[0] or s not in alive:
                continue
            best = min((c + sum(p * V.get(s2, _BIG) for p, s2 in br) for _l, c, br in opts), default=_BIG)
            best = min(best, _BIG)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < 1e-9:
            break
    return {s: (math.inf if v >= _BIG else v) for s, v in V.items()}, succ


def replay(model, V, succ, base, real_facts, changeable, max_steps: int = 60) -> set:
    """Follow the pattern's fastest route in the REAL model from a real state; the facts of
    the first real precondition that fails (statics excepted: the model has their value)."""
    ops = model.ops()
    real_ops = base.ops()
    facts = frozenset(real_facts)
    for _ in range(max_steps):
        local = local_state(model, facts)
        if model.goals <= local[0] or V.get(local, math.inf) == math.inf:
            return set()
        opts = succ.get(local)
        if not opts:
            return set()
        label, _c, _br = min(opts, key=lambda o: o[1] + sum(p * V.get(s2, math.inf) for p, s2 in o[2]))
        kind, k = label
        op = real_ops[k]
        ev = op.end_action if kind == "end" else op.start_action
        broken = {f for f in ev.pos_preconditions if f not in facts and not _is_exec(f) and f in changeable}
        broken |= {f for f in ev.neg_preconditions if f in facts and not _is_exec(f) and f in changeable}
        if broken:
            return broken
        outs = base.outcomes(ev, facts)
        if not outs:
            return set()
        facts = frozenset(max(outs, key=lambda o: (-V.get(local_state(model, o[0]), math.inf), o[1]))[0])
        if k not in ops:
            return set()
    return set()


def rollout_states(mdp, n: int, depth: int, rng: random.Random, choose=None) -> set:
    """Real fact sets visited by n walks of at most `depth` steps from the initial state;
    `choose(state, legal)` picks the action (None = uniformly random). The global random state
    (used by mdp.step to sample outcomes) is saved and restored."""
    import numpy as np

    saved = (random.getstate(), np.random.get_state())
    out = set()
    try:
        for i in range(n):
            random.seed(rng.randrange(1 << 30))
            np.random.seed(rng.randrange(1 << 30))
            state = mdp.initial_state()
            for _step in range(depth):
                out.add(frozenset(state.predicates))
                if mdp.is_terminal(state):
                    break
                legal = list(mdp.legal_actions(state))
                if not legal:
                    break
                a = rng.choice(legal) if choose is None else choose(state, legal)
                _term, state, _r = mdp.step(state, a)
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
    return out


def logic_cegar_patterns(mdp, base, groups: Sequence[Sequence], initial_facts, max_facts: int,
                         sources: str = "random", rollouts: int = 20, depth: int = 60, seed: int = 0,
                         max_rounds: int = 40, cap_states: int = 300_000,
                         rank: str = "frequency") -> List[Dict[str, object]]:
    """One fact list per goal group: ``[{"facts", "log", "stop"}]``."""
    if sources not in ("random", "greedy"):
        raise ValueError(f"logic CEGAR sources {sources!r}: use random | greedy")
    if rank not in ("frequency", "restrict"):
        raise ValueError(f"logic CEGAR rank {rank!r}: use frequency | restrict")
    needed_by = collections.Counter()
    for op in base.ops().values():
        for f in op_reads(op):
            needed_by[f] += 1
    rng = random.Random(seed)
    initial = frozenset(initial_facts)
    changeable = _changeable(base.ops())
    phis = [list(dict.fromkeys(g)) for g in groups]
    done = [None] * len(groups)
    logs: List[list] = [[] for _ in groups]
    fixed_sources = {initial} | (rollout_states(mdp, rollouts, depth, rng) if sources == "random" else set())
    prev_sources = {initial}
    for rnd in range(max_rounds):
        models = [FactPatternModel(base, phi, g, initial) for phi, g in zip(phis, groups)]
        if sources == "greedy":
            # values from the previous sources guide the rollouts
            guides = [solve_untimed(m, {local_state(m, f) for f in prev_sources}, cap_states)[0] for m in models]

            def choose(state, legal, models=models, guides=guides):
                def cost(a):
                    total = 0.0
                    for nxt, p in mdp.transition_function(state, a):
                        f = frozenset(nxt.predicates)
                        total += p * sum(min(V.get(local_state(m, f), _BIG), _BIG) for m, V in zip(models, guides))
                    return total
                scored = [(cost(a), rng.random(), a) for a in legal]
                return min(scored, key=lambda x: (x[0], x[1]))[2]

            src = {initial} | rollout_states(mdp, rollouts, depth, rng, choose)
            prev_sources = src
        else:
            src = fixed_sources
        added_any = False
        for gi, (model, phi) in enumerate(zip(models, phis)):
            if done[gi] is not None:
                continue
            roots = {local_state(model, f) for f in src}
            V, succ = solve_untimed(model, roots, cap_states)
            flaws = collections.Counter()
            for f in src:
                for x in replay(model, V, succ, base, f, changeable):
                    flaws[x] += 1
            if rank == "restrict":
                order = sorted(flaws.items(), key=lambda kv: (-needed_by[kv[0]], -kv[1], str(kv[0])))
            else:
                order = sorted(flaws.items(), key=lambda kv: (-kv[1], str(kv[0])))
            new = [x for x, _n in order if x not in phi]
            logs[gi].append({"round": rnd, "facts": len(phi), "untimed_states": len(succ), "sources": len(src),
                             "flaws": len(new)})
            if not new:
                done[gi] = "no flaw"
                continue
            room = max_facts - len(phi)
            if room <= 0:
                done[gi] = "cap"
                continue
            phi.extend(new[:room])
            logs[gi][-1]["added"] = [str(x) for x in new[:room]]
            added_any = True
        if not added_any:
            break
    return [{"facts": phi, "log": log, "stop": stop or "rounds"} for phi, log, stop in zip(phis, logs, done)]
