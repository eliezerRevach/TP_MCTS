"""Standalone probe: inspect the PTRPG (baseline_survival) layer-by-layer
propagation in isolation from MCTS / the prefix rollout.

Goal: test the hypothesis that the cross-layer bias originates in the
propagation itself -- P_t(goal) is a layer-cumulative quantity, so the
final-layer "product" score depends on the horizon, and a layer-average
("area") lands at a different operating point.

For a handful of real nasa_rover states reached by greedy stepping, we dump:
  - the per-layer goal-conjunction product  G_t = prod_g P_t(g)
  - the FINAL-layer value  G_T            (what aggregation="product" returns)
  - the layer-MEAN value   mean_t G_t     (what aggregation="area" returns)
at several horizons H, so we can see how each moves with the horizon and with
how far the state has progressed (elapsed).

Run:
  PYTHONUTF8=1 PYTHONPATH=$PWD python scripts/probe_propagation.py
"""

import sys
import os

# parser.py calls parse_known_args() at import; give it a valid run_domain-style
# argv so construction matches a normal run (extra args are ignored).
sys.argv = [
    "probe_propagation.py",
    "--domain", "nasa_rover",
    "--object_amount", "3",
    "--deadline", "35",
    "--solver", "mcts",
    "--heuristic_name", "temporal_probabilistic_rpg",
    "--temporal_heuristic_strategy", "baseline_survival",
]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unified_planning as up
from unified_planning.shortcuts import *  # noqa: F401,F403  (Convert_problem, MDP, ...)
import unified_planning.domains  # noqa: F401
from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)

DOMAIN = "nasa_rover"
DEADLINE = 35
OBJECT_AMOUNT = 3


def build_mdp():
    model = up.domains.Nasa_Rover(
        kind="regular", deadline=DEADLINE, object_amount=OBJECT_AMOUNT, garbage_amount=0
    )
    grounder = up.engines.compilers.Grounder(model.grounding_map())
    ground_problem = grounder._compile(model.problem).problem
    converted = Convert_problem(ground_problem)._converted_problem  # noqa: F405
    mdp = MDP(converted, discount_factor=0.95, reward_mode="terminal", step_penalty=-0.05)  # noqa: F405
    return mdp


def goal_product_by_layer(result, goals):
    """G_t = prod_g P_t(g) for every layer t."""
    out = {}
    for t, probs in result.probabilities_by_layer.items():
        p = 1.0
        for g in goals:
            p *= max(0.0, min(1.0, probs.get(g, 0.0)))
        out[t] = p
    return out


def state_elapsed(state):
    return float(getattr(state, "current_time", 0.0) or 0.0)


def advance_to_elapsed(mdp, state, target_elapsed, max_steps=400):
    """Step (random legal) until current_time advances to >= target_elapsed
    (committing end actions advances the STN clock). Returns the new state."""
    import random
    s = state
    for _ in range(max_steps):
        if state_elapsed(s) >= target_elapsed:
            break
        legal = mdp.legal_actions(s)
        if not legal:
            break
        _terminal, s2, _r = mdp.step(s, random.choice(list(legal)))
        s = s2
    return s


def rise_time(G, g_inf, theta=0.5):
    """t* = first layer where G_t crosses theta*g_inf (horizon-invariant signal)."""
    if g_inf <= 0:
        return None
    target = theta * g_inf
    for t in sorted(G.keys()):
        if G[t] >= target:
            return t
    return None


def main():
    mdp = build_mdp()
    goals = set(mdp.problem.goals)
    deadline = mdp.deadline()
    heur = TemporalProbabilisticRPGHeuristic.from_problem(mdp.problem)
    print(f"domain={DOMAIN} object_amount={OBJECT_AMOUNT} deadline={deadline} "
          f"#goals={len(goals)}")

    import random
    random.seed(7)
    s0 = mdp.initial_state()
    # States at DIFFERENT elapsed (remaining deadline) so we can test cross-
    # deadline comparability, not just cross-horizon for one state.
    states = {
        "initial (elapsed~0)": s0,
        "advanced to elapsed~5": advance_to_elapsed(mdp, s0, 5),
        "advanced to elapsed~10": advance_to_elapsed(mdp, s0, 10),
        "advanced to elapsed~15": advance_to_elapsed(mdp, s0, 15),
    }

    horizons = [10, 20, 31, 35]

    print("\n### Part 1: per-state, per-horizon  product (final) vs area (mean) vs "
          "saturation-normalized\n")
    for label, st in states.items():
        el = state_elapsed(st)
        print(f"================ state: {label}  elapsed={el} ================")
        for H in horizons:
            rem = max(0, int(deadline - el)) if deadline is not None else H
            eff = min(H, rem)
            _score, result = heur.heuristic_score(
                st, goals, aggregation="product", fixed_depth=eff,
                start_time=el, strategy="baseline_survival", debug=True,
            )
            G = goal_product_by_layer(result, goals)
            layers = sorted(G.keys())
            if not layers:
                print(f"  H={H:<3} (eff={eff}) -> no layers")
                continue
            final = G[layers[-1]]
            mean = sum(G[t] for t in layers) / len(layers)
            g_inf = max(G.values())  # plateau estimate
            area_norm = (mean / g_inf) if g_inf > 0 else 0.0  # ~ (1 - t*/H)
            tstar = rise_time(G, g_inf)
            print(f"  H={H:<3}(eff={eff:<3}) product={final:.4f}  area={mean:.4f}  "
                  f"g_inf={g_inf:.4f}  area/g_inf={area_norm:.4f}  "
                  f"t*={tstar}  H*(1-area/g_inf)={eff*(1-area_norm):.1f}")
        print()

    print("\n### Part 2: PROPOSED deadline-normalized value V = 1 - t*/remaining\n")
    print("Run each state's DP at its OWN remaining deadline; extract horizon-")
    print("invariant t* and normalize by that state's remaining budget. Compare")
    print("against the saturated product (which cannot order these states).\n")
    rows = []
    for label, st in states.items():
        el = state_elapsed(st)
        rem = max(0, int(deadline - el)) if deadline is not None else 35
        _score, result = heur.heuristic_score(
            st, goals, aggregation="product", fixed_depth=rem,
            start_time=el, strategy="baseline_survival", debug=True,
        )
        G = goal_product_by_layer(result, goals)
        g_inf = max(G.values()) if G else 0.0
        tstar = rise_time(G, g_inf)
        product = G[max(G.keys())] if G else 0.0
        if tstar is None:
            V = 0.0
        else:
            V = max(0.0, 1.0 - tstar / rem) if rem > 0 else 0.0
        rows.append((label, el, rem, product, g_inf, tstar, V))
        print(f"  {label:<26} elapsed={el:<5} remaining={rem:<4} "
              f"product={product:.4f}  t*={tstar}  V=1-t*/rem={V:.4f}")
    print("\n  -> product is ~flat (can't rank); V orders states by deadline-")
    print("     normalized time-to-goal. Lower t* relative to remaining = better.")

    print("\n### Part 3: REAL aggregation modes on strategy='baseline' (initial state)\n")
    print("Calls heuristic_score(..., aggregation=...) directly so we exercise the")
    print("new 'time_to_goal' mode end-to-end and compare it to 'product'.\n")
    el = 0.0
    for H in [10, 15, 20, 25, 31, 35]:
        rem = max(0, int(deadline - el)) if deadline is not None else H
        eff = min(H, rem)
        prod_score = heur.heuristic_score(
            s0, goals, aggregation="product", fixed_depth=eff,
            start_time=el, strategy="baseline",
        )
        ttg_score = heur.heuristic_score(
            s0, goals, aggregation="time_to_goal", fixed_depth=eff,
            start_time=el, strategy="baseline",
        )
        print(f"  R={eff:<3} product={prod_score:.4f}   time_to_goal(V)={ttg_score:.4f}")
    print("\n  Expect: product saturates/flat across R; V increases with R (more")
    print("  budget for the same intrinsic t* = larger feasibility margin).")


if __name__ == "__main__":
    main()
