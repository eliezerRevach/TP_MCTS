"""ILAO* on TP-MCTS's (state, STN) MDP (engines/lao_tp_mdp.py)."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

pytest.importorskip("unified_planning")


def _mdp(domain, deadline, object_amount=1, garbage_amount=0):
    import unified_planning as up
    import unified_planning.domains  # noqa: F401
    from unified_planning.engines.convert_problem import Convert_problem
    from unified_planning.engines.mdp import MDP

    cls = {"prob_conc": up.domains.Prob_Conc, "stuck_car": up.domains.Stuck_Car}[domain]
    model = cls(kind="regular", deadline=deadline, object_amount=object_amount, garbage_amount=garbage_amount)
    ground = up.engines.compilers.Grounder()._compile(model.problem).problem
    return MDP(Convert_problem(ground)._converted_problem, discount_factor=1.0, reward_mode="terminal",
               step_penalty=0.0)


@pytest.fixture
def pdb_env(monkeypatch):
    for k, v in {"TP_MCTS_WILAO_OFFLINE_SECONDS": "0", "TP_MCTS_WILAO_MISS": "one", "TP_MCTS_WILAO_AGG": "min",
                 "TP_MCTS_WILAO_GROUPING": "1", "TP_MCTS_WILAO_PATTERN_GROWTH": "pairs",
                 "TP_MCTS_WILAO_BLUR": "3", "TP_MCTS_WILAO_TABLE_CACHE": "", "TP_MCTS_WILAO_REPORT": "0"}.items():
        monkeypatch.setenv(k, v)


def test_prob_conc_value_is_the_known_optimum(pdb_env):
    """prob_conc: independent goals, V* = prod 1 - (1 - p)^floor(D / d) = 0.853784 at D = 10."""
    from comdp_plus_no_deadline.engines.lao_tp_mdp import TPMDPLAO, windows_pdb_heuristic

    mdp = _mdp("prob_conc", 10, garbage_amount=2)
    vstar = math.prod(1 - (1 - p) ** (10 // d) for d, p in ((8, 1.0), (4, 0.7), (2, 0.49), (1, 0.3)))
    lao = TPMDPLAO(mdp, heuristic=windows_pdb_heuristic(mdp))
    assert lao.solve() == pytest.approx(vstar, abs=1e-6)
    assert lao.stats["complete"] and lao.stats["cycles"] == 0


def test_pdb_and_blind_search_agree_on_stuck_car(pdb_env):
    """The PDB must never cut off the optimum: same V(s0) as h = 1, with fewer expansions."""
    from comdp_plus_no_deadline.engines.lao_tp_mdp import TPMDPLAO, windows_pdb_heuristic

    mdp = _mdp("stuck_car", 10, object_amount=1)
    guided = TPMDPLAO(mdp, heuristic=windows_pdb_heuristic(mdp))
    blind = TPMDPLAO(mdp, heuristic=None)
    v_guided, v_blind = guided.solve(), blind.solve()
    assert v_guided == pytest.approx(v_blind, abs=1e-9)
    assert guided.stats["expanded"] < blind.stats["expanded"]
    assert guided.root.value <= 0.946668 + 1e-6            # below h(s0): the PDB is an upper bound


def test_exact_stn_accepts_a_consistent_retry():
    """After start_one runs twice, TP-MCTS's incremental check confuses the two end_one events
    (STNPlanNode.__eq__) and calls this path inconsistent; Floyd-Warshall finds no negative cycle."""
    from comdp_plus_no_deadline.engines.lao_tp_mdp import exact_stn
    from unified_planning.engines.utils import create_init_stn, update_stn

    mdp = _mdp("prob_conc", 10, garbage_amount=0)
    act = {a.name: a for a in mdp.problem.actions}
    path = ["start_four", "start_eight", "start_one", "end_four", "start_two", "end_one", "start_one",
            "end_two", "end_eight"]
    library, exact = create_init_stn(mdp), exact_stn(create_init_stn(mdp))
    lp = ep = None
    for name in path:
        lp = update_stn(library, act[name], lp)
        ep = update_stn(exact, act[name], ep)
    assert not library.is_consistent()            # the known false rejection, left as is in TP-MCTS
    assert exact.is_consistent()


def test_policy_executes_as_planned(pdb_env):
    """Executed in CoMDP+ with every time fixed when it happens (TP-MCTS's SetTime), the policy
    reaches the goal with exactly the probability ILAO* promised, and nothing it assumed fails."""
    from comdp_plus_no_deadline.engines.lao_tp_mdp import PolicyExecution, TPMDPLAO, windows_pdb_heuristic

    mdp = _mdp("stuck_car", 10, object_amount=1)
    lao = TPMDPLAO(mdp, heuristic=windows_pdb_heuristic(mdp))
    v = lao.solve()
    result = PolicyExecution(lao).exact()
    assert result["complete"] and result["violation_mass"] == {}
    assert result["executed_value"] == pytest.approx(v, abs=1e-6)
    sampled = PolicyExecution(lao).run(300, seed=1)
    assert sampled["violations"] == 0
    assert abs(sampled["success_rate"] - v) < 4 * max(sampled["ci95"], 0.01)


def test_different_histories_with_the_same_future_merge():
    """'one' failing twice and 'two' failing once both end at t = 2 with nothing gained and nothing
    running: the same facts and the same constraints on the future, so ONE node (TP-MCTS keeps
    two). The order of starts is not merged: in the STN it is a constraint (a later start may be
    pushed back), so start_one;start_two and start_two;start_one stay apart."""
    from comdp_plus_no_deadline.engines.lao_tp_mdp import TPMDPLAO

    mdp = _mdp("prob_conc", 10, garbage_amount=0)
    lao = TPMDPLAO(mdp, heuristic=None)

    def step(node, action, keep_failure=True):
        if node.options is None:
            lao._expand(node)
        branches = dict(node.options)[action]
        if not keep_failure:
            return [lao.nodes[k] for _p, k in branches]
        # the failure outcome: the branch that gained no new goal fact
        return min((lao.nodes[k] for _p, k in branches), key=lambda n: len(n.key[0]))

    one = lao.root
    for _ in range(2):
        one = step(step(one, "start_one"), "end_one")
    two = step(step(lao.root, "start_two"), "end_two")
    assert one.current_time == two.current_time == 2
    assert one is two

    ab = step(step(lao.root, "start_one"), "start_two")
    ba = step(step(lao.root, "start_two"), "start_one")
    assert ab.key[0] == ba.key[0] and ab is not ba


def test_diagnostics_on_a_finished_run(pdb_env):
    """Counters add up, the PDB never under-estimates V* on the optimal policy (admissible), and the
    lower bound meets V(s0) once the run is optimal."""
    from comdp_plus_no_deadline.engines.lao_tp_mdp import TPMDPLAO, windows_pdb_heuristic

    mdp = _mdp("stuck_car", 10, object_amount=1)
    lao = TPMDPLAO(mdp, heuristic=windows_pdb_heuristic(mdp), diagnostics=True)
    v = lao.solve()
    report = lao.diagnostics()
    assert sum(r["expanded"] for r in report["by_depth"]) == lao.stats["expanded"]
    assert report["tightness"]["violations"] == 0
    assert report["tightness"]["gap_mean"] >= 0
    assert report["progress"][-1]["lower_bound"] == pytest.approx(v, abs=1e-6)
    assert 0 <= report["h_pruned_options_pct"] <= 100


def test_time_budget_stops_inside_a_pass():
    """The limit is checked at every expansion: a run cut short stays an upper bound above a lower bound."""
    from comdp_plus_no_deadline.engines.lao_tp_mdp import TPMDPLAO

    mdp = _mdp("prob_conc", 10, garbage_amount=2)
    lao = TPMDPLAO(mdp, heuristic=None, time_budget=1e-6, diagnostics=True)
    upper = lao.solve()
    assert not lao.stats["complete"]
    assert lao.policy_lower_bound() <= upper + 1e-12


def test_canonical_starts_keep_the_optimum(pdb_env):
    """Independent starts taken only in name order: same V* on these domains, fewer nodes, and the
    policy still executes as planned (the executor rebuilds the same node keys)."""
    from comdp_plus_no_deadline.engines.lao_tp_mdp import PolicyExecution, TPMDPLAO, windows_pdb_heuristic

    for domain, kwargs in (("prob_conc", dict(garbage_amount=2)), ("stuck_car", dict(object_amount=1))):
        mdp = _mdp(domain, 10, **kwargs)
        h = windows_pdb_heuristic(mdp)
        full = TPMDPLAO(mdp, heuristic=h)
        canon = TPMDPLAO(mdp, heuristic=h, canonical_starts=True)
        assert canon.solve() == pytest.approx(full.solve(), abs=1e-9)
        assert canon.stats["canonical_skipped"] > 0
        assert canon.stats["generated"] <= full.stats["generated"]
        assert PolicyExecution(canon).exact()["executed_value"] == pytest.approx(canon.root.value, abs=1e-6)


def test_dominance_is_sound_on_exact_values():
    """Same facts and a looser STN (every bound >=) can never be worth less: checked on the whole MDP
    with exact values."""
    from comdp_plus_no_deadline.engines.lao_tp_mdp import TPMDPLAO

    mdp = _mdp("stuck_car", 10, object_amount=1)
    lao = TPMDPLAO(mdp, heuristic=None)
    lao.expand_all()
    for _ in range(100):
        change = 0.0
        for n in lao.nodes.values():
            if n.options and not n.goal and not n.dead:
                old = n.value
                lao._backup(n)
                change = max(change, abs(old - n.value))
        if change < 1e-13:
            break
    pairs = list(lao.dominance_pairs())
    assert pairs
    assert all(a.value >= b.value - 1e-9 for a, b in pairs)


def test_bounded_search_matches_ilao(pdb_env):
    """algorithm="bounded" (upper + lower bounds, FRTDP-style trials) with gap 0 returns the same V* as
    ILAO*, both bounds meet, and its policy executes at exactly that value."""
    from comdp_plus_no_deadline.engines.lao_tp_mdp import PolicyExecution, TPMDPLAO, windows_pdb_heuristic

    for domain, kwargs in (("stuck_car", dict(object_amount=1)), ("prob_conc", dict(garbage_amount=2))):
        mdp = _mdp(domain, 10, **kwargs)
        h = windows_pdb_heuristic(mdp)
        exact = TPMDPLAO(mdp, heuristic=h).solve()
        bounded = TPMDPLAO(mdp, heuristic=h, algorithm="bounded")
        v = bounded.solve()
        assert v == pytest.approx(exact, abs=1e-9)
        assert bounded.stats["complete"] and bounded.root.value - bounded.root.lower <= 1e-9
        assert PolicyExecution(bounded).exact()["executed_value"] == pytest.approx(v, abs=1e-6)
