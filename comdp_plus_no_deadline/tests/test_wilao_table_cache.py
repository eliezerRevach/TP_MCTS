"""TP_MCTS_WILAO_TABLE_CACHE: build once, save, every other process loads its own copy.
A loaded heuristic must answer every lookup exactly like the one that built the tables."""

import os
import random
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _mdp():
    up = pytest.importorskip("unified_planning")
    import unified_planning.domains  # noqa: F401
    from unified_planning.engines.convert_problem import Convert_problem
    from unified_planning.engines.mdp import MDP

    domain = up.domains.Nasa_Rover(kind="regular", deadline=25, object_amount=1)
    grounded = up.engines.Grounder(domain.grounding_map())._compile(domain.problem).problem
    return MDP(Convert_problem(grounded)._converted_problem, discount_factor=1.0, reward_mode="terminal",
               step_penalty=0.0)


@pytest.fixture
def small_env(monkeypatch, tmp_path):
    for k, v in {"TP_MCTS_WILAO_REPORT": "0", "TP_MCTS_WILAO_PATTERN_GROWTH": "static",
                 "TP_MCTS_WILAO_PATTERN_FACTS": "3", "TP_MCTS_WILAO_OFFLINE_SECONDS": "0",
                 "TP_MCTS_WILAO_OFFLINE_EXTEND": "0", "TP_MCTS_WILAO_BLUR": "3", "TP_MCTS_WILAO_MISS": "one",
                 "TP_MCTS_WILAO_TABLE_CACHE": str(tmp_path)}.items():
        monkeypatch.setenv(k, v)
    return tmp_path


def _states(mdp, n=40, seed=0):
    rng = random.Random(seed)
    out, state = [], mdp.initial_state()
    for _ in range(n):
        out.append(state)
        legal = list(mdp.legal_actions(state))
        if not legal or mdp.is_terminal(state):
            state = mdp.initial_state()
            continue
        state = rng.choice(mdp.transition_function(state, rng.choice(legal)))[0]
    return out


def test_second_process_loads_and_answers_identically(small_env):
    from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic
    mdp = _mdp()
    built = WindowsILAOPDBHeuristic(mdp)
    built.build()
    assert built.offline_report["cache"]["hit"] is False and built.offline_report["cache"]["saved"] is True
    files = [f for f in os.listdir(small_env) if f.endswith(".pkl")]
    assert len(files) == 1 and not any(f.endswith(".lock") for f in os.listdir(small_env))
    loaded = WindowsILAOPDBHeuristic(mdp)
    loaded.build()
    assert loaded.offline_report["cache"]["hit"] is True
    assert loaded.groups == built.groups and len(loaded.patterns) == len(built.patterns)
    for state in _states(mdp):
        for r in (25, 18, 11, 4):
            assert loaded.heuristic_score(state, fixed_depth=r, running_remaining={}) == \
                built.heuristic_score(state, fixed_depth=r, running_remaining={})


def test_build_data_is_dropped_but_lookups_still_work(small_env):
    from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic
    mdp = _mdp()
    h = WindowsILAOPDBHeuristic(mdp)
    h.build()
    assert all(p["table"].options is None and p["table"].states is None for p in h.patterns)
    assert 0.0 < h.heuristic_score(mdp.initial_state(), fixed_depth=25, running_remaining={}) <= 1.0


def test_a_different_setting_uses_a_different_file(small_env, monkeypatch):
    from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic
    mdp = _mdp()
    WindowsILAOPDBHeuristic(mdp).build()
    monkeypatch.setenv("TP_MCTS_WILAO_BLUR", "5")
    h = WindowsILAOPDBHeuristic(mdp)
    h.build()
    assert h.offline_report["cache"]["hit"] is False
    assert len([f for f in os.listdir(small_env) if f.endswith(".pkl")]) == 2


def test_a_stale_lock_is_taken_over(small_env, monkeypatch):
    from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic
    mdp = _mdp()
    h = WindowsILAOPDBHeuristic(mdp)
    lock = os.path.join(str(small_env), f"wilao_{h._cache_key()}.pkl.lock")
    open(lock, "w").close()
    old = time.time() - 3600
    os.utime(lock, (old, old))
    monkeypatch.setenv("TP_MCTS_WILAO_CACHE_STALE", "60")
    h.build()
    assert h.offline_report["cache"]["hit"] is False and not os.path.exists(lock)


def test_cache_off_writes_nothing(small_env, monkeypatch):
    from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic
    monkeypatch.setenv("TP_MCTS_WILAO_TABLE_CACHE", "")
    h = WindowsILAOPDBHeuristic(_mdp())
    h.build()
    assert "cache" not in h.offline_report and os.listdir(small_env) == []
