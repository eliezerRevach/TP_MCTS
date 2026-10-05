"""``fact_pattern.resource_patterns``: one pattern per (goal group, shared resource)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.fact_pattern import independent_goal_groups, resource_patterns  # noqa: E402


def _nasa(obj):
    up = pytest.importorskip("unified_planning")
    import unified_planning.domains  # noqa: F401
    from unified_planning.engines.convert_problem import Convert_problem
    from unified_planning.engines.mdp import MDP
    from comdp_plus_no_deadline.engines.temporal_stn_pdb import EngineModel

    domain = up.domains.Nasa_Rover(kind="regular", deadline=25, object_amount=obj)
    grounded = up.engines.Grounder(domain.grounding_map())._compile(domain.problem).problem
    mdp = MDP(Convert_problem(grounded)._converted_problem, discount_factor=1.0, reward_mode="terminal",
              step_penalty=0.0)
    base = EngineModel(mdp)
    goals = sorted(mdp.problem.goals, key=str)
    groups = [sorted(g, key=str) for g in independent_goal_groups(base.ops(), goals)]
    return resource_patterns(base.ops(), groups, mdp.initial_state().predicates)


def test_nasa_rover_one_pattern_per_hand_and_store():
    pats = {res: sorted(str(f) for f in facts) for _goals, facts, res in _nasa(1)}
    assert set(pats) == {"base", "h0", "h1", "s0", "s1"}
    assert pats["base"] == ["calibrated(c0, o0)", "communicated_image_data(o0)", "have_image(r0, o0)"]
    rocks = ["communicated_rock_data(x0)", "communicated_rock_data(x1)",
             "have_rock_analysis(r0, x0)", "have_rock_analysis(r0, x1)"]
    assert pats["h0"] == sorted(rocks + ["free_h(h0)", "ready(h0, x0)", "ready(h0, x1)"])
    assert pats["s1"] == sorted(rocks + ["free_s(s1)", "full(s1)", "ready_to_drop(s1)"])


def test_every_pattern_holds_all_goals_of_its_group():
    for goals, facts, _res in _nasa(2):
        assert set(goals) <= set(facts)
        assert len(facts) <= 7


def test_logic_rank_is_validated():
    from comdp_plus_no_deadline.engines.logic_cegar import logic_cegar_patterns
    with pytest.raises(ValueError):
        logic_cegar_patterns(None, None, [], set(), 1, rank="bogus")
