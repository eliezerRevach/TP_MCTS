"""TP_MCTS_WILAO_PATTERN_GROWTH = llm_by_hand_nasa_two / _one: hand-picked facts for nasa_rover obj 2 / obj 1."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

pytest.importorskip("unified_planning")

from comdp_plus_no_deadline.engines.hand_patterns import hand_patterns  # noqa: E402


def _nasa(obj):
    import unified_planning as up
    import unified_planning.domains  # noqa: F401
    from unified_planning.engines.convert_problem import Convert_problem
    from unified_planning.engines.mdp import MDP
    from comdp_plus_no_deadline.engines.temporal_stn_pdb import EngineModel
    from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic

    domain = up.domains.Nasa_Rover(kind="regular", deadline=25, object_amount=obj)
    grounded = up.engines.Grounder(domain.grounding_map())._compile(domain.problem).problem
    mdp = MDP(Convert_problem(grounded)._converted_problem, discount_factor=1.0, reward_mode="terminal",
              step_penalty=0.0)
    base = EngineModel(mdp)
    return mdp, WindowsILAOPDBHeuristic(mdp)._facts_by_name(base)


def test_nasa_two_resolves_every_fact_and_covers_every_goal():
    mdp, facts = _nasa(2)
    entries = hand_patterns("llm_by_hand_nasa_two", facts, mdp.problem.goals)
    labels = {label: sorted(str(f) for f in phi) for _g, phi, label in entries}
    assert sorted(labels) == ["hands_r0", "hands_r1", "image_o0", "image_o1", "stores_r0", "stores_r1"]
    assert len(labels["hands_r0"]) == 10 and len(labels["stores_r1"]) == 10 and len(labels["image_o0"]) == 3
    assert {"free_h(h0)", "free_h(h1)", "ready(h1, x0)", "ready(h0, x1)"} <= set(labels["hands_r0"])
    assert {"full(s2)", "full(s3)", "ready_to_drop(s3)"} <= set(labels["stores_r1"])
    covered = {str(g) for goals, _phi, _l in entries for g in goals}
    assert covered == {str(g) for g in mdp.problem.goals}


def test_nasa_two_refuses_another_problem():
    mdp, facts = _nasa(1)
    with pytest.raises(ValueError, match="does not fit this problem"):
        hand_patterns("llm_by_hand_nasa_two", facts, mdp.problem.goals)


def test_nasa_one_is_rover_r0_of_nasa_two():
    """llm_by_hand_nasa_one: the same picks for the single rover of nasa_rover obj 1."""
    mdp, facts = _nasa(1)
    entries = hand_patterns("llm_by_hand_nasa_one", facts, mdp.problem.goals)
    labels = sorted(label for _g, _phi, label in entries)
    assert labels == ["hands_r0", "image_o0", "stores_r0"]
    covered = {str(g) for goals, _phi, _l in entries for g in goals}
    assert covered == {str(g) for g in mdp.problem.goals}


@pytest.mark.parametrize("domain, obj, expected", [
    ("nasa_rover", 1, "llm_by_hand_nasa_one"),
    ("nasa_rover", 2, "llm_by_hand_nasa_two"),
    ("nasa_rover", 3, "llm_by_hand_nasa_three"),
    ("stuck_car", 2, "llm_by_hand_stuck_car_two"),
    ("machine_shop", 2, "llm_by_hand_machine_shop_two"),
    ("machine_shop", 3, "llm_by_hand_machine_shop_three"),
    ("machine_shop", 4, "llm_by_hand_machine_shop_four"),
])
def test_llm_by_hand_picks_the_collection_that_fits(domain, obj, expected):
    import unified_planning as up
    import unified_planning.domains  # noqa: F401
    from unified_planning.engines.convert_problem import Convert_problem
    from unified_planning.engines.mdp import MDP
    from comdp_plus_no_deadline.engines.hand_patterns import HAND_PATTERNS, pick_hand_collection
    from comdp_plus_no_deadline.engines.temporal_stn_pdb import EngineModel
    from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic

    cls = {"nasa_rover": up.domains.Nasa_Rover, "stuck_car": up.domains.Stuck_Car,
           "machine_shop": up.domains.Machine_Shop}[domain]
    model = cls(kind="regular", deadline=20, object_amount=obj)
    grounder = up.engines.Grounder(model.grounding_map()) if domain == "nasa_rover" else up.engines.Grounder()
    mdp = MDP(Convert_problem(grounder._compile(model.problem).problem)._converted_problem,
              discount_factor=1.0, reward_mode="terminal", step_penalty=0.0)
    facts = WindowsILAOPDBHeuristic(mdp)._facts_by_name(EngineModel(mdp))
    assert pick_hand_collection(facts, mdp.problem.goals) == expected
    entries = hand_patterns("llm_by_hand", facts, mdp.problem.goals)
    assert len(entries) == len(HAND_PATTERNS[expected])
    assert {str(g) for goals, _phi, _l in entries for g in goals} == {str(g) for g in mdp.problem.goals}
