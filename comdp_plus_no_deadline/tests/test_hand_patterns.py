"""TP_MCTS_WILAO_PATTERN_GROWTH = llm_by_hand_nasa_two: hand-picked facts for nasa_rover obj 2."""

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
