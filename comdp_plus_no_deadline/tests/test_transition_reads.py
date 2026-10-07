"""Facts an outcome probability reads (fact_pattern.transition_reads) and the per-value options
for the ones outside a pattern (FactPatternModel.outcome_variants): Q = max over their values."""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

pytest.importorskip("unified_planning")

from comdp_plus_no_deadline.engines import fact_pattern as fp  # noqa: E402


# -- toy: the event's outcome depends on f1, f2, f3 (none is a precondition) --------------
#   f1 and f2 and f3 -> s3      f1 and f2 -> s1      f2 only -> s2      else -> nothing

def _choose(state, _params):
    p = state.predicates
    if "f1" in p and "f2" in p:
        return {1.0: {"s3": True}} if "f3" in p else {1.0: {"s1": True}}
    if "f2" in p:
        return {1.0: {"s2": True}}
    return {1.0: {}}


def _event(name, pre=(), adds=(), pes=()):
    return SimpleNamespace(name=name, pos_preconditions=set(pre), neg_preconditions=set(), add_effects=set(adds),
                           del_effects=set(), probabilistic_effects=list(pes))


class _ToyBase:
    def __init__(self):
        pe = SimpleNamespace(probability_function=_choose, fluents=["s1", "s2", "s3"])
        self.event = _event("e", pes=[pe])
        self._ops = {
            "e": SimpleNamespace(key="e", start_action=self.event, end_action=None),
            "set": SimpleNamespace(key="set", start_action=_event("set", adds=["f1", "f3"]), end_action=None),
        }

    def ops(self):
        return self._ops

    def outcomes(self, action, full):
        out = []
        for p, eff in action.probabilistic_effects[0].probability_function(SimpleNamespace(predicates=set(full)),
                                                                           None).items():
            out.append((frozenset(set(full) | {f for f, v in eff.items() if v}), p))
        return out


def _results(model, facts):
    return sorted(sorted(set(nf) & {"s1", "s2", "s3"}) for v in model.outcome_variants(
        model.base.event, frozenset(facts)) for nf, _p in v)


def test_reads_follow_every_branch():
    # the first run (nothing true) only checks f1 and f2; f3 is read only when both hold
    assert fp.transition_reads(_ToyBase().event) == frozenset({"f1", "f2", "f3"})


def test_pattern_with_f2_only_can_choose_every_outcome_f2_allows():
    base = _ToyBase()
    model = fp.FactPatternModel(base, ["f2", "s1", "s2", "s3"], ["s3"], [], exact_statics=True)
    assert model.hidden_reads(base.event) == ("f1", "f3")
    assert _results(model, {"f2"}) == [["s1"], ["s2"], ["s3"]]     # f2 true: f1, f3 free -> s1 | s2 | s3
    assert _results(model, set()) == [[]]                            # f2 false: nothing, whatever f1, f3


def test_a_fact_in_the_pattern_filters_the_choice():
    base = _ToyBase()
    model = fp.FactPatternModel(base, ["f1", "f2", "s1", "s2", "s3"], ["s3"], [], exact_statics=True)
    assert model.hidden_reads(base.event) == ("f3",)
    assert _results(model, {"f1", "f2"}) == [["s1"], ["s3"]]         # f1 held true: s1 or s3
    assert _results(model, {"f2"}) == [["s2"]]                       # f1 held false: only s2


def test_knob_off_gives_the_old_single_distribution(monkeypatch):
    monkeypatch.setenv("TP_MCTS_WILAO_TRANSITION_READS", "0")
    base = _ToyBase()
    model = fp.FactPatternModel(base, ["f2", "s1", "s2", "s3"], ["s3"], [], exact_statics=True)
    assert model.hidden_reads(base.event) == ()
    assert _results(model, {"f2"}) == [["s2"]]                       # f1, f3 at their initial value (false)


# -- real domains -----------------------------------------------------------------------

def _mdp(domain, obj):
    import unified_planning as up
    import unified_planning.domains  # noqa: F401
    from unified_planning.engines.convert_problem import Convert_problem
    from unified_planning.engines.mdp import MDP

    if domain == "stuck_car":
        model = up.domains.Stuck_Car(kind="regular", deadline=15, object_amount=obj)
        grounder = up.engines.Grounder()
    else:
        model = up.domains.Nasa_Rover(kind="regular", deadline=25, object_amount=obj)
        grounder = up.engines.Grounder(model.grounding_map())
    grounded = grounder._compile(model.problem).problem
    return MDP(Convert_problem(grounded)._converted_problem, discount_factor=1.0, reward_mode="terminal",
               step_penalty=0.0)


def test_stuck_car_every_pattern_sees_the_good_rock():
    """push succeeds 0.9 with a good rock under the car, 0.2 without; the rock is no precondition.
    Before: 8 of 10 pairs patterns read it at its initial value (no rock) -> h(s) = 0.29 < V*(s) >= 0.81."""
    from comdp_plus_no_deadline.engines.temporal_stn_pdb import EngineModel

    mdp = _mdp("stuck_car", 2)
    base = EngineModel(mdp)
    ops = base.ops()
    goals = sorted(mdp.problem.goals, key=str)
    init = frozenset(mdp.initial_state().predicates)
    car = next(g for g in goals if "c0" in str(g))
    good = next(f for f in mdp.problem.initial_values if str(f) == "rock_under_car(c0, good)")
    push = ops["start_push_car_gas_r0_c0"].end_action
    assert {str(f) for f in fp.transition_reads(push)} == {"car_out(c0)", "rock_under_car(c0, bad)",
                                                          "rock_under_car(c0, good)"}
    state = init | set(push.pos_preconditions) | {good}
    real = sum(p for nf, p in base.outcomes(push, frozenset(state)) if car in nf)
    groups = [sorted(g, key=str) for g in fp.independent_goal_groups(ops, goals)]
    patterns = fp.resource_patterns(ops, groups, init)
    assert patterns
    for g, facts, _res in patterns:
        model = fp.FactPatternModel(base, list(facts), g, init)
        best = max(sum(p for nf, p in v if car in nf)
                   for v in model.outcome_variants(push, model.project(frozenset(state))))
        assert best >= real - 1e-9


def test_nasa_has_no_reads_and_the_same_patterns(monkeypatch):
    from comdp_plus_no_deadline.engines.temporal_stn_pdb import EngineModel

    mdp = _mdp("nasa", 1)
    ops = EngineModel(mdp).ops()
    assert not any(fp.transition_reads(ev) for op in ops.values() for ev in fp._events(op))
    goals = sorted(mdp.problem.goals, key=str)
    init = mdp.initial_state().predicates

    def patterns():
        groups = [sorted(g, key=str) for g in fp.independent_goal_groups(ops, goals)]
        return sorted((r, sorted(map(str, f))) for _g, f, r in fp.resource_patterns(ops, groups, init))

    on = patterns()
    monkeypatch.setenv("TP_MCTS_WILAO_TRANSITION_READS", "0")
    assert patterns() == on
