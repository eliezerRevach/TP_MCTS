"""Static facts in ``FactPatternModel``: a fact some action reads and no action changes keeps its
initial value in every pattern (it used to count as true, so a bad hand could do a good-hand sample)."""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.fact_pattern import FactPatternModel, grow_pattern, static_facts  # noqa: E402
from comdp_plus_no_deadline.engines.temporal_stn_pdb import DurativeOp  # noqa: E402
from comdp_plus_no_deadline.engines.windows_lao import WindowsLAO, WindowsTable  # noqa: E402


def event(pre=(), neg=(), adds=(), dels=()):
    return SimpleNamespace(pos_preconditions=set(pre), neg_preconditions=set(neg), add_effects=set(adds),
                           del_effects=set(dels), probabilistic_effects=[])


def durative(name, d, pre=(), neg=(), adds=(), dels=()):
    running = f"inExecution({name})"
    start = event(pre=pre, neg=tuple(neg) + (running,), adds=(running,))
    end = event(pre=(running,), adds=adds, dels=tuple(dels) + (running,))
    return DurativeOp(key=name, duration=d, start_action=start, end_action=end)


class Base:
    """Deterministic engine-like model; records the completed states outcomes() is asked about."""

    def __init__(self, ops):
        self._ops = {op.key: op for op in ops}
        self.asked = []

    def ops(self):
        return self._ops

    def outcomes(self, action, facts):
        self.asked.append(frozenset(facts))
        return ((frozenset((set(facts) - action.del_effects) | action.add_effects), 1.0),)

    def start_is_inert(self, key):
        return False

    def deadline(self):
        return 10


# g <- fast (1, needs the static `good`) | slow (3, needs NOT good);  both need `ready`, which prep adds
OPS = [
    durative("fast", 1, pre=("good", "ready"), adds=("g",)),
    durative("slow", 3, pre=("ready",), neg=("good",), adds=("g",)),
    durative("prep", 1, adds=("ready",)),
]


def test_static_facts_are_read_and_never_changed():
    assert static_facts(Base(OPS).ops()) == {"good"}


def test_a_false_static_blocks_its_action_in_the_pattern():
    model = FactPatternModel(Base(OPS), ["g"], ["g"], initial_facts=set(), exact_statics=True)
    assert "fast" not in model.ops()                      # needs good, which is false forever
    assert model.legal_action_names(model.project(set())) == ("slow",)


def test_a_true_static_blocks_the_action_that_needs_it_false():
    model = FactPatternModel(Base(OPS), ["g"], ["g"], initial_facts={"good"}, exact_statics=True)
    assert "slow" not in model.ops()
    assert model.legal_action_names(model.project({"good"})) == ("fast",)


def test_old_relaxation_lets_the_impossible_action_run():
    model = FactPatternModel(Base(OPS), ["g"], ["g"], initial_facts=set(), exact_statics=False)
    assert set(model.legal_action_names(model.project(set()))) == {"fast", "slow"}


def test_the_value_sees_the_slow_route_only():
    # real: good is false, so only slow (3) reaches g; ready is outside the pattern (counted true)
    for exact, at_2 in ((True, 0.0), (False, 1.0)):
        model = FactPatternModel(Base(OPS), ["g"], ["g"], initial_facts=set(), exact_statics=exact)
        table = WindowsTable(WindowsLAO(model, heuristic="none"), 10)
        assert table.build(model.project(set()))
        assert table.lookup(model.project(set()), 2)[0] == at_2     # fast would finish by 1
        assert table.lookup(model.project(set()), 3)[0] == 1.0      # slow finishes by 3


def test_outcomes_complete_the_state_with_the_statics_initial_value():
    base = Base(OPS)
    model = FactPatternModel(base, ["g"], ["g"], initial_facts={"good"}, exact_statics=True)
    model.outcomes(base.ops()["fast"].start_action, model.project({"good"}))
    assert "good" in base.asked[-1] and "ready" in base.asked[-1]   # static as is; outside `ready` freed


def test_environment_knob_restores_the_old_relaxation(monkeypatch):
    monkeypatch.setenv("TP_MCTS_WILAO_EXACT_STATICS", "0")
    model = FactPatternModel(Base(OPS), ["g"], ["g"], initial_facts=set())
    assert "fast" in model.ops()
    monkeypatch.setenv("TP_MCTS_WILAO_EXACT_STATICS", "1")
    model = FactPatternModel(Base(OPS), ["g"], ["g"], initial_facts=set())
    assert "fast" not in model.ops()


def test_grow_pattern_never_spends_a_slot_on_a_static():
    ops = Base(OPS).ops()
    assert "good" not in grow_pattern(ops, ["g"], set(), 5)
    assert "good" in grow_pattern(ops, ["g"], set(), 5, skip_statics=False)   # old: 'not initial' -> first


def _nasa_obj1():
    up = pytest.importorskip("unified_planning")
    import unified_planning.domains  # noqa: F401
    from unified_planning.engines.convert_problem import Convert_problem
    from unified_planning.engines.mdp import MDP
    from comdp_plus_no_deadline.engines.temporal_stn_pdb import EngineModel

    domain = up.domains.Nasa_Rover(kind="regular", deadline=25, object_amount=1)
    grounded = up.engines.Grounder(domain.grounding_map())._compile(domain.problem).problem
    mdp = MDP(Convert_problem(grounded)._converted_problem, discount_factor=1.0, reward_mode="terminal",
              step_penalty=0.0)
    return mdp, EngineModel(mdp)


def test_nasa_bad_hand_never_does_a_good_hand_sample():
    mdp, base = _nasa_obj1()
    init = frozenset(mdp.initial_state().predicates)
    by_name = {str(f): f for f in init | set(mdp.problem.goals)}
    for op in base.ops().values():
        for ev in (op.start_action, op.end_action):
            if ev is not None:
                for f in set(ev.pos_preconditions) | set(ev.add_effects) | set(ev.del_effects):
                    by_name.setdefault(str(f), f)
    assert "good(h0)" in {str(f) for f in init} and "good(h1)" not in {str(f) for f in init}
    names = ["communicated_rock_data(x1)", "have_rock_analysis(r0, x1)", "free_h(h1)", "ready(h1, x1)"]
    model = FactPatternModel(base, [by_name[n] for n in names], [by_name[names[0]]], init, exact_statics=True)
    impossible = [k for k in model.ops() if k.startswith("start_sample_rock_good_") and k.endswith("_h1")]
    impossible += [k for k in model.ops() if k.startswith("turn_on_hand_h0_")]
    assert impossible == []
    old = FactPatternModel(base, [by_name[n] for n in names], [by_name[names[0]]], init, exact_statics=False)
    assert any(k.startswith("start_sample_rock_good_") and k.endswith("_h1") for k in old.ops())
    # pointed at x1, h1 can start only its normal sample in the pattern -- as in reality
    state = model.project((init | {by_name["ready(h1, x1)"]}) - {by_name["free_h(h1)"]})
    samples = [k for k in model.legal_action_names(state) if "sample" in k and k.endswith("_h1")]
    assert samples and all(not k.startswith("start_sample_rock_good_") for k in samples)
