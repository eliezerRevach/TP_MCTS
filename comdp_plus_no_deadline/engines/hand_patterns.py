"""Pattern collections picked by hand (by an LLM reading the domain), for comparison with the
automatic growth modes (cegar | logic | pairs). TP_MCTS_WILAO_PATTERN_GROWTH = <name>.

llm_by_hand_nasa_two -- nasa_rover, object_amount = 2 (rovers r0, r1; rocks x0..x3; images o0, o1).
Per rover, two joint patterns over BOTH of its rock goals (min inside the rover, product across
rovers and images, as for every collection):

    hands  : communicated_rock_data x2, have_rock_analysis x2, free_h of both hands, ready(h, x)
             for both hands x both rocks                                            (10 facts)
             Both hands in one pattern: pointing a hand at the wrong rock (hand locking) is
             visible, which pairs (one hand per pattern, the other counted free) cannot see.
    stores : communicated_rock_data x2, have_rock_analysis x2, free_s / full / ready_to_drop of
             both stores                                                            (10 facts)
             Store blocking: a full store stops the next sample until a drop.
    image  : communicated_image_data, have_image, calibrated                         (3 facts)

llm_by_hand_nasa_one -- nasa_rover, object_amount = 1 (rover r0; rocks x0, x1; image o0): the same
picks for the one rover.
llm_by_hand_nasa_three -- nasa_rover, object_amount = 3: the same picks for rovers r0, r1, r2.

llm_by_hand_stuck_car_two -- stuck_car, object_amount = 2 (robots r0, r1; cars c0, c1). ONE joint
pattern over both cars (any robot can push any car, so the cars share both robots):
    car_out x2, tired x2, free(r, hands) x2, rock_under_car(c, good) x2, got_rock(r, good) x2
                                                                                    (10 facts)
    Hands are what every push / search / rest needs; legs are left out (counted free). The bad
    rock is left out: a hidden rock fact is read by push's probability, so VI takes its best value.

llm_by_hand_machine_shop_two | _three | _four -- machine_shop, object_amount = 2 | 3 | 4. Pieces
come in pairs (x_2k shaped + smooth on m_2k; x_2k+1 starts at m_2k, painted + polished on m_2k+1),
and every machine must end free. Two chain patterns per pair:
    even : shaped, smooth (x_2k), free(m_2k), on(x_2k, m_2k)                          (4 facts)
    odd  : painted, polished (x_2k+1), free(m_2k+1), on(x_2k+1, m_2k+1),
           at(x_2k+1, m_2k+1), hasimmersion(m_2k+1)                                   (6 facts)
A lone even piece (object_amount 3: x2 on m2) gets the even pattern.
(One 10-fact joint pattern per pair -- both chains together -- did not build in 30 min at D = 17;
the split builds in ~100 s: 1.5k + 536k table states, h(s0) 0.8019 = the odd chain.)

llm_by_hand -- picks the collection above whose goals are exactly the problem's goals and whose
facts all exist (one name for every table row).

The goals of a pattern are its communicated_* facts. Every listed fact must exist in the problem.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def _rover(rover: str, rocks: Tuple[str, str], hands: Tuple[str, str], stores: Tuple[str, str]):
    goals = [f"communicated_rock_data({x})" for x in rocks]
    have = [f"have_rock_analysis({rover}, {x})" for x in rocks]
    hand_facts = [f"free_h({h})" for h in hands] + [f"ready({h}, {x})" for h in hands for x in rocks]
    store_facts = [f for s in stores for f in (f"free_s({s})", f"full({s})", f"ready_to_drop({s})")]
    return [(goals, goals + have + hand_facts, f"hands_{rover}"),
            (goals, goals + have + store_facts, f"stores_{rover}")]


def _image(rover: str, objective: str, camera: str):
    goal = f"communicated_image_data({objective})"
    return [([goal], [goal, f"have_image({rover}, {objective})", f"calibrated({camera}, {objective})"],
             f"image_{objective}")]


def _stuck_car_two():
    goals = ["car_out(c0)", "car_out(c1)"]
    facts = goals + [f"{name}({r}{arg})" for name, arg in (("tired", ""), ("free", ", hands"), ("got_rock", ", good"))
                     for r in ("r0", "r1")] + ["rock_under_car(c0, good)", "rock_under_car(c1, good)"]
    return [(goals, facts, "robots_joint")]


def _machine_even(i: int):
    goals = [f"free(m{i})", f"shaped(x{i})", f"smooth(x{i})"]
    return [(goals, goals + [f"on(x{i}, m{i})"], f"even_x{i}")]


def _machine_pair(k: int):
    xo, mo = f"x{2 * k + 1}", f"m{2 * k + 1}"
    odd = [f"free({mo})", f"painted({xo})", f"polished({xo})"]
    return _machine_even(2 * k) + [(odd, odd + [f"on({xo}, {mo})", f"at({xo}, {mo})", f"hasimmersion({mo})"],
                                    f"odd_{xo}")]


HAND_PATTERNS: Dict[str, List[Tuple[List[str], List[str], str]]] = {
    "llm_by_hand_nasa_two": (_image("r0", "o0", "c0") + _image("r1", "o1", "c1")
                             + _rover("r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"))
                             + _rover("r1", ("x2", "x3"), ("h2", "h3"), ("s2", "s3"))),
    # nasa_rover object_amount = 1 is rover r0 of the problem above: the same picks for one rover.
    "llm_by_hand_nasa_one": (_image("r0", "o0", "c0")
                             + _rover("r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"))),
    "llm_by_hand_nasa_three": (_image("r0", "o0", "c0") + _image("r1", "o1", "c1") + _image("r2", "o2", "c2")
                               + _rover("r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"))
                               + _rover("r1", ("x2", "x3"), ("h2", "h3"), ("s2", "s3"))
                               + _rover("r2", ("x4", "x5"), ("h4", "h5"), ("s4", "s5"))),
    "llm_by_hand_stuck_car_two": _stuck_car_two(),
    "llm_by_hand_machine_shop_two": _machine_pair(0),
    "llm_by_hand_machine_shop_three": _machine_pair(0) + _machine_even(2),
    "llm_by_hand_machine_shop_four": _machine_pair(0) + _machine_pair(1),
}
AUTO = "llm_by_hand"          # picks the collection that fits the problem
HAND_GROWTHS = (AUTO,) + tuple(HAND_PATTERNS)


def pick_hand_collection(fact_by_name: Dict[str, object], goals) -> str:
    """The collection whose goals are exactly ``goals`` and whose facts all exist."""
    goal_names = {str(g) for g in goals}
    fits = [name for name, spec in HAND_PATTERNS.items()
            if {x for g, _f, _l in spec for x in g} == goal_names
            and all(x in fact_by_name for _g, facts, _l in spec for x in facts)]
    if len(fits) != 1:
        raise ValueError(f"TP_MCTS_WILAO_PATTERN_GROWTH={AUTO!r}: {len(fits)} hand collections fit this problem "
                         f"{sorted(fits)}; goals {sorted(goal_names)[:6]}")
    return fits[0]


def hand_patterns(name: str, fact_by_name: Dict[str, object], goals) -> List[Tuple[list, list, str]]:
    """``[(goals, facts, label)]`` with the problem's own fact objects. Raises if the problem does
    not have every listed fact or goal (a collection is written for one problem). ``llm_by_hand``
    picks the collection that fits."""
    if name == AUTO:
        name = pick_hand_collection(fact_by_name, goals)
    spec = HAND_PATTERNS[name]
    goal_names = {str(g) for g in goals}
    out = []
    missing = sorted({x for g, facts, _l in spec for x in list(g) + list(facts) if x not in fact_by_name})
    if missing:
        raise ValueError(f"TP_MCTS_WILAO_PATTERN_GROWTH={name!r} does not fit this problem; missing facts: "
                         f"{missing[:6]}{' ...' if len(missing) > 6 else ''}")
    uncovered = goal_names - {x for g, _f, _l in spec for x in g}
    if uncovered:
        raise ValueError(f"TP_MCTS_WILAO_PATTERN_GROWTH={name!r} leaves goals without a pattern: {sorted(uncovered)}")
    for g, facts, label in spec:
        out.append(([fact_by_name[x] for x in g], [fact_by_name[x] for x in facts], label))
    return out
