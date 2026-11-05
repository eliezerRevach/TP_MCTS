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

Smaller nasa_rover object_amount = 1 picks, for the STN rules (windows_lao.py "rules = stn"), whose
table grows much faster with pattern size than the windows table; name them explicitly (llm_by_hand
would see several fitting collections):
llm_by_hand_nasa_one_rock -- one rock per pattern:
    hand  : communicated_rock_data(x), have_rock_analysis(r0, x), free_h(h), ready(h, x)    (4 facts)
    store : communicated_rock_data(x), have_rock_analysis(r0, x), free_s / full / ready_to_drop(s)
                                                                                         (5 facts)
    for each rock x0, x1 and each hand / store, plus the image pattern. The two rocks share a group,
    so their values meet in a min (single-rock values: "both rocks by the deadline" is not seen).
llm_by_hand_nasa_one_rock6 | _rock7 | _rock12 -- one rock per pattern, more resource facts (sized for the STN
    rules' build time):
    rock6  : communicated, have_rock_analysis, free_h + ready of BOTH hands (6 facts), + the 5-fact store patterns
    rock7  : communicated, have_rock_analysis, free_h + ready of one hand, free_s / full / ready_to_drop of one
             store (7 facts; a sample needs both at once), one pattern per hand x store
    rock12 : communicated, have_rock_analysis, both hands, both stores (12 facts)
llm_by_hand_nasa_one_joint -- both rocks in every pattern, fewer resource facts:
    store : communicated_rock_data x2, have_rock_analysis x2, full(s)                       (5 facts)
            a full store needs a drop before the next sample (ready_to_drop counted free)
    hand  : communicated_rock_data x2, have_rock_analysis x2, ready(h, x0), ready(h, x1)    (6 facts)
            the turn-on before a sample is seen (free_h counted free: no hand lock)
    for each store s0, s1 and hand h0, h1, plus the image pattern.

llm_by_hand_stuck_car_two -- stuck_car, object_amount = 2 (robots r0, r1; cars c0, c1). ONE joint
pattern over both cars (any robot can push any car, so the cars share both robots):
    car_out x2, tired x2, free(r, hands) x2, rock_under_car(c, good) x2, got_rock(r, good) x2
                                                                                    (10 facts)
    Hands are what every push / search / rest needs; legs are left out (counted free). The bad
    rock is left out: a hidden rock fact is read by push's probability, so VI takes its best value.

llm_by_hand_stuck_car_one -- stuck_car, object_amount = 1: one pattern with the robot's state too
    car_out(c0), tired(r0), free(r0, hands), free(r0, legs), got_rock(r0, good|bad),
    rock_under_car(c0, good|bad)                                                     (8 facts)
    The automatic pattern (5 facts) leaves out tired / free: resting after a push is invisible.

llm_by_hand_conc -- conc (deterministic; one5 needs b, h, f and DELETES b, so b is redone after i):
    b_i   : got(b), got(i), got(f), got(h)          the delete interaction         (4 facts)
    chain : got(c), got(d), got(e), got(f), got(g), got(h)   the precondition chain   (6 facts)
    a     : got(a)                                                                   (1 fact)
    The automatic collection (a + one 8-goal joint) did not build in 30 min.

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


def _rover_by_rock(rover: str, rocks: Tuple[str, str], hands: Tuple[str, str], stores: Tuple[str, str]):
    out = []
    for x in rocks:
        base = [f"communicated_rock_data({x})", f"have_rock_analysis({rover}, {x})"]
        out += [([base[0]], base + [f"free_h({h})", f"ready({h}, {x})"], f"hand_{h}_{x}") for h in hands]
        out += [([base[0]], base + [f"free_s({s})", f"full({s})", f"ready_to_drop({s})"], f"store_{s}_{x}")
                for s in stores]
    return out


def _rover_by_rock_grown(rover: str, rocks: Tuple[str, str], hands: Tuple[str, str], stores: Tuple[str, str],
                         how: str):
    """One rock per pattern, more resource facts than _rover_by_rock (see the module doc)."""
    out = []
    for x in rocks:
        base = [f"communicated_rock_data({x})", f"have_rock_analysis({rover}, {x})"]
        hand = {h: [f"free_h({h})", f"ready({h}, {x})"] for h in hands}
        store = {s: [f"free_s({s})", f"full({s})", f"ready_to_drop({s})"] for s in stores}
        if how == "hands":                                   # 6 facts: both hands, + the 5-fact store patterns
            out.append(([base[0]], base + hand[hands[0]] + hand[hands[1]], f"hands_{x}"))
            out += [([base[0]], base + store[s], f"store_{s}_{x}") for s in stores]
        elif how == "hand_store":                            # 7 facts: one hand x one store
            out += [([base[0]], base + hand[h] + store[s], f"{h}_{s}_{x}") for h in hands for s in stores]
        else:                                                # 12 facts: both hands, both stores
            out.append(([base[0]], base + hand[hands[0]] + hand[hands[1]] + store[stores[0]] + store[stores[1]],
                        f"all_{x}"))
    return out


def _rover_joint_small(rover: str, rocks: Tuple[str, str], hands: Tuple[str, str], stores: Tuple[str, str]):
    goals = [f"communicated_rock_data({x})" for x in rocks]
    base = goals + [f"have_rock_analysis({rover}, {x})" for x in rocks]
    return ([(goals, base + [f"full({s})"], f"store_{s}") for s in stores]
            + [(goals, base + [f"ready({h}, {x})" for x in rocks], f"hand_{h}") for h in hands])


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


def _stuck_car_one():
    goal = ["car_out(c0)"]
    facts = goal + ["tired(r0)", "free(r0, hands)", "free(r0, legs)", "got_rock(r0, good)", "got_rock(r0, bad)",
                    "rock_under_car(c0, good)", "rock_under_car(c0, bad)"]
    return [(goal, facts, "robot_car")]


def _conc():
    got = lambda *xs: [f"got({x})" for x in xs]
    return [(got("b", "i"), got("b", "i", "f", "h"), "b_i"),
            (got("c", "d", "e", "f", "g", "h"), got("c", "d", "e", "f", "g", "h"), "chain"),
            (got("a"), got("a"), "a")]


HAND_PATTERNS: Dict[str, List[Tuple[List[str], List[str], str]]] = {
    "llm_by_hand_nasa_two": (_image("r0", "o0", "c0") + _image("r1", "o1", "c1")
                             + _rover("r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"))
                             + _rover("r1", ("x2", "x3"), ("h2", "h3"), ("s2", "s3"))),
    # nasa_rover object_amount = 1 is rover r0 of the problem above: the same picks for one rover.
    "llm_by_hand_nasa_one": (_image("r0", "o0", "c0")
                             + _rover("r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"))),
    "llm_by_hand_nasa_one_rock": (_image("r0", "o0", "c0")
                                  + _rover_by_rock("r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"))),
    "llm_by_hand_nasa_one_joint": (_image("r0", "o0", "c0")
                                   + _rover_joint_small("r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"))),
    "llm_by_hand_nasa_one_rock6": (_image("r0", "o0", "c0")
                                   + _rover_by_rock_grown("r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"), "hands")),
    "llm_by_hand_nasa_one_rock7": (_image("r0", "o0", "c0") + _rover_by_rock_grown(
        "r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"), "hand_store")),
    "llm_by_hand_nasa_one_rock12": (_image("r0", "o0", "c0")
                                    + _rover_by_rock_grown("r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"), "all")),
    "llm_by_hand_nasa_three":(_image("r0", "o0", "c0") + _image("r1", "o1", "c1") + _image("r2", "o2", "c2")
                               + _rover("r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"))
                               + _rover("r1", ("x2", "x3"), ("h2", "h3"), ("s2", "s3"))
                               + _rover("r2", ("x4", "x5"), ("h4", "h5"), ("s4", "s5"))),
    "llm_by_hand_stuck_car_one": _stuck_car_one(),
    "llm_by_hand_stuck_car_two": _stuck_car_two(),
    "llm_by_hand_conc": _conc(),
    "llm_by_hand_machine_shop_two": _machine_pair(0),
    "llm_by_hand_machine_shop_three": _machine_pair(0) + _machine_even(2),
    "llm_by_hand_machine_shop_four": _machine_pair(0) + _machine_pair(1),
}
AUTO = "llm_by_hand"          # picks the collection that fits the problem
EXPLICIT_ONLY = frozenset({"llm_by_hand_nasa_one_rock", "llm_by_hand_nasa_one_joint", "llm_by_hand_nasa_one_rock6",
                           "llm_by_hand_nasa_one_rock7", "llm_by_hand_nasa_one_rock12"})   # never picked by AUTO
HAND_GROWTHS = (AUTO,) + tuple(HAND_PATTERNS)


def pick_hand_collection(fact_by_name: Dict[str, object], goals) -> str:
    """The collection whose goals are exactly ``goals`` and whose facts all exist."""
    goal_names = {str(g) for g in goals}
    fits = [name for name, spec in HAND_PATTERNS.items()
            if name not in EXPLICIT_ONLY and {x for g, _f, _l in spec for x in g} == goal_names
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
