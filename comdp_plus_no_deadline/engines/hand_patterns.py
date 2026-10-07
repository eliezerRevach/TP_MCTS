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


HAND_PATTERNS: Dict[str, List[Tuple[List[str], List[str], str]]] = {
    "llm_by_hand_nasa_two": (_image("r0", "o0", "c0") + _image("r1", "o1", "c1")
                             + _rover("r0", ("x0", "x1"), ("h0", "h1"), ("s0", "s1"))
                             + _rover("r1", ("x2", "x3"), ("h2", "h3"), ("s2", "s3"))),
}


def hand_patterns(name: str, fact_by_name: Dict[str, object], goals) -> List[Tuple[list, list, str]]:
    """``[(goals, facts, label)]`` with the problem's own fact objects. Raises if the problem does
    not have every listed fact or goal (a collection is written for one problem)."""
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
