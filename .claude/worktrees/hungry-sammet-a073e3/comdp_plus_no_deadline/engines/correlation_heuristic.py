"""
Pre-planning correlation analysis and pessimistic/optimistic DP heuristics
for CoMDP+-style fact achievement under relaxed independence assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from types import SimpleNamespace
from typing import Dict, Hashable, List, Mapping, Sequence, Set, Tuple

from comdp_plus_no_deadline.engines.probabilistic_rpg import compute_precondition_support

Fact = Hashable
EPS = 1e-6


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def joint_add_distribution_from_action(action) -> Dict[frozenset, float]:
    """
    Joint distribution over sets of facts that become true (positive adds only).
    Deterministic adds merge with probabilistic outcomes via cartesian product (independent PEs).
    """
    base_add: Set[Fact] = set(getattr(action, "add_effects", set()))
    dist: Dict[frozenset, float] = {frozenset(base_add): 1.0}

    for pe in getattr(action, "probabilistic_effects", []):
        try:
            outcomes = pe.probability_function(SimpleNamespace(predicates=set()), None)
        except Exception:
            outcomes = {}
        if not outcomes:
            continue
        expanded: Dict[frozenset, float] = {}
        for add_prev, p_prev in dist.items():
            for prob, assignments in outcomes.items():
                p = _clamp01(float(prob))
                add_set = set(add_prev)
                del_set: Set[Fact] = set()
                for fact, value in assignments.items():
                    is_true = bool(value.bool_constant_value()) if hasattr(value, "bool_constant_value") else bool(value)
                    if is_true:
                        add_set.add(fact)
                    else:
                        del_set.add(fact)
                new_add = frozenset((add_set | set(add_prev)) - del_set)
                expanded[new_add] = expanded.get(new_add, 0.0) + p_prev * p
        dist = {k: _clamp01(v) for k, v in expanded.items() if v > 0}

    total = sum(dist.values())
    if total <= 0:
        return {frozenset(): 1.0}
    if abs(total - 1.0) > 1e-9:
        dist = {k: _clamp01(v / total) for k, v in dist.items()}
    return dist


def _marginal_add_prob(dist: Mapping[frozenset, float], fact: Fact) -> float:
    s = 0.0
    for outcome, p in dist.items():
        if fact in outcome:
            s += p
    return _clamp01(s)


def _joint_both_add(dist: Mapping[frozenset, float], f1: Fact, f2: Fact) -> float:
    s = 0.0
    for outcome, p in dist.items():
        if f1 in outcome and f2 in outcome:
            s += p
    return _clamp01(s)


def _joint_neither_add(dist: Mapping[frozenset, float], f1: Fact, f2: Fact) -> float:
    s = 0.0
    for outcome, p in dist.items():
        if f1 not in outcome and f2 not in outcome:
            s += p
    return _clamp01(s)


@dataclass(frozen=True)
class CorrActionSpec:
    name: str
    preconditions: frozenset[Fact]
    effect_delay_steps: int
    joint_adds: Dict[frozenset, float]


def _extract_effect_delay_steps(action) -> int:
    if hasattr(action, "start_action") and getattr(action, "start_action", None) is not None:
        try:
            total_duration = int(action.start_action.duration_int())
            return max(1, total_duration - 1)
        except Exception:
            return 1
    if hasattr(action, "end_action") and getattr(action, "end_action", None) is not None:
        return 1
    if hasattr(action, "duration_int"):
        try:
            return max(1, int(action.duration_int()))
        except Exception:
            pass
    if hasattr(action, "duration"):
        try:
            return max(1, int(action.duration.lower.int_constant_value()))
        except Exception:
            pass
    return 1


def build_action_specs(actions: Sequence[object]) -> List[CorrActionSpec]:
    specs: List[CorrActionSpec] = []
    for action in actions:
        if hasattr(action, "actions") and not hasattr(action, "add_effects"):
            continue
        pre = frozenset(getattr(action, "pos_preconditions", set()))
        joint = joint_add_distribution_from_action(action)
        if not pre and not joint:
            continue
        specs.append(
            CorrActionSpec(
                name=getattr(action, "name", repr(action)),
                preconditions=pre,
                effect_delay_steps=_extract_effect_delay_steps(action),
                joint_adds=joint,
            )
        )
    return specs


def _pair_key(f1: Fact, f2: Fact) -> frozenset:
    return frozenset({f1, f2})


def compute_correlation_preplanning(
    action_specs: Sequence[CorrActionSpec],
    goal_facts: Set[Fact],
    all_facts: Set[Fact],
) -> Tuple[Dict[frozenset, str], Set[frozenset], Dict[Fact, Set[str]], Dict[frozenset, List[Tuple[str, str]]]]:
    """
    Returns (correlation_table, joint_pairs, achievers_by_fact, pair_action_signs).
    """
    achievers_by_fact: Dict[Fact, Set[str]] = {}
    name_to_spec: Dict[str, CorrActionSpec] = {}
    for spec in action_specs:
        name_to_spec[spec.name] = spec
        for outcome in spec.joint_adds.keys():
            for fact in outcome:
                achievers_by_fact.setdefault(fact, set()).add(spec.name)

    # Step 1: backward BFS from goals (cycle nodes: second visit → do not expand further)
    relevance: Set[Fact] = set(goal_facts)
    queue: List[Fact] = list(goal_facts)
    expanded: Set[Fact] = set()
    while queue:
        f = queue.pop(0)
        if f in expanded:
            continue
        expanded.add(f)
        for aname in achievers_by_fact.get(f, set()):
            spec = name_to_spec[aname]
            for prec in spec.preconditions:
                if prec not in relevance:
                    relevance.add(prec)
                    queue.append(prec)

    relevance |= all_facts
    for spec in action_specs:
        relevance |= set(spec.preconditions)
        for out in spec.joint_adds:
            relevance |= set(out)

    # Step 2: worklist
    worklist: Set[frozenset] = set()
    gl = list(goal_facts)
    for i in range(len(gl)):
        for j in range(i + 1, len(gl)):
            worklist.add(_pair_key(gl[i], gl[j]))
    for spec in action_specs:
        pl = list(spec.preconditions)
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                worklist.add(_pair_key(pl[i], pl[j]))

    worklist = {p for p in worklist if len(p) == 2 and all(x in relevance for x in p)}

    # Step 3: per-action covariance signs
    pair_action_signs: Dict[frozenset, List[Tuple[str, str]]] = {}
    for pair in worklist:
        f1, f2 = tuple(pair)
        ach1 = achievers_by_fact.get(f1, set())
        ach2 = achievers_by_fact.get(f2, set())
        for aname in ach1 & ach2:
            spec = name_to_spec[aname]
            dist = spec.joint_adds
            p_joint = _joint_both_add(dist, f1, f2)
            p1 = _marginal_add_prob(dist, f1)
            p2 = _marginal_add_prob(dist, f2)
            cov = p_joint - p1 * p2
            if cov > EPS:
                sign = "POS"
            elif cov < -EPS:
                sign = "NEG"
            else:
                sign = "INDEP"
            pair_action_signs.setdefault(pair, []).append((aname, sign))

    tags: Dict[frozenset, str] = {p: "UNKNOWN" for p in worklist}

    def aggregate_signs(signs: List[str]) -> str:
        non_indep = [s for s in signs if s != "INDEP"]
        if not non_indep:
            return "INDEP"
        if all(s == "POS" for s in non_indep):
            return "POS"
        if all(s == "NEG" for s in non_indep):
            return "NEG"
        return "MIXED"

    for iteration in range(5):
        new_tags: Dict[frozenset, str] = {}
        for pair in worklist:
            signs: List[str] = []
            for _, s in pair_action_signs.get(pair, []):
                signs.append(s)
            f1, f2 = tuple(pair)
            for a1n in achievers_by_fact.get(f1, set()):
                for a2n in achievers_by_fact.get(f2, set()):
                    if a1n == a2n:
                        continue
                    s1 = name_to_spec[a1n]
                    s2 = name_to_spec[a2n]
                    for p1 in s1.preconditions:
                        for p2 in s2.preconditions:
                            if p1 == p2:
                                continue
                            pk = _pair_key(p1, p2)
                            if pk not in tags:
                                continue
                            up = tags[pk]
                            if up in ("UNKNOWN", "MIXED"):
                                continue
                            signs.append(up)
            new_tags[pair] = aggregate_signs(signs)
        tags = new_tags

    for pair in worklist:
        if tags[pair] == "UNKNOWN":
            tags[pair] = "MIXED"

    correlation_table: Dict[frozenset, str] = dict(tags)
    joint_pairs: Set[frozenset] = set()

    # Step 5: promotion for MIXED
    for pair in worklist:
        if correlation_table[pair] != "MIXED":
            continue
        f1, f2 = tuple(pair)
        ach = achievers_by_fact.get(f1, set()) | achievers_by_fact.get(f2, set())
        stable = True
        if not stable:
            joint_pairs.add(pair)
            continue
        M = 1.0
        prod_neg1 = 1.0
        prod_neg2 = 1.0
        for aname in ach:
            spec = name_to_spec[aname]
            dist = spec.joint_adds
            p_n = _joint_neither_add(dist, f1, f2)
            p_n1 = _clamp01(1.0 - _marginal_add_prob(dist, f1))
            p_n2 = _clamp01(1.0 - _marginal_add_prob(dist, f2))
            M *= p_n
            prod_neg1 *= p_n1
            prod_neg2 *= p_n2
        N = prod_neg1 * prod_neg2
        if abs(M - N) > EPS:
            correlation_table[pair] = "POS" if M > N else "NEG"
        else:
            correlation_table[pair] = "INDEP"
        if correlation_table[pair] == "MIXED":
            joint_pairs.add(pair)

    return correlation_table, joint_pairs, achievers_by_fact, pair_action_signs


def _and_pairwise(
    facts: Sequence[Fact],
    P: Mapping[Fact, float],
    J: Mapping[frozenset, float],
    correlation_table: Mapping[frozenset, str],
    pessimistic: bool,
) -> float:
    if len(facts) < 2:
        return _clamp01(P.get(facts[0], 0.0)) if facts else 1.0
    f1, f2 = facts[0], facts[1]
    p1 = _clamp01(P.get(f1, 0.0))
    p2 = _clamp01(P.get(f2, 0.0))
    key = _pair_key(f1, f2)
    tag = correlation_table.get(key, "INDEP")
    if tag == "MIXED":
        j = J.get(key, (1 - p1) * (1 - p2))
        return _clamp01(1 - (1 - p1) - (1 - p2) + j)
    if pessimistic:
        if tag in ("INDEP", "POS"):
            return p1 * p2
        if tag == "NEG":
            return max(0.0, p1 + p2 - 1.0)
    else:
        if tag in ("INDEP", "NEG"):
            return p1 * p2
        if tag == "POS":
            return min(p1, p2)
    return p1 * p2


def _and_n_facts(
    facts: Sequence[Fact],
    P: Mapping[Fact, float],
    J: Mapping[frozenset, float],
    correlation_table: Mapping[frozenset, str],
    pessimistic: bool,
) -> float:
    facts = list(facts)
    n = len(facts)
    if n == 0:
        return 1.0
    if n == 1:
        return _clamp01(P.get(facts[0], 0.0))
    if n == 2:
        return _and_pairwise(facts, P, J, correlation_table, pessimistic)

    sum_marg = sum(_clamp01(P.get(f, 0.0)) for f in facts)
    if pessimistic:
        return max(0.0, sum_marg - (n - 1))
    return min(_clamp01(P.get(f, 0.0)) for f in facts)


def run_correlation_dp(
    state_facts: Set[Fact],
    goal_facts: Sequence[Fact],
    action_specs: Sequence[CorrActionSpec],
    achievers_by_fact: Mapping[Fact, Set[str]],
    name_to_spec: Dict[str, CorrActionSpec],
    correlation_table: Mapping[frozenset, str],
    joint_pairs: Set[frozenset],
    deadline: int,
    pessimistic: bool,
) -> float:
    """Forward DP; returns estimated P(all goals by deadline)."""
    T = max(0, int(deadline))
    facts_needed: Set[Fact] = set(state_facts)
    for g in goal_facts:
        facts_needed.add(g)
    for spec in action_specs:
        facts_needed |= set(spec.preconditions)
        for o in spec.joint_adds:
            facts_needed |= set(o)

    P: Dict[Fact, float] = {f: 1.0 if f in state_facts else 0.0 for f in facts_needed}
    J: Dict[frozenset, float] = {}
    for pair in joint_pairs:
        f1, f2 = tuple(pair)
        p1 = P.get(f1, 0.0)
        p2 = P.get(f2, 0.0)
        J[pair] = (1 - p1) * (1 - p2)

    saturated: Set[Fact] = {f for f, v in P.items() if v >= 1.0 - EPS}

    for t in range(1, T + 1):
        P_old = dict(P)
        P_new = dict(P_old)

        for f in facts_needed:
            if f in saturated:
                continue
            ach = list(achievers_by_fact.get(f, set()))
            if not ach:
                continue
            hazard_parts: List[float] = []
            for aname in ach:
                spec = name_to_spec[aname]
                d = spec.effect_delay_steps
                layer = t - d
                if layer < 0:
                    hazard_parts.append(0.0)
                    continue
                probs_for_prec = {p: P_old.get(p, 0.0) for p in spec.preconditions}
                r = _clamp01(compute_precondition_support(spec.preconditions, probs_for_prec, strict=True))
                pa_f = _marginal_add_prob(spec.joint_adds, f)
                hazard_parts.append(_clamp01(r * pa_f))
            if not hazard_parts or max(hazard_parts) <= EPS:
                continue
            H = 1.0 - prod(1.0 - h for h in hazard_parts)
            old = P_new.get(f, 0.0)
            P_new[f] = _clamp01(old + (1.0 - old) * H)
            if P_new[f] >= 1.0 - EPS:
                saturated.add(f)

        for pair in joint_pairs:
            f1, f2 = tuple(pair)
            factor = 1.0
            for aname in achievers_by_fact.get(f1, set()) | achievers_by_fact.get(f2, set()):
                spec = name_to_spec[aname]
                d = spec.effect_delay_steps
                layer = t - d
                if layer < 0:
                    r = 0.0
                else:
                    probs_for_prec = {p: P_old.get(p, 0.0) for p in spec.preconditions}
                    r = _clamp01(compute_precondition_support(spec.preconditions, probs_for_prec, strict=True))
                p_miss = _joint_neither_add(spec.joint_adds, f1, f2)
                factor *= 1.0 - r + r * p_miss
            J[pair] = _clamp01(J.get(pair, 0.0) * factor)

        P = P_new

    goals = list(goal_facts)
    return _clamp01(_and_n_facts(goals, P, J, correlation_table, pessimistic))
