"""Part I of the symbolic-STN temporal PDB (``temporal_stn_pdb``).

Implements ``artifacts/Temporal_STN_PDB_Exact_Draft.docx`` sections 1-9, naive
variant: full history, no compression (section 10 is Part II and is NOT here),
and no fact projection (phi = identity, per section 7's "Start with the full
semantic state").

What makes this different from every other engine in this package
-----------------------------------------------------------------
The PTRPG family and ``exact_pattern_mdp`` both RE-DERIVE the domain: they read
``add_effects``/``probabilistic_effects`` off the action objects and rebuild an
abstract transition law. Section 1 forbids that here -- "Retain the supplied
domain's start and end effects, adds and deletes, ... None are removed merely to
simplify storage" -- so this module never rebuilds anything. It drives expansion
through the planner's OWN ``mdp.legal_actions`` and ``mdp.transition_function``
(see :class:`EngineModel`). Conditions, deletes, mutex via ``inExecution``, and
state-dependent outcome probabilities are therefore exact by construction rather
than by careful re-implementation: they are the same calls the search makes.

The temporal model
------------------
In ``domain_type='regular'`` this codebase's ``MDP`` is UNTIMED -- states are
bare fact sets and ``step`` never advances a clock. Time lives in the STN that
TP-MCTS maintains alongside the tree (``unified_planning/engines/utils.py``).
So this heuristic maintains its own STN in exactly the same style, which is
precisely the object section 2 describes:

    X = annotated Z;   (s, R, C) = REPLAY(initial snapshot, X)

``Z`` is a :class:`~comdp_plus_no_deadline.engines.strict_dbm.StrictDBM` over
event variables; the annotation is the processed-event history with outcomes.
Facts are DERIVED by replay and merely cached on the node ("Facts may be cached,
but are not required as a separate field in Part I").

Two deliberate, provable simplifications
----------------------------------------
1. **Section 3's endpoint placements are not enumerated at START time.** They
   are exactly the ways the pending ends can be processed, so section 4's END
   group enumeration already generates all of them: ``E_a < E_r`` is END{a}
   first, ``E_a = E_r`` is END{a,r} tied, ``E_a > E_r`` is END{r} first. Doing
   both would double-count the same order-regions. Set ``eager_placement=True``
   to branch section-3-literally instead and check the two agree.

2. **A start strictly after D is never generated** (``S_a <= D``). Reward is
   collected "at a permitted event by D" (section 6), and every event of an
   instance started after D is itself after D, so such a start can change
   neither the reward nor any fact observed before D. This is the ONLY cap
   imposed; section 5's "add no unproved cap on legal actions or concurrent
   instances" is otherwise respected -- concurrency is bounded by the domain's
   own ``inExecution`` mutex, not by a prototype restriction.

Value: an INTERVAL, not a scalar
--------------------------------
Sections 5 and 11 insist "UNKNOWN is not value zero" and "Do not label UNKNOWN
or an unevaluated graph an exact solution". A single float cannot express that,
so :meth:`TemporalSTNPDB.solve` returns ``(lo, hi)``: a budget-cut leaf is
``(0, 1)``, and a fully expanded subtree has ``lo == hi``, which is the exact
value. ``complete`` on the result says which happened -- so "this number is
exact" is a fact the caller can check, not a claim this module makes.

Exactness, and how to bracket it
--------------------------------
Section 6's backup is ``V(X, sigma) = sup_{u, eta} Sum_omega p [r + V]`` where
``eta`` is a scheduling commitment SHARED across the outcome branches, and the
draft states plainly that implementing it is unresolved and that "independent
scalar child maxima are not an exact substitute".

The obstacle is structural, not an oversight. A symbolic node stands for a SET
of ground states -- one per feasible assignment of its event times -- and one
scalar max over that set silently lets two outcome branches of the same decision
each pick a different assignment of a time that was committed BEFORE the branch.
A real execution cannot: it committed one number, then observed. So a per-node
scalar max is an upper bound exactly when the node's region holds more than one
point, and symbolic time is what creates those regions. Symbolic time and this
backup's exactness are in direct tension; keeping both is section 6's open
problem, not something a careful implementation avoids.

What this module does instead is BRACKET the value with two runs:

* ``pin_starts_to_clock=False`` (default, section 4 as written): a start may sit
  anywhere in ``[C, next end)``. Regions are not points, so ``hi`` is an UPPER
  bound on the nonanticipating optimum.
* ``pin_starts_to_clock=True``: a start happens exactly AT the current event
  boundary, ``S_a = C``. Every event time is then pinned to a rational, every
  region is a single point, the per-node max IS the ordinary MDP backup, and the
  value is EXACT for that restricted policy class -- hence a LOWER bound on the
  unrestricted optimum.

So ``V_pinned <= V* <= V_symbolic``, and the bracket collapses exactly when
restricting starts to event boundaries loses nothing. This repository has
already checked that condition once: epoch-only starts were optimality-
preserving on 430/430 exhaustive cases with no at-end conditions, and need the
generator set seeded with the DEADLINE otherwise. Section 1 keeps end
conditions, so the seeded form is the one that would apply here.

``starts_pinned`` and ``branchwise_gap_witnessed`` on the result say which side
of the bracket a number came from. ``exact`` is True only when the graph closed
AND every region was a point -- a certificate the caller can check, never a
claim this module makes on its own.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import (
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from comdp_plus_no_deadline.engines.strict_dbm import StrictDBM

Fact = Hashable

DEFAULT_NODE_BUDGET = 20_000
DEFAULT_MAX_PREFIX_EVENTS = 64
_EPS = 1e-12


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "", "false", "no")


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DurativeOp:
    """One startable action type, with its two snap actions and duration.

    ``key`` identifies the TYPE. Section 8 requires repeated instances of one
    type to stay distinct, so instances carry a serial number on top of this.
    """

    key: str
    duration: Fraction
    start_action: object
    end_action: Optional[object]        # None => instantaneous (duration 0)


class EngineModel:
    """Adapter over this codebase's own ``MDP``. Re-implements nothing.

    ``legal_actions`` gives section 4's condition check (positive AND negative
    preconditions, the ``inExecution`` mutex that forbids a second concurrent
    instance, and the engine's ``check_action_relevant`` pruning).
    ``transition_function`` gives section 1's ``APPLY(s, e, omega)`` together
    with section 4's outcome kernel ``p(omega | X, u)`` -- evaluated on the REAL
    derived state, so a state-dependent probability function is read exactly as
    the model defines it rather than probed at guessed extremes.
    """

    def __init__(self, mdp, respect_relevance_pruning: bool = True, allowed_ops=None,
                 goals=None):
        self._mdp = mdp
        # A pattern must be scored against ITS OWN goal. Leaving the problem's
        # full goal conjunction here while restricting the action set makes
        # every pattern unsatisfiable, and the solver dutifully returns 0.
        self._goals = None if goals is None else frozenset(goals)
        self._respect_relevance = respect_relevance_pruning
        # Action-set projection (see goal_relevance_closure). Facts are NOT
        # projected: the derived state stays complete, so preconditions and
        # state-dependent probability functions are still evaluated exactly.
        # Section 7's hazard -- "Omitting a fact read by a condition,
        # probability or reward can change the solution" -- therefore does not
        # arise; only actions that can neither achieve nor threaten anything in
        # the closure are dropped.
        self._allowed_ops = None if allowed_ops is None else frozenset(allowed_ops)
        self._state_cls = type(mdp.initial_state())
        self._ops: Optional[Dict[str, DurativeOp]] = None
        self._legal_cache: Dict[FrozenSet, Tuple[object, ...]] = {}
        self._outcome_cache: Dict[Tuple[FrozenSet, int], Tuple] = {}
        self._inert_starts: Dict[str, bool] = {}
        self.lost_probability_mass = 0.0

    # -- structure ------------------------------------------------------
    @property
    def goals(self) -> FrozenSet:
        if self._goals is not None:
            return self._goals
        return frozenset(self._mdp.problem.goals)

    def deadline(self):
        return self._mdp.deadline()

    def ops(self) -> Dict[str, DurativeOp]:
        if self._ops is not None:
            return self._ops
        import unified_planning as up

        ops: Dict[str, DurativeOp] = {}
        for action in self._mdp.problem.actions:
            if isinstance(action, up.engines.NoOpAction):
                continue
            end_action = getattr(action, "end_action", None)
            if end_action is not None:                      # snap start action
                ops[action.name] = DurativeOp(
                    key=action.name,
                    duration=Fraction(int(action.duration_int())),
                    start_action=action,
                    end_action=end_action,
                )
            elif getattr(action, "start_action", None) is None:
                # A plain instantaneous action: one event, zero duration.
                ops[action.name] = DurativeOp(
                    key=action.name,
                    duration=Fraction(0),
                    start_action=action,
                    end_action=None,
                )
        if self._allowed_ops is not None:
            ops = {k: v for k, v in ops.items() if k in self._allowed_ops}
        self._ops = ops
        return ops

    def _state(self, facts: FrozenSet):
        return self._state_cls(set(facts))

    # -- queries --------------------------------------------------------
    def legal_action_names(self, facts: FrozenSet) -> Tuple[str, ...]:
        cached = self._legal_cache.get(facts)
        if cached is not None:
            return cached
        state = self._state(facts)
        if self._respect_relevance:
            actions = self._mdp.legal_actions(state)
        else:
            actions = [
                a for a in self._mdp.problem.actions
                if not _is_noop(a)
                and a.pos_preconditions.issubset(facts)
                and a.neg_preconditions.isdisjoint(facts)
            ]
        names = tuple(a.name for a in actions)
        if self._allowed_ops is not None:
            allowed = self._allowed_ops
            # End actions keep their own name, so only START candidates are
            # filtered here; ends are gated by the running set instead.
            names = tuple(n for n in names if n in allowed or n not in self._all_start_names())
        self._legal_cache[facts] = names
        return names

    def _all_start_names(self):
        cached = getattr(self, "_start_names", None)
        if cached is None:
            cached = frozenset(
                a.name for a in self._mdp.problem.actions
                if getattr(a, "end_action", None) is not None
                or (not _is_noop(a) and getattr(a, "start_action", None) is None)
            )
            self._start_names = cached
        return cached

    def outcomes(self, action, facts: FrozenSet) -> Tuple[Tuple[FrozenSet, float], ...]:
        """``[(next_facts, p)]`` for one processed event, folded and summed.

        Section 4: "Sum outcomes reaching an identical annotated successor."
        Two probabilistic branches whose projected effects coincide must become
        ONE successor with the summed mass, or the same successor is counted as
        two competing plans.
        """
        key = (facts, id(action))
        cached = self._outcome_cache.get(key)
        if cached is not None:
            return cached
        folded: Dict[FrozenSet, float] = {}
        for next_state, prob in self._mdp.transition_function(self._state(facts), action):
            p = float(prob)
            if p <= _EPS:
                continue
            nxt = frozenset(next_state.predicates)
            folded[nxt] = folded.get(nxt, 0.0) + p
        total = sum(folded.values())
        if not folded:
            folded = {facts: 1.0}
            total = 1.0
        if total < 1.0 - 1e-9:
            # The engine's own sampler (np.random.choice) assumes the outcome
            # masses sum to 1, so this is a malformed probability function
            # rather than a modelling choice. Recorded, never silently
            # renormalised -- renormalising would promote a partial chance to a
            # certainty, which section 4's "Preserve every positive-probability
            # failure branch" forbids.
            self.lost_probability_mass = max(self.lost_probability_mass, 1.0 - total)
        out = tuple(folded.items())
        self._outcome_cache[key] = out
        return out

    def start_is_inert(self, key: str) -> bool:
        """Is starting this op observationally useless until it ENDS?

        True when the start action changes nothing except its own
        ``inExecution`` slot AND no other action reads that slot positively.
        Under those two conditions an instance whose end falls after D is
        strictly dominated by not starting it at all: it cannot change a fact,
        cannot enable anything, and the slot it occupies only blocks its own
        retry. Section 2 rightly forbids the blanket ``E_a <= D`` -- "a start
        effect may achieve the goal while the action continues" -- but that
        caveat is exactly what this test checks for, per action, rather than
        assuming it away or ignoring it.
        """
        cached = self._inert_starts.get(key)
        if cached is not None:
            return cached
        op = self.ops().get(key)
        if op is None or op.end_action is None:
            self._inert_starts[key] = False
            return False
        start = op.start_action
        own_slot = set(getattr(start, "add_effects", set()) or set())
        own_slot &= set(getattr(op.end_action, "del_effects", set()) or set())
        adds = set(getattr(start, "add_effects", set()) or set()) - own_slot
        dels = set(getattr(start, "del_effects", set()) or set())
        probabilistic = list(getattr(start, "probabilistic_effects", []) or [])
        inert = not adds and not dels and not probabilistic and bool(own_slot)
        if inert:
            for other in self._mdp.problem.actions:
                if other is start or other is op.end_action:
                    continue
                if own_slot & set(getattr(other, "pos_preconditions", set()) or set()):
                    inert = False           # somebody benefits from it running
                    break
        self._inert_starts[key] = inert
        return inert

    def action_by_name(self, name: str):
        return self._mdp.problem.action_by_name(name)


def _action_adds(action) -> Set[Fact]:
    out = set(getattr(action, "add_effects", set()) or set())
    for pe in getattr(action, "probabilistic_effects", []) or []:
        out |= set(getattr(pe, "fluents", []) or [])
    return out


def goal_relevance_closure(model, goal: Fact, with_threats: bool = True) -> FrozenSet[str]:
    """Actions that can matter for ``goal``, grown BACKWARD to a fixpoint.

    Seed with ``goal``; keep any action that ADDS something needed (an achiever)
    or -- when ``with_threats`` -- DELETES something needed (a threat); then add
    those actions' own preconditions to the needed set and repeat.

    Keeping threats is what makes the projection exact rather than merely
    optimistic. Dropping an achiever would lower the value (an OR-drop, unsound
    for a bound); dropping a threat would RAISE it. Only actions that can do
    neither are discarded, and for those the projected problem and the original
    have the same value -- nothing outside the closure can touch any fact the
    closure reads or writes.

    Note this projects the ACTION set only. Facts are untouched, so section 7's
    "Do not free omitted preconditions by default" is respected by construction:
    no precondition is freed, because no fact is dropped.
    """
    ops = model.ops()
    needed: Set[Fact] = {goal}
    relevant: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for key, op in ops.items():
            if key in relevant:
                continue
            events = [op.start_action] + ([op.end_action] if op.end_action else [])
            touches = any(_action_adds(a) & needed for a in events)
            if with_threats and not touches:
                touches = any(
                    set(getattr(a, "del_effects", set()) or set()) & needed for a in events
                )
            if not touches:
                continue
            relevant.add(key)
            changed = True
            for a in events:
                needed |= set(getattr(a, "pos_preconditions", set()) or set())
    return frozenset(relevant)


def _is_noop(action) -> bool:
    import unified_planning as up

    return isinstance(action, up.engines.NoOpAction)


def in_execution(key: str) -> Tuple[str, str]:
    """The toy stand-in for the compilation's ``inExecution(start-<a>)`` fact."""
    return ("inExecution", key)


class ToyModel:
    """Explicit little model for the section-9 checks and unit tests.

    Same interface as :class:`EngineModel`. Used ONLY where the spec states the
    expected number itself, so a disagreement points at this module rather than
    at a re-derived domain.

    It mirrors ``Convert_problem`` rather than inventing its own semantics: a
    start adds ``inExecution(a)`` and is barred while that fact holds, and the
    matching end requires it and removes it. That single fact is what serialises
    retries; without it a duration-2 action could start a fresh overlapping copy
    at every instant and the graph would not be finite. Concurrency is therefore
    bounded by the modelled domain, exactly as section 5 requires -- not by a
    cap this module imposes.
    """

    def __init__(self, ops: Sequence[dict], goals: Iterable[Fact], deadline):
        self._ops: Dict[str, DurativeOp] = {}
        self._spec: Dict[str, dict] = {}
        for spec in ops:
            key = spec["name"]
            self._spec[key] = spec
            duration = Fraction(spec.get("duration", 1))
            self._ops[key] = DurativeOp(
                key=key,
                duration=duration,
                start_action=key,
                end_action=(key if duration != 0 else None),
            )
        self._goals = frozenset(goals)
        self._deadline = deadline
        self.lost_probability_mass = 0.0

    @property
    def goals(self) -> FrozenSet:
        return self._goals

    def deadline(self):
        return self._deadline

    def ops(self) -> Dict[str, DurativeOp]:
        return self._ops

    def _durative(self, key: str) -> bool:
        return self._ops[key].end_action is not None

    def legal_action_names(self, facts: FrozenSet) -> Tuple[str, ...]:
        out = []
        for key, spec in self._spec.items():
            if self._durative(key) and in_execution(key) in facts:
                continue                       # the inExecution mutex
            if not set(spec.get("pre", ())).issubset(facts):
                continue
            if set(spec.get("neg_pre", ())) & facts:
                continue
            out.append(key)
        return tuple(out)

    def end_legal(self, key: str, facts: FrozenSet) -> bool:
        if in_execution(key) not in facts:
            return False
        spec = self._spec[key]
        if not set(spec.get("end_pre", ())).issubset(facts):
            return False
        return not (set(spec.get("end_neg_pre", ())) & facts)

    def outcomes(self, action, facts: FrozenSet) -> Tuple[Tuple[FrozenSet, float], ...]:
        key, which = action
        spec = self._spec[key]
        branches = spec.get("start" if which == "S" else "end", [((), (), 1.0)])
        durative = self._durative(key)
        folded: Dict[FrozenSet, float] = {}
        for adds, dels, p in branches:
            nxt = (set(facts) - set(dels)) | set(adds)
            if durative:
                if which == "S":
                    nxt.add(in_execution(key))
                else:
                    nxt.discard(in_execution(key))
            f = frozenset(nxt)
            folded[f] = folded.get(f, 0.0) + float(p)
        return tuple(folded.items())

    def start_is_inert(self, key: str) -> bool:
        return self._durative(key) and not self._spec[key].get("start")

    def action_by_name(self, name: str):
        return name


# ---------------------------------------------------------------------------
# Symbolic states
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    """One running or completed action instance.

    ``serial`` keeps repeated instances of a type distinct, as section 8
    demands ("Repeated instances of one type are distinct"). Reusing a name
    would silently identify a retry with the attempt it replaced.
    """

    key: str
    serial: int

    @property
    def start_var(self) -> str:
        return f"S:{self.key}#{self.serial}"

    @property
    def end_var(self) -> str:
        return f"E:{self.key}#{self.serial}"


@dataclass
class Node:
    """The spec's ``X``: an annotated STN plus its replayed semantic state."""

    facts: FrozenSet                       # cached REPLAY result (section 2)
    running: Tuple[Instance, ...]          # R, derived
    clock: str                             # C, a VARIABLE not a timestamp
    z: StrictDBM                           # the annotated network
    serials: Tuple[Tuple[str, int], ...]   # next serial per action type
    history: Tuple                         # the annotation: events + outcomes
    goal_reached: bool = False

    def serial_for(self, key: str) -> int:
        for k, n in self.serials:
            if k == key:
                return n
        return 0


def _bump_serial(serials: Tuple[Tuple[str, int], ...], key: str) -> Tuple[Tuple[str, int], ...]:
    """Next instance number for one action type (section 8: retries are distinct)."""
    out = dict(serials)
    out[key] = out.get(key, 0) + 1
    return tuple(sorted(out.items()))


@dataclass
class SolveResult:
    """What a solve returns. ``lo == hi`` is the only claim of exactness."""

    lo: float
    hi: float
    complete: bool
    nodes_expanded: int
    unknown_leaves: int
    branchwise_gap_witnessed: bool = False
    order_ambiguous_groups: int = 0
    lost_probability_mass: float = 0.0
    starts_pinned: bool = False

    @property
    def exact(self) -> bool:
        """True only when the graph was fully expanded AND no branchwise
        scheduling conflict was witnessed. Both are necessary: the first rules
        out UNKNOWN leaves, the second rules out section 6's relaxation."""
        return (
            self.complete
            and not self.branchwise_gap_witnessed
            and self.order_ambiguous_groups == 0
        )

    def __repr__(self) -> str:
        if self.exact:
            tag = "exact"
        elif self.complete:
            tag = "lower-bound" if self.starts_pinned else "upper-bound"
        else:
            tag = "cut"
        return (
            f"SolveResult({self.lo:.6f}, {self.hi:.6f}, {tag}, "
            f"nodes={self.nodes_expanded}, unknown={self.unknown_leaves})"
        )


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class TemporalSTNPDB:
    """Builds and backs up the symbolic graph of sections 4-6.

    One instance per query. ``node_budget`` limits computation only: an
    exhausted budget produces UNKNOWN = ``(0, 1)`` leaves, never failure
    (section 5: "Budget limits computation, not the domain").
    """

    ORIGIN = "O"
    DEADLINE = "D"

    def __init__(
        self,
        model,
        *,
        node_budget: int = DEFAULT_NODE_BUDGET,
        max_prefix_events: int = DEFAULT_MAX_PREFIX_EVENTS,
        pin_starts_to_clock: bool = False,
        prune_inert_unfinishable: bool = True,
        allow_simultaneous_start_with_end: bool = False,
        eager_placement: bool = False,
        merge_equal_states: bool = False,
    ):
        self.model = model
        self.node_budget = max(1, int(node_budget))
        # Positive durations bound the number of starts (each instance occupies
        # d(a) and starts are capped at D), but a ZERO-duration action has no
        # such bound: it can fire again at the same instant forever. Section 5
        # demands "a proved bound on starts/events" and offers none for that
        # case, so an over-long prefix becomes UNKNOWN rather than looping.
        self.max_prefix_events = max(1, int(max_prefix_events))
        # See the module header, "Bracketing the value". False = section 4 as
        # written (S symbolic in [C, next end)); True = start only AT an event
        # boundary, which pins every time to a point and makes the backup exact
        # for that restricted policy class.
        self.pin_starts_to_clock = bool(pin_starts_to_clock)
        # Sound dominance pruning, see EngineModel.start_is_inert. On by default
        # because it removes only dominated instances; set False to measure it.
        self.prune_inert_unfinishable = bool(prune_inert_unfinishable)
        self.allow_tie_start_end = bool(allow_simultaneous_start_with_end)
        self.eager_placement = bool(eager_placement)
        # Part I is a TREE: section 5 says "One prefix has one parent before
        # merging". Merging equal (facts, running, Z) keys is section 10 and
        # needs its sufficiency proof, so it is OFF unless asked for.
        self.merge_equal_states = bool(merge_equal_states)
        self.nodes_expanded = 0
        self.unknown_leaves = 0
        self.branchwise_gap_witnessed = False
        self.order_ambiguous_groups = 0
        self._memo: Dict[Tuple, Tuple[float, float, bool]] = {}

    # -- root ------------------------------------------------------------
    def root(
        self,
        facts: Iterable[Fact],
        horizon,
        running: Sequence[Tuple[str, object]] = (),
    ) -> Node:
        """Section 8's query projection, normalised so the query sits at time 0.

        ``running`` is ``[(action_type_key, remaining_time)]`` -- the in-flight
        instances and how long each still has to run, i.e. ``E_r = t + d(r)``
        expressed relative to the query instant. ``remaining_time`` may be
        ``None``, which leaves ``E_r`` symbolic in ``(0, d(r)]``; that is a
        RELAXATION (it hands the solver a choice the real execution has already
        made) and it is what makes ``hi`` an upper bound rather than the value.
        """
        z = StrictDBM(self.ORIGIN)
        z.add_eq(self.DEADLINE, self.ORIGIN, Fraction(horizon))
        clock = "C0"
        z.add_eq(clock, self.ORIGIN, 0)

        ops = self.model.ops()
        instances: List[Instance] = []
        serials: Dict[str, int] = {}
        for key, remaining in running:
            op = ops.get(key)
            if op is None:
                continue
            serial = serials.get(key, 0)
            serials[key] = serial + 1
            inst = Instance(key, serial)
            instances.append(inst)
            # The instance started BEFORE the query instant, so S_r is negative
            # in query-relative time. Its duration equality always holds.
            z.add_eq(inst.end_var, inst.start_var, op.duration)
            if remaining is None:
                # Unobserved end time: E_r is somewhere in (C0, C0 + d(r)], and
                # S_r follows from the duration. This is the relaxation flagged
                # in the docstring -- the real execution already fixed E_r.
                z.add_lt(clock, inst.end_var, 0)
                z.add_le(inst.end_var, self.ORIGIN, op.duration)
            else:
                z.add_eq(inst.end_var, self.ORIGIN, Fraction(remaining))
        return Node(
            facts=frozenset(facts),
            running=tuple(instances),
            clock=clock,
            z=z,
            serials=tuple(sorted(serials.items())),
            history=(),
            goal_reached=False,
        )

    # -- driver ----------------------------------------------------------
    def solve(self, node: Node) -> SolveResult:
        self.nodes_expanded = 0
        self.unknown_leaves = 0
        self.branchwise_gap_witnessed = False
        self.order_ambiguous_groups = 0
        self._memo.clear()
        lo, hi, complete = self._value(node)
        return SolveResult(
            lo=lo,
            hi=hi,
            complete=complete,
            nodes_expanded=self.nodes_expanded,
            unknown_leaves=self.unknown_leaves,
            branchwise_gap_witnessed=self.branchwise_gap_witnessed,
            order_ambiguous_groups=self.order_ambiguous_groups,
            starts_pinned=self.pin_starts_to_clock,
            lost_probability_mass=getattr(self.model, "lost_probability_mass", 0.0),
        )

    # -- value -----------------------------------------------------------
    def _value(self, node: Node) -> Tuple[float, float, bool]:
        if node.goal_reached:
            return (1.0, 1.0, True)
        # A prefix whose last processed event is forced past D can never collect
        # the reward: every later event is later still.
        if not self._clock_can_reach_deadline(node):
            return (0.0, 0.0, True)
        if self.nodes_expanded >= self.node_budget or len(node.history) >= self.max_prefix_events:
            # Section 5: UNKNOWN, and explicitly NOT value zero.
            self.unknown_leaves += 1
            return (0.0, 1.0, False)

        key = self._memo_key(node) if self.merge_equal_states else None
        if key is not None:
            hit = self._memo.get(key)
            if hit is not None:
                return hit

        self.nodes_expanded += 1
        choices = list(self._choices(node))
        if not choices:
            # No legal continuation, goal not reached: terminal failure.
            return (0.0, 0.0, True)

        if not self._region_is_a_point(node):
            # Section 6: this node covers more than one ground state, so a single
            # scalar max over it is a relaxation rather than the backup.
            self.branchwise_gap_witnessed = True

        best_lo, best_hi = 0.0, 0.0        # stopping is always available (0)
        complete = True
        for _label, successors in choices:
            u_lo = u_hi = 0.0
            u_complete = True
            for prob, child in successors:
                c_lo, c_hi, c_complete = self._value(child)
                u_lo += prob * c_lo
                u_hi += prob * c_hi
                u_complete = u_complete and c_complete
            complete = complete and u_complete
            if u_lo > best_lo:
                best_lo = u_lo
            if u_hi > best_hi:
                best_hi = u_hi

        out = (best_lo, best_hi, complete)
        if key is not None and complete:
            self._memo[key] = out
        return out

    def _memo_key(self, node: Node) -> Tuple:
        """Section 10's merge key -- PART II, and UNPROVED. Off by default.

        ``F = {O, D, C} + retained running start/end events`` (section 10),
        projected out of the closed network and renamed by ROLE rather than by
        the variable names history happened to hand out. Without the renaming
        the key never fires: starting ``a`` then ``b`` and starting ``b`` then
        ``a`` reach the same semantic state with the same feasible timings, but
        ``C`` points at ``S:a#0`` in one and ``S:b#0`` in the other, so the raw
        canonical forms differ and every interleaving is re-solved from scratch.

        Section 10 demands more than this before the merge is legitimate:
        "Merge only if keys define the same future control problem" and
        "compressed-state sufficiency for conditions, probabilities, resources
        and timing" is listed in section 11 as an outstanding proof obligation.
        This key is a MEASUREMENT instrument, not a discharged proof.
        """
        frontier = [self.ORIGIN, self.DEADLINE, node.clock]
        roles = {self.ORIGIN: "O", self.DEADLINE: "D", node.clock: "C"}
        for slot, inst in enumerate(sorted(node.running, key=lambda i: (i.key, i.serial))):
            for var, tag in ((inst.start_var, "S"), (inst.end_var, "E")):
                if var in node.z:
                    frontier.append(var)
                    roles.setdefault(var, f"{tag}{slot}:{inst.key}")
        projected = node.z.project(frontier)
        order = [v for v in frontier if v in projected]
        canonical = projected.canonical(order)
        return (
            node.facts,
            tuple(sorted((i.key, i.serial) for i in node.running)),
            tuple(roles[v] for v in order),
            canonical[1],
        )

    def _start_is_inert(self, key: str) -> bool:
        getter = getattr(self.model, "start_is_inert", None)
        return bool(getter(key)) if getter is not None else False

    def _clock_can_reach_deadline(self, node: Node) -> bool:
        """Can the last processed event still sit at or before D?

        If not, every later event is later still, so no reward is reachable and
        the prefix is worth exactly 0 -- a sound prune, not a budget cut.
        """
        probe = node.z.copy()
        return probe.add_le(node.clock, self.DEADLINE, 0)

    # -- section 6's shared-commitment obligation --------------------------
    def _region_is_a_point(self, node: Node) -> bool:
        """Is every event time in this node pinned to a single rational?

        This is the CERTIFICATE for section 6. The backup here takes one max per
        node, but a node stands for a whole SET of ground states -- one per
        feasible time assignment in its region. When the region holds more than
        one point, two outcome branches of a decision may each rely on a
        different value of a time that was committed BEFORE the branch, which is
        exactly the ``sup_eta Sum_omega`` vs ``Sum_omega sup_eta`` gap that
        section 6 refuses to call exact.

        When every region IS a point there is no such freedom, the per-branch
        max is the ordinary MDP backup, and the value is exact. Checking that is
        cheap and SOUND in the useful direction: it never certifies a run that
        had freedom. Detecting the converse -- that available freedom actually
        changed the answer -- needs the parameterised value function section 6
        leaves open, so it is not attempted.
        """
        z = node.z
        for var in z.variables:
            if var == self.ORIGIN:
                continue
            lo, lo_strict, hi, hi_strict = z.window(var)
            if lo is None or hi is None or lo != hi or lo_strict or hi_strict:
                return False
        return True

    # -- section 4: legal event transitions -------------------------------
    def _choices(self, node: Node):
        """Yield ``(label, [(prob, child)])`` for every legal event choice."""
        for choice in self._end_choices(node):
            yield choice
        for choice in self._start_choices(node):
            yield choice

    # -- END --------------------------------------------------------------
    def _end_choices(self, node: Node):
        """Section 4 END: process one pending completion that can be earliest.

        Singleton, not a subset. Section 4 offers "a consistent earliest pending
        completion or legal tied group", and in this codebase's ``regular``
        encoding the planner applies exactly ONE snap action per step -- a tied
        group is not something the model can do atomically. So the tied group is
        generated the way the engine would actually execute it: END{a} with
        ``E_a <= E_r`` (non-strict, ties admitted), then END{b} at the same
        instant, with the processing ORDER an explicit planner choice. That also
        removes the ``|L|!`` permutation blowup and the false order-ambiguity it
        reported, because there is no longer a group whose internal order the
        model must resolve for us.

        Enumerating which end goes next IS section 3's placement enumeration:
        ``E_a < E_r`` is END{a} first, ``E_a = E_r`` is the tie reached by two
        successive singletons, ``E_a > E_r`` is END{r} first.
        """
        running = node.running
        for inst in running:
            z = node.z.copy()
            ok = z.add_le(node.clock, inst.end_var, 0)
            for r in running:
                if r is inst:
                    continue
                ok = z.add_le(inst.end_var, r.end_var, 0) and ok
            if not ok or not z.consistent:
                continue
            orders = self._process_events(
                node, z, [(inst, "E")], new_clock=inst.end_var
            )
            if orders is None:
                continue
            for successors in orders:
                yield (("END", inst.key, inst.serial), successors)

    # -- START ------------------------------------------------------------
    def _start_choices(self, node: Node):
        """Section 4 START: one action, started at or after C, before the next
        pending completion.

        A start LATER than a pending completion is deliberately not generated
        here: section 4 says such a start "is reached by processing that
        completion and its outcome first; the later action choice may then
        depend on that observation". Generating it directly would let the
        planner commit to a start time using information it has not observed.
        """
        ops = self.model.ops()
        legal = self.model.legal_action_names(node.facts)
        for name in legal:
            op = ops.get(name)
            if op is None:
                continue
            serial = node.serial_for(name)
            inst = Instance(name, serial)
            for placement, z in enumerate(self._placements(node, inst, op)):
                orders = self._process_events(
                    node,
                    z,
                    [(inst, "S")],
                    new_clock=inst.start_var,
                    started=inst,
                    op=op,
                )
                if orders is None:
                    continue
                for successors in orders:
                    yield (("START", name, serial, placement), successors)

    def _placements(self, node: Node, inst: Instance, op: DurativeOp):
        """Temporal placements of a new instance's two endpoints.

        Base constraints are section 2's: ``E_a - S_a = d(a)``, ``C <= S_a``.
        ``E_a <= D`` is NOT imposed -- section 2 forbids assuming completion by
        D -- but ``S_a <= D`` is, for the reason given in this module's header.
        """
        base = node.z.copy()
        ok = base.add_eq(inst.end_var, inst.start_var, op.duration)
        if self.pin_starts_to_clock:
            ok = base.add_eq(inst.start_var, node.clock, 0) and ok
        else:
            ok = base.add_le(node.clock, inst.start_var, 0) and ok
        if self.prune_inert_unfinishable and self._start_is_inert(inst.key):
            # This instance can only ever pay off at its END, so an end past D
            # makes the whole instance dominated by not starting it. Sound, and
            # it is what stops a long action from being started (and then
            # interleaved with everything else) at deadlines it cannot reach.
            ok = base.add_le(inst.end_var, self.DEADLINE, 0) and ok
        else:
            ok = base.add_le(inst.start_var, self.DEADLINE, 0) and ok
        for r in node.running:
            if self.allow_tie_start_end:
                ok = base.add_le(inst.start_var, r.end_var, 0) and ok
            else:
                ok = base.add_lt(inst.start_var, r.end_var, 0) and ok
        if not ok or not base.consistent:
            return
        if not self.eager_placement or not node.running:
            yield base
            return
        # Section-3-literal mode: split now on E_a vs each pending E_r.
        relations = ("lt", "eq", "gt")
        for combo in itertools.product(relations, repeat=len(node.running)):
            z = base.copy()
            good = True
            for r, rel in zip(node.running, combo):
                if rel == "lt":
                    good = z.add_lt(inst.end_var, r.end_var, 0) and good
                elif rel == "eq":
                    good = z.add_eq(inst.end_var, r.end_var, 0) and good
                else:
                    good = z.add_lt(r.end_var, inst.end_var, 0) and good
                if not good:
                    break
            if good and z.consistent:
                yield z

    # -- applying events ---------------------------------------------------
    def _process_events(
        self,
        node: Node,
        z: StrictDBM,
        events: Sequence[Tuple[Instance, str]],
        *,
        new_clock: str,
        started: Optional[Instance] = None,
        op: Optional[DurativeOp] = None,
    ):
        """Apply a group of simultaneous events, branching on their outcomes.

        Returns ``[(prob, child)]`` or ``None`` when the group is illegal.

        Non-commuting simultaneous effects: section 2 requires the orderings to
        be split rather than silently resolved, so a tied group whose members do
        not commute yields one branch set per processing order. Commuting groups
        collapse to a single order, which is the common case and costs nothing.
        """
        results = []
        for order in itertools.permutations(events):
            branches = self._apply_order(node, z, order, new_clock, started, op)
            if branches is None:
                continue
            merged: Dict[FrozenSet, Tuple[float, Node]] = {}
            for prob, child in branches:
                prev = merged.get(child.facts)
                if prev is None:
                    merged[child.facts] = (prob, child)
                else:
                    merged[child.facts] = (prev[0] + prob, prev[1])
            results.append([(p, c) for p, c in merged.values()])
            if len(events) <= 1:
                break
        if not results:
            return None
        if len(results) > 1:
            # Section 2: "If two represented orderings give different facts
            # because effects do not commute, split them or retain separate
            # semantic branches." Distinct outcome distributions mean the tied
            # group is order-ambiguous in the model; that is recorded rather
            # than resolved by picking one.
            signatures = {
                tuple(sorted((c.facts, round(p, 12)) for p, c in branches))
                for branches in results
            }
            if len(signatures) > 1:
                self.order_ambiguous_groups += 1
        return results

    def _apply_order(
        self,
        node: Node,
        z: StrictDBM,
        order: Sequence[Tuple[Instance, str]],
        new_clock: str,
        started: Optional[Instance],
        op: Optional[DurativeOp],
    ):
        """Apply one processing order, expanding the outcome cross-product."""
        ops = self.model.ops()
        frontier: List[Tuple[float, FrozenSet]] = [(1.0, node.facts)]
        for inst, which in order:
            model_op = ops.get(inst.key)
            if model_op is None:
                return None
            action = self._event_action(model_op, which)
            nxt: Dict[FrozenSet, float] = {}
            for prob, facts in frontier:
                if not self._event_legal(model_op, which, facts):
                    return None
                for out_facts, p in self.model.outcomes(action, facts):
                    if p <= _EPS:
                        continue
                    nxt[out_facts] = nxt.get(out_facts, 0.0) + prob * p
            if not nxt:
                return None
            frontier = [(p, f) for f, p in nxt.items()]

        running = list(node.running)
        serials = node.serials
        for inst, which in order:
            if which == "E":
                running = [r for r in running if r != inst]
        if started is not None and op is not None:
            if op.end_action is not None:
                running.append(started)
            serials = _bump_serial(serials, started.key)

        goals = self.model.goals
        annotation = tuple((i.key, i.serial, w) for i, w in order)
        out: List[Tuple[float, Node]] = []
        for prob, facts in frontier:
            reached = goals.issubset(facts)
            child_z = z
            if reached:
                # Section 6: the reward is collected at a permitted event by D,
                # so claiming it requires this event to fit before the deadline.
                probe = z.copy()
                if not probe.add_le(new_clock, self.DEADLINE, 0):
                    reached = False
                else:
                    child_z = probe
            out.append((
                prob,
                Node(
                    facts=facts,
                    running=tuple(running),
                    clock=new_clock,
                    z=child_z,
                    serials=serials,
                    history=node.history + (annotation + (facts,),),
                    goal_reached=reached,
                ),
            ))
        return out

    @staticmethod
    def _event_action(op: DurativeOp, which: str):
        if isinstance(op.start_action, str):        # ToyModel
            return (op.key, which)
        return op.start_action if which == "S" else op.end_action

    def _event_legal(self, op: DurativeOp, which: str, facts: FrozenSet) -> bool:
        """Section 4's condition check, at the event that is actually firing."""
        if isinstance(op.start_action, str):        # ToyModel
            if which == "S":
                return op.key in self.model.legal_action_names(facts)
            return self.model.end_legal(op.key, facts)
        action = self._event_action(op, which)
        if action is None:
            return False
        if not action.pos_preconditions.issubset(facts):
            return False
        return action.neg_preconditions.isdisjoint(facts)


# ---------------------------------------------------------------------------
# Heuristic entry point
# ---------------------------------------------------------------------------

def _extract_state_facts(state) -> Set[Fact]:
    preds = getattr(state, "predicates", None)
    if preds is None and isinstance(state, (set, frozenset)):
        return set(state)
    return set(preds or set())


class TemporalSTNPDBHeuristic:
    """``temporal_stn_pdb``: section 7/8 lookup over the Part I graph.

    ``phi`` is the IDENTITY here. Section 7 demands a proved Markov abstraction
    ("phi(q1) = phi(q2) => same legal controls and same abstract transition
    law", and "Do not free omitted preconditions by default"), and no pattern
    builder in this repository supplies that proof -- ``build_pattern`` frees
    exactly the preconditions section 7 forbids freeing. So the naive variant
    keeps the full semantic state and the "smaller space" comes only from
    symbolic time, not from a fact projection.
    """

    def __init__(self, mdp, node_budget: Optional[int] = None):
        self._model = EngineModel(mdp)
        self._budget = int(
            node_budget
            if node_budget is not None
            else _env_int("TP_MCTS_STN_PDB_NODE_BUDGET", DEFAULT_NODE_BUDGET)
        )
        self._max_prefix_events = _env_int(
            "TP_MCTS_STN_PDB_MAX_PREFIX_EVENTS", DEFAULT_MAX_PREFIX_EVENTS
        )
        self._merge = _env_flag("TP_MCTS_STN_PDB_MERGE", False)
        self._tie_starts = _env_flag("TP_MCTS_STN_PDB_TIE_STARTS", False)
        self._pin_starts = _env_flag("TP_MCTS_STN_PDB_PIN_STARTS", False)
        self._cache: Dict[Tuple, SolveResult] = {}
        self.last_result: Optional[SolveResult] = None
        self.queries = 0
        self.incomplete_queries = 0

    @classmethod
    def from_mdp(cls, mdp) -> "TemporalSTNPDBHeuristic":
        return cls(mdp)

    # -- section 8 ---------------------------------------------------------
    def project_query(self, state, running_remaining: Optional[Dict[str, object]] = None):
        """``key(q) = (phi(S), running signature)`` plus the entry interface.

        Running instances are recovered from the ``inExecution`` facts the
        compilation writes at every start -- that is where this codebase keeps
        R. Their end times must come from the search's STN; when they are not
        supplied each ``E_r`` stays symbolic, which is a relaxation and is
        reported through ``complete``/``exact`` rather than hidden.
        """
        facts = frozenset(_extract_state_facts(state))
        running: List[Tuple[str, object]] = []
        for key in sorted(self._in_execution_keys(facts)):
            remaining = None if running_remaining is None else running_remaining.get(key)
            running.append((key, remaining))
        return facts, tuple(running)

    def _in_execution_keys(self, facts: FrozenSet) -> List[str]:
        ops = self._model.ops()
        out = []
        for fact in facts:
            text = str(fact)
            if "inExecution" not in text:
                continue
            for key in ops:
                # The compilation names the object "start-<action>" and the snap
                # action "start_<action>", so match on the shared action name.
                base = key[len("start_"):] if key.startswith("start_") else key
                if f"start-{base}" in text:
                    out.append(key)
                    break
        return out

    # -- scoring -----------------------------------------------------------
    def heuristic_score(
        self,
        state,
        goal_facts: Iterable[Fact] = (),
        fixed_depth: int = 25,
        start_time: float = 0.0,
        running_remaining: Optional[Dict[str, object]] = None,
        **_ignored,
    ) -> float:
        """Upper end of the value interval, for use as an MCTS leaf value.

        Returns ``hi``. When ``self.last_result.exact`` is True this IS the
        value; otherwise it is an upper bound and the gap is visible in
        ``last_result``.
        """
        result = self.evaluate(state, fixed_depth, running_remaining)
        return result.hi

    def evaluate(
        self,
        state,
        horizon,
        running_remaining: Optional[Dict[str, object]] = None,
    ) -> SolveResult:
        facts, running = self.project_query(state, running_remaining)
        horizon = max(0, int(horizon))
        key = (facts, running, horizon, self._pin_starts)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.queries += 1
        solver = TemporalSTNPDB(
            self._model,
            node_budget=self._budget,
            max_prefix_events=self._max_prefix_events,
            pin_starts_to_clock=self._pin_starts,
            merge_equal_states=self._merge,
            allow_simultaneous_start_with_end=self._tie_starts,
        )
        result = solver.solve(solver.root(facts, horizon, running))
        if not result.complete:
            self.incomplete_queries += 1
        self._cache[key] = result
        self.last_result = result
        return result
