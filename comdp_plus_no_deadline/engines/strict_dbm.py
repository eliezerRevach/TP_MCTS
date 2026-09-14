"""Strict-bound rational DBM for the symbolic-STN temporal PDB.

Part I of ``artifacts/Temporal_STN_PDB_Prototype.docx`` keeps every future
execution time SYMBOLIC and branches only on event ORDER. That needs a
constraint store able to say ``S2 < E1`` and ``S2 = E1`` as *different*
constraints, which is exactly what the spec demands:

    "Use strict-bound DBMs for <; do not replace strictness by an arbitrary
     epsilon."                                           (spec, section 3)

An epsilon encoding is not merely inelegant here: with rational durations the
smallest meaningful gap is unbounded below, so any fixed epsilon either rejects
a feasible order or admits an infeasible one. So a bound is a PAIR
``(value, strict)`` standing for

    x - y <= value        (strict=False)
    x - y <  value        (strict=True)

and the two compose under the usual ordered-semiring rules: bounds ADD along a
path (strictness is contagious -- one strict edge makes the whole path strict)
and the TIGHTER of two parallel bounds wins (equal values: strict beats
non-strict).

``Fraction`` is the value type, not ``float``: the spec's model is positive
RATIONAL durations, and a closure over floats would turn an exact ``E1 = E2``
tie -- a feasible placement the spec explicitly requires us to keep (section 3:
"Also include S2 = E1 < E2 and S2 < E2 = E1") -- into a coin flip on rounding.

This module is deliberately free of any planning concept. It knows nothing about
actions, facts, outcomes, patterns or probabilities; it is the temporal kernel
the rest of the prototype sits on.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

Var = Hashable

# A bound on ``x - y``. A ``None`` value means +infinity (no constraint);
# ``strict`` is meaningless there and is stored as False.
Bound = Tuple[Optional[Fraction], bool]

INF: Bound = (None, False)
ZERO: Bound = (Fraction(0), False)


# ---------------------------------------------------------------------------
# Bound arithmetic
# ---------------------------------------------------------------------------

def bound_add(a: Bound, b: Bound) -> Bound:
    """Compose along a path: ``x-y <= a`` and ``y-z <= b`` give ``x-z``.

    Strictness is contagious -- a path through one strict edge is strict.
    """
    av, astrict = a
    bv, bstrict = b
    if av is None or bv is None:
        return INF
    return (av + bv, astrict or bstrict)


def bound_tighter(a: Bound, b: Bound) -> bool:
    """True when ``a`` constrains strictly more than ``b``.

    At equal value the STRICT bound is tighter: ``x-y < 3`` rules out the
    assignment ``x-y = 3`` that ``x-y <= 3`` admits.
    """
    av, astrict = a
    bv, bstrict = b
    if av is None:
        return False
    if bv is None:
        return True
    if av != bv:
        return av < bv
    return astrict and not bstrict


def bound_min(a: Bound, b: Bound) -> Bound:
    return a if bound_tighter(a, b) else b


def bound_is_negative(b: Bound) -> bool:
    """True when ``t - t <= b`` is unsatisfiable (a negative cycle).

    ``t-t <= 0`` is fine, ``t-t < 0`` is not: a zero-valued STRICT self-bound is
    as inconsistent as a negative one. Missing that case is the classic way a
    strict DBM silently accepts ``a < b < a``.
    """
    v, strict = b
    if v is None:
        return False
    if v < 0:
        return True
    return v == 0 and strict


def bound_repr(b: Bound) -> str:
    v, strict = b
    if v is None:
        return "inf"
    return ("<" if strict else "<=") + str(v)


# ---------------------------------------------------------------------------
# The DBM
# ---------------------------------------------------------------------------

class StrictDBM:
    """Difference-bound matrix over named time points, with strict bounds.

    ``m[i][j]`` bounds ``t_i - t_j``. The ``origin`` variable (the spec's ``O``)
    is the reference point: ``O = 0``, so ``m[i][origin]`` is an upper bound on
    ``t_i`` and ``m[origin][i]`` is the negation of a lower bound.

    The matrix is kept CLOSED (all-pairs shortest path) after every mutation, so
    :attr:`consistent` and :meth:`bound` are exact rather than "not yet
    refuted", and :meth:`project` can simply read off rows. Closure is
    incremental for a single added edge (O(N^2), tighten-around-the-new-edge)
    and full Floyd-Warshall only on demand -- the spec's O(N^3) figure.
    """

    __slots__ = ("_index", "_vars", "_m", "_consistent")

    def __init__(self, origin: Var = "O"):
        self._index: Dict[Var, int] = {}
        self._vars: List[Var] = []
        self._m: List[List[Bound]] = []
        self._consistent = True
        self.add_var(origin)

    # -- structure ------------------------------------------------------
    @property
    def origin(self) -> Var:
        return self._vars[0]

    @property
    def variables(self) -> Tuple[Var, ...]:
        return tuple(self._vars)

    def __contains__(self, v: Var) -> bool:
        return v in self._index

    def __len__(self) -> int:
        return len(self._vars)

    def add_var(self, v: Var) -> int:
        """Add an unconstrained time point (every relation to it is +inf)."""
        existing = self._index.get(v)
        if existing is not None:
            return existing
        i = len(self._vars)
        self._index[v] = i
        self._vars.append(v)
        for row in self._m:
            row.append(INF)
        new_row = [INF] * (i + 1)
        new_row[i] = ZERO
        self._m.append(new_row)
        return i

    def copy(self) -> "StrictDBM":
        out = StrictDBM.__new__(StrictDBM)
        out._index = dict(self._index)
        out._vars = list(self._vars)
        out._m = [row[:] for row in self._m]
        out._consistent = self._consistent
        return out

    # -- queries --------------------------------------------------------
    @property
    def consistent(self) -> bool:
        return self._consistent

    def bound(self, x: Var, y: Var) -> Bound:
        """Tightest implied bound on ``x - y`` (both must be declared)."""
        return self._m[self._index[x]][self._index[y]]

    def interval(self, x: Var) -> Tuple[Bound, Bound]:
        """``(lower, upper)`` on ``x`` relative to the origin, as raw bounds.

        ``lower`` bounds ``O - x`` (so ``x >= -lower``) and ``upper`` bounds
        ``x - O``. Returned unconverted so strictness is not lost.
        """
        o = self.origin
        return (self.bound(o, x), self.bound(x, o))

    def window(self, x: Var) -> Tuple[Optional[Fraction], bool, Optional[Fraction], bool]:
        """``(lo, lo_strict, hi, hi_strict)`` for ``x`` in origin-relative time."""
        (lo_b, lo_strict), (hi, hi_strict) = self.interval(x)
        lo = None if lo_b is None else -lo_b
        return (lo, lo_strict, hi, hi_strict)

    # -- mutation -------------------------------------------------------
    def add_bound(self, x: Var, y: Var, value, strict: bool = False) -> bool:
        """Assert ``x - y <= value`` (or ``<`` when strict); returns consistency.

        Closes incrementally. Once inconsistent the DBM stays inconsistent --
        the spec wants such a scheduling choice DROPPED, not repaired.
        """
        if not self._consistent:
            return False
        i = self.add_var(x)
        j = self.add_var(y)
        new: Bound = (Fraction(value), bool(strict))
        if not bound_tighter(new, self._m[i][j]):
            return True                      # subsumed: nothing to propagate
        self._m[i][j] = new
        if i == j and bound_is_negative(new):
            self._consistent = False
            return False
        return self._tighten_around(i, j)

    def add_le(self, x: Var, y: Var, value=0) -> bool:
        """``x <= y + value``."""
        return self.add_bound(x, y, value, strict=False)

    def add_lt(self, x: Var, y: Var, value=0) -> bool:
        """``x < y + value``. The spec's strict order, with no epsilon."""
        return self.add_bound(x, y, value, strict=True)

    def add_eq(self, x: Var, y: Var, value=0) -> bool:
        """``x - y == value``, as the two opposite bounds."""
        if not self.add_bound(x, y, value, strict=False):
            return False
        return self.add_bound(y, x, -Fraction(value), strict=False)

    def fix(self, x: Var, value) -> bool:
        """Pin ``x`` to an absolute time (the spec's query conditioning I = x)."""
        return self.add_eq(x, self.origin, value)

    def _tighten_around(self, i: int, j: int) -> bool:
        """Restore closure after tightening edge ``(i, j)``. O(N^2)."""
        m = self._m
        n = len(self._vars)
        e = m[i][j]
        for u in range(n):
            ui = m[u][i]
            if ui[0] is None:
                continue
            via = bound_add(ui, e)
            if via[0] is None:
                continue
            for v in range(n):
                jv = m[j][v]
                if jv[0] is None:
                    continue
                cand = bound_add(via, jv)
                if bound_tighter(cand, m[u][v]):
                    m[u][v] = cand
                    if u == v and bound_is_negative(cand):
                        self._consistent = False
                        return False
        return True

    def close(self) -> bool:
        """Full Floyd-Warshall closure. O(N^3), the spec's stated cost."""
        m = self._m
        n = len(self._vars)
        for k in range(n):
            mk = m[k]
            for i in range(n):
                ik = m[i][k]
                if ik[0] is None:
                    continue
                mi = m[i]
                for j in range(n):
                    kj = mk[j]
                    if kj[0] is None:
                        continue
                    cand = bound_add(ik, kj)
                    if bound_tighter(cand, mi[j]):
                        mi[j] = cand
        for i in range(n):
            if bound_is_negative(m[i][i]):
                self._consistent = False
                return False
        return self._consistent

    # -- Part II support ------------------------------------------------
    def project(self, keep: Iterable[Var]) -> "StrictDBM":
        """Existentially eliminate every variable outside ``keep``.

        Because the matrix is closed, dropping rows and columns IS the
        projection: every bound implied through an eliminated point has already
        been written onto the surviving pairs. This is the spec's
        ``Z_F = PROJECT(CLOSE(Z), F)`` (section 9), and it is exact rather than
        an approximation -- which is the whole reason the frontier is allowed to
        forget completed history.
        """
        keep_list = [self.origin] + [v for v in keep if v != self.origin and v in self._index]
        seen = set()
        ordered = [v for v in keep_list if not (v in seen or seen.add(v))]
        out = StrictDBM.__new__(StrictDBM)
        out._index = {v: i for i, v in enumerate(ordered)}
        out._vars = list(ordered)
        out._consistent = self._consistent
        src = [self._index[v] for v in ordered]
        out._m = [[self._m[a][b] for b in src] for a in src]
        return out

    def canonical(self, order: Optional[Sequence[Var]] = None) -> Tuple:
        """Hashable canonical form of the closed matrix, for merge keys.

        Two closed DBMs with the same canonical form under the same variable
        order describe the SAME set of feasible assignments -- that is what
        makes it safe as part of the spec's section-9 merge key.
        """
        names = list(order) if order is not None else sorted(self._vars, key=repr)
        names = [v for v in names if v in self._index]
        idx = [self._index[v] for v in names]
        rows = tuple(
            tuple(self._m[a][b] for b in idx)
            for a in idx
        )
        return (tuple(repr(v) for v in names), rows)

    def __repr__(self) -> str:
        if not self._consistent:
            return "StrictDBM(INCONSISTENT)"
        parts = []
        for v in self._vars[1:]:
            lo, lo_s, hi, hi_s = self.window(v)
            lo_txt = "-inf" if lo is None else str(lo)
            hi_txt = "inf" if hi is None else str(hi)
            parts.append(f"{v!r} in {'(' if lo_s else '['}{lo_txt},{hi_txt}{')' if hi_s else ']'}")
        return "StrictDBM(" + ", ".join(parts) + ")"
