"""Interval arithmetic and forward-mode derivatives, to bound f_b and its Jacobian for (12)."""
import numpy as np


class Iv:
    """Interval [lo, hi]."""
    __array_ufunc__ = None                     # numpy defers to the operators below

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def __getitem__(self, k):
        return Iv(self.lo[k], self.hi[k])

    mid = property(lambda s: 0.5 * (s.lo + s.hi))
    rad = property(lambda s: 0.5 * (s.hi - s.lo))

    def __add__(self, o):
        if isinstance(o, Dual):
            return NotImplemented
        o = iv(o)
        return Iv(self.lo + o.lo, self.hi + o.hi)

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __sub__(self, o):
        return NotImplemented if isinstance(o, Dual) else self + (-iv(o))

    def __rsub__(self, o):
        return iv(o) + (-self)

    def __mul__(self, o):
        if isinstance(o, Dual):
            return NotImplemented
        o = iv(o)
        p = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Iv(np.minimum.reduce(p), np.maximum.reduce(p))

    def __truediv__(self, o):                  # positive divisors only
        if isinstance(o, Dual):
            return NotImplemented
        o = iv(o)
        assert np.all(o.lo > 0.0)
        return self * Iv(1.0 / o.hi, 1.0 / o.lo)

    def __rtruediv__(self, o):
        return iv(o) / self

    __radd__, __rmul__ = __add__, __mul__

    def sum(self, axis):
        return Iv(self.lo.sum(axis), self.hi.sum(axis))


class Dual:
    """Value and its derivatives along six directions."""
    __array_ufunc__ = None

    def __init__(self, v, d):
        self.v, self.d = v, list(d)

    def __getitem__(self, k):
        return Dual(self.v[k], [e[k] for e in self.d])

    def __add__(self, o):
        o = dual(o)
        return Dual(self.v + o.v, [a + b for a, b in zip(self.d, o.d)])

    def __neg__(self):
        return Dual(-self.v, [-a for a in self.d])

    def __sub__(self, o):
        return self + (-dual(o))

    def __rsub__(self, o):
        return dual(o) + (-self)

    def __mul__(self, o):
        o = dual(o)
        return Dual(self.v * o.v, [self.v * b + o.v * a for a, b in zip(self.d, o.d)])

    def __truediv__(self, o):
        o = dual(o)
        q = self.v / o.v
        return Dual(q, [(a - q * b) / o.v for a, b in zip(self.d, o.d)])

    def __rtruediv__(self, o):
        return dual(o) / self

    __radd__, __rmul__ = __add__, __mul__


def iv(x):
    return x if isinstance(x, Iv) else Iv(x, x)


def dual(x):
    return x if isinstance(x, Dual) else Dual(x, [0.0] * 6)


def mono(f, x, df=None, dec=False):
    """Monotone f with derivative df, on floats, intervals and duals."""
    if isinstance(x, Dual):
        g = df(x.v)
        return Dual(mono(f, x.v, dec=dec), [g * a for a in x.d])
    if not isinstance(x, Iv):
        return f(x)
    a, b = f(x.lo), f(x.hi)
    return Iv(b, a) if dec else Iv(a, b)


def sq(x):
    if isinstance(x, Dual):
        return Dual(sq(x.v), [2.0 * x.v * a for a in x.d])
    if not isinstance(x, Iv):
        return x * x
    a, b = x.lo ** 2, x.hi ** 2
    return Iv(np.where((x.lo < 0) & (x.hi > 0), 0.0, np.minimum(a, b)), np.maximum(a, b))


def sin(x):
    if isinstance(x, Dual):
        return Dual(sin(x.v), [cos(x.v) * a for a in x.d])
    if not isinstance(x, Iv):
        return np.sin(x)
    a, b = np.sin(x.lo), np.sin(x.hi)
    hit = lambda c: np.ceil((x.lo - c) / (2 * np.pi)) * 2 * np.pi + c <= x.hi  # noqa: E731
    return Iv(np.where(hit(-np.pi / 2), -1.0, np.minimum(a, b)),
              np.where(hit(np.pi / 2), 1.0, np.maximum(a, b)))


def cos(x):
    if isinstance(x, Dual):
        return Dual(cos(x.v), [-sin(x.v) * a for a in x.d])
    return sin(x + np.pi / 2) if isinstance(x, Iv) else np.cos(x)


def stack(parts, axis=-1):
    """np.stack for floats, intervals and duals."""
    if any(isinstance(p, Dual) for p in parts):
        parts = [dual(p) for p in parts]
        return Dual(stack([p.v for p in parts], axis),
                    [stack([p.d[j] for p in parts], axis) for j in range(6)])
    if any(isinstance(p, Iv) for p in parts):
        parts = [iv(p) for p in parts]
        return Iv(np.stack(np.broadcast_arrays(*[p.lo for p in parts]), axis),
                  np.stack(np.broadcast_arrays(*[p.hi for p in parts]), axis))
    if np.ndim(parts[0]) == 0:                 # one state
        return np.array(parts, float)
    return np.stack(np.broadcast_arrays(*parts), axis)
