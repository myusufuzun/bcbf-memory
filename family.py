"""Ball family, Definition 3, and links along backup arcs, Theorem 2."""
import numpy as np
from interval import Dual, Iv
from model import A_MIN, BACKUP, G, P, P_SQRT, P_SQRT_INV, R0, T, XI_STAR, f_b, h, norm

BOX = np.sqrt(np.diag(np.linalg.inv(P)))                            # B(c, r) lies in c +- r BOX
SIG = np.sqrt(np.linalg.eigvalsh(np.linalg.inv(P)[:2, :2]).max())   # B(c, r) lies in C_S if h_1(c) >= SIG r
R_ETA = R0 * 2.0 ** (-np.arange(32) / 4)                            # candidates for R_a + eta_a
SHAVE = 1.0 - 1e-9


class Family:
    """Stored balls S_a = B(c_a, R_a) with successors p(a).  Index 0 is S_0."""

    def __init__(self):
        self.c, self.R, self.p = XI_STAR[None], np.array([R0]), np.array([-1])

    def dist(self, X):
        CX, Cc = X @ P_SQRT.T, self.c @ P_SQRT.T
        return np.sqrt(np.maximum((CX ** 2).sum(-1)[..., None] - 2 * CX @ Cc.T
                                  + (Cc ** 2).sum(-1), 0.0))

    def beta(self, X):
        """Ball functions beta_a(x)."""
        return (self.R ** 2 - self.dist(X) ** 2) / self.R

    def H(self, X):
        """Terminal envelope H_A(x)."""
        return self.beta(X).max(-1)

    def room(self, X):
        """max_a (R_a - ||x - c_a||) and its maximizer."""
        R = self.R - self.dist(X)
        return R.max(-1), R.argmax(-1)

    def appended(self, links, theta):
        """Append extension by the links, or None if one fails Definition 3."""
        new, entrance = Family(), None
        new.c, new.R, new.p = self.c, self.R, self.p
        for X, R, R_eta, successor in links:
            successor = entrance if successor is None else successor
            if not certified(X, R, R_eta, theta, new, successor):
                return None
            entrance = len(new.R)
            new.c, new.R = np.vstack([new.c, X]), np.r_[new.R, R]
            new.p = np.r_[new.p, np.arange(entrance + 1, entrance + len(R)), successor]
        return new


def boxes(X, r):
    """Boxes of B(x_i, r_i), and where a_z >= A_MIN on them."""
    gp, gd, _, zd = BACKUP
    lo, hi = X - r[:, None] * BOX, X + r[:, None] * BOX
    return lo, hi, G - gp * (hi[:, 1] - zd) - gd * hi[:, 4] >= A_MIN


def sup_norm(w):
    """Upper bound of ||w|| over an interval vector."""
    return norm(w.mid) + np.linalg.norm(w.rad @ np.abs(P_SQRT).T, axis=-1)


def speed(X, r, third=False, chunk=4096):
    """sup ||f_b|| over B(x_i, r_i), or of the third time derivative of phi_b if third."""
    out = np.full(len(X), np.inf)
    for i in range(0, len(X), chunk):
        lo, hi, ok = boxes(X[i:i + chunk], r[i:i + chunk])
        if not ok.any():
            continue
        w = Iv(lo[ok], hi[ok])
        if third:
            v, J = f_b(Dual(w, [np.broadcast_to(e, lo[ok].shape) for e in np.eye(6)]), jac=True)
            f, A = v.v, J.v
            w = (A * (A * f[:, None, :]).sum(-1)[:, None, :]).sum(-1)
            for j in range(6):
                w = w + (J.d[j] * f[:, None, :]).sum(-1) * f[:, j][:, None]
        else:
            w = f_b(w)
        out[i:i + chunk][ok] = sup_norm(w)
    return out


def sym_norm(A):
    """||(A + A')/2||_2."""
    return np.abs(np.linalg.eigvalsh(0.5 * (A + np.swapaxes(A, 1, 2)))).max(-1)


def spread(X, r, chunk=4096):
    """Growth of the logarithmic norm of the Jacobian of f_b over B(x_i, r_i)."""
    out = np.full(len(X), np.inf)
    for i in range(0, len(X), chunk):
        lo, hi, ok = boxes(X[i:i + chunk], r[i:i + chunk])
        if ok.any():
            seed = [np.broadcast_to(P_SQRT_INV[:, k], lo[ok].shape) for k in range(6)]
            J = f_b(Dual(Iv(lo[ok], hi[ok]), seed), jac=True)[1]
            m = [sym_norm(P_SQRT @ H.mid @ P_SQRT_INV) + sym_norm(np.abs(P_SQRT) @ H.rad @ np.abs(P_SQRT_INV))
                 for H in J.d]
            out[i:i + chunk][ok] = np.sqrt(np.square(m).sum(0))
    return out


def sweep(X, theta):
    """loc_i with phi_b(t, x_i) in B(x_i, loc_i) for t <= theta."""
    loc = 2.0 * theta * norm(f_b(X)) + 1e-9
    for _ in range(8):
        short = theta * speed(X, loc) >= loc
        loc = np.where(short, 2.0 * loc, loc)
    return np.where(theta * speed(X, loc) < loc, loc, np.inf)


def rates(X, R_eta, theta):
    """Per flow step: mu_a of (12) for each R_a + eta_a, loc, E_a of (11), arrival error delta."""
    v, J = f_b(X[:-1], jac=True)
    S = P_SQRT @ J @ P_SQRT_INV
    lam = np.linalg.eigvalsh(0.5 * (S + np.swapaxes(S, 1, 2)))[:, -1]
    loc = sweep(X[:-1], theta)
    d = X[1:] - X[:-1] - theta * v
    E = norm(d)
    delta = (norm(d - 0.5 * theta ** 2 * np.einsum("nij,nj->ni", J, v))
             + theta ** 3 / 6.0 * speed(X[:-1], loc, third=True))
    k = R_eta.shape[1]
    s = spread(np.repeat(X[:-1], k, 0), R_eta.ravel()).reshape(-1, k)
    return lam[:, None] + 0.5 * (R_eta + loc[:, None]) * s, loc, E, delta


def largest(R1, E, delta, d2, mu, loc, R_eta, cap, theta):
    """Largest R_a satisfying arrival and exposure (11) into radius R1."""
    a, c = 1.0 + 2.0 * mu * theta, R1 ** 2 - d2
    with np.errstate(all="ignore"):
        expo = np.where((a > 0) & (c > 0), (np.sqrt(E ** 2 + a * c) - E) / a, 0.0)
        R = SHAVE * np.minimum.reduce([(R_eta - loc) / np.maximum(1.0, np.exp(mu * theta)),
                                       (R1 - delta) * np.exp(-mu * theta), expo, cap + 0 * mu])
    return np.where(np.isfinite(mu) & np.isfinite(loc) & np.isfinite(delta), R, 0.0)


def certified(X, R, R_eta, theta, fam, successor):
    """Definition 3 for one link."""
    n, d = len(R) - 1, X[1:] - X[:-1]
    mu, loc, E, delta = rates(X, R_eta[:, None], theta)
    mu = mu[:, 0]
    with np.errstate(all="ignore"):
        steps = ((np.maximum(1.0, np.exp(mu * theta)) * R[:-1] < R_eta - loc)
                 & (np.exp(mu * theta) * R[:-1] + delta <= R[1:])                  # arrival
                 & (R[1:] ** 2 >= (1 + 2 * mu * theta) * R[:-1] ** 2               # exposure (11)
                    + norm(d) ** 2 + 2 * R[:-1] * E))
    gap = fam.R[successor] - norm(X[-1] - fam.c[successor]) - R[-1]
    # docks: S_a in S_p(a)
    return bool(steps.all() and (h(X) >= SIG * R).all() and R.min() > 0
                and gap >= 0 and 1 <= n and n * theta <= T)


def place_links(X, theta, fam, N):
    """Links along the backup arc X into the family, Theorem 2.  N flow steps per link."""
    n = len(X) - 1
    room, successor = fam.room(X)
    cap = np.maximum(h(X), 0.0) / SIG
    d2 = norm(X[1:] - X[:-1]) ** 2
    mu, loc, E, delta = rates(X, np.tile(R_ETA, (n, 1)), theta)
    W, R_eta, dock = np.zeros(n + 1), np.zeros(n), np.zeros(n + 1, bool)
    W[n], dock[n] = SHAVE * min(room[n], cap[n]) if room[n] > 0 else 0.0, True
    for i in range(n - 1, -1, -1):
        cont = largest(W[i + 1], E[i], delta[i], d2[i], mu[i], loc[i], R_ETA, cap[i], theta)
        k = int(np.argmax(cont))
        here = SHAVE * min(room[i], cap[i]) if room[i] > 0 and i > 0 else 0.0
        if here >= cont[k]:
            W[i], dock[i] = here, True
        else:
            W[i], R_eta[i] = cont[k], R_ETA[k]
    if dock[0] or W[0] <= 0:
        return None
    e = int(np.argmax(dock[1:])) + 1                 # first dock from X[0]
    cuts = list(range(0, e, N)) + [e]
    return [(X[a:b + 1], W[a:b + 1], R_eta[a:b], int(successor[e]) if b == e else None)
            for a, b in reversed(list(zip(cuts[:-1], cuts[1:])))]
