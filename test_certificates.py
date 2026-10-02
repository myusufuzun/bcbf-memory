"""python -m pytest test_certificates.py -q"""
import numpy as np
from scipy.optimize import minimize

from family import Family, place_links
from interval import Dual, Iv
from model import P_SQRT_INV, P, T, U_HI, U_LO, XI_STAR, h, norm, f_b, f_b_nb
from simulate import N, flow, qp

rng = np.random.default_rng(0)
X = XI_STAR + rng.normal(size=(40, 6)) * [0.3, 3, 0.2, 0.5, 1, 1]


def fd(f, x, eps=1e-6):
    return np.stack([(f(x + eps * e) - f(x - eps * e)) / (2 * eps) for e in np.eye(6)], -1)


def test_jacobian():
    """Jacobian of f_b."""
    for x in X[:10]:
        v, J = f_b(x, jac=True)
        assert np.allclose(J, fd(f_b, x), atol=1e-6)
        assert np.allclose(f_b_nb(x)[0], v, rtol=1e-13, atol=1e-13) and np.allclose(f_b_nb(x)[1], J, rtol=1e-13, atol=1e-13)


def test_intervals():
    """Interval bounds of f_b and its Jacobian."""
    r = 0.03 * np.sqrt(np.diag(np.linalg.inv(P)))
    for D in (np.eye(6), P_SQRT_INV):                        # derivatives along the columns of D
        seed = [np.broadcast_to(D[:, k], X.shape) for k in range(6)]
        v, J = f_b(Dual(Iv(X - r, X + r), seed), jac=True)
        for _ in range(25):
            Y = X + r * rng.uniform(-1, 1, X.shape)
            vy, Jy = f_b(Y, jac=True)
            dJ = np.stack([fd(lambda y: f_b(y, jac=True)[1], y) for y in Y]) @ D
            assert np.all((v.v.lo <= vy) & (vy <= v.v.hi)) and np.all((J.v.lo <= Jy) & (Jy <= J.v.hi))
            for k in range(6):
                assert np.all((J.d[k].lo - 1e-5 <= dJ[..., k]) & (dJ[..., k] <= J.d[k].hi + 1e-5))


def test_qp():
    """QP (19)."""
    for _ in range(200):
        A, kb, y = rng.normal(size=(5, 2)), rng.uniform(U_LO, U_HI), rng.uniform(U_LO - 3, U_HI + 3)
        b = A @ kb - rng.uniform(0, 3, 5)                  # k_b feasible, as in the filter
        left, u = qp(y, A, b)
        assert left
        ref = minimize(lambda w: np.sum((w - y) ** 2), kb, bounds=list(zip(U_LO, U_HI)),
                       constraints={"type": "ineq", "fun": lambda w: A @ w - b}, tol=1e-12).x
        assert np.sum((u - y) ** 2) <= np.sum((ref - y) ** 2) + 1e-7 and np.all(A @ u >= b - 1e-9)


def test_link():
    """A certified link maps each ball into the next and lies in C_S."""
    q = flow(np.array([0, 6, 0, 0, -0.5, 0]), round(T / 0.02), 0.02)[-1]
    fam = Family()
    links = place_links(flow(q, 3 * N, T / N), T / N, fam, N)
    assert links and fam.appended(links, T / N) is not None
    Xl, R = links[-1][0], links[-1][1]                  # the entrance link
    for i in range(0, len(R) - 1, 25):
        u = rng.normal(size=(64, 6))
        Y = Xl[i] + R[i] * rng.uniform(0, 1, (64, 1)) ** (1 / 6) * u / norm(u)[:, None]
        assert np.all(h(Y) >= 0)
        Z = flow(Y, 32, T / N / 32)[-1]
        assert np.all(norm(Z - Xl[i + 1]) <= R[i + 1] * (1 + 1e-6))
