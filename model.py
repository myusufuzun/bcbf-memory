"""Planar quadrotor, Section VI."""
import numpy as np
from numba import njit
from interval import Dual, Iv, mono, sq, sin, cos, stack

G, M_V, J_V = 9.81, 1.0, 0.25       # (22)
U_LO, U_HI = np.array([0.0, -12.0]), np.array([12.0, 12.0])   # U
K_SAT = 20.0                        # sharpness of sat in (23)
A_MIN = 0.1                         # psi(a) = a for a >= A_MIN
BACKUP = (0.16, 0.8, 0.0, 12.0)     # k_b: gamma_p, gamma_d, y_d, z_d
NOMINAL = (0.81, 1.8, 0.0, 0.0)     # k_p
T = 5.0                             # prediction horizon
XI_STAR = np.array([0.0, 12.0, 0.0, 0.0, 0.0, 0.0])   # hover equilibrium
R0 = np.sqrt(0.22)                  # S_0 = B(XI_STAR, R0)
P = np.array([[0.1262772990736155, 0, 0.3833074147152414, 0.1321800788082787, 0, 0.0204909742231402],
              [0, 0.1842744530308829, 0, 0, 0.1856861328660534, 0],
              [0.3833074147152414, 0, 5.137097482375775, 0.8406824418134173, 0, 0.2057914334783590],
              [0.1321800788082787, 0, 0.8406824418134173, 0.3152658344973051, 0, 0.05651191889632153],
              [0, 0.1856861328660534, 0, 0, 0.4642153328871146, 0],
              [0.0204909742231402, 0, 0.2057914334783590, 0.05651191889632153, 0, 0.1213819402569779]])
_w, _V = np.linalg.eigh(P)
P_SQRT = (_V * np.sqrt(_w)) @ _V.T   # P^(1/2)
P_SQRT_INV = np.linalg.inv(P_SQRT)


def norm(v):
    """||v|| = (v' P v)^(1/2)."""
    return np.linalg.norm(v @ P_SQRT.T, axis=-1)


# gains of (17) and (18)
alpha_1 = lambda s: 2.0 * s + s ** 3                        # noqa: E731
alpha_hat = lambda R, s: (10.0 * R * s + (R * s) ** 3) / R  # noqa: E731, alpha_b(R_a s)/R_a
rho = lambda s: 40.0 * s                                    # noqa: E731
KAPPA = 50.0                                                # sharpness of the smooth min and max in h_1


def h(X):
    """Safety function h_1, Section VI."""
    y, z = X[..., 0], X[..., 1]
    inner = -np.logaddexp(np.logaddexp(-KAPPA * (y + 0.5), -KAPPA * (0.5 - y)),
                          -KAPPA * (z + 2.0)) / KAPPA
    return (np.logaddexp(KAPPA * inner, KAPPA * (z - 10.0)) - np.log(2.0)) / KAPPA


def grad_h(X):
    """Gradient of h_1."""
    y, z = X[..., 0], X[..., 1]
    a = -KAPPA * np.stack([y + 0.5, 0.5 - y, z + 2.0], -1)
    w = np.exp(a - a.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)              # weights of the smooth minimum
    lse = np.logaddexp(np.logaddexp(a[..., 0], a[..., 1]), a[..., 2])
    p = 0.5 * (1.0 - np.tanh(0.5 * (KAPPA * (z - 10.0) + lse)))   # weight of the minimum
    out = np.zeros(X.shape)
    out[..., 0] = p * (w[..., 0] - w[..., 1])
    out[..., 1] = p * w[..., 2] + 1.0 - p
    return out


def psi(a):
    """psi and its slope, Section VI."""
    if isinstance(a, (Iv, Dual)):               # used only where a >= A_MIN
        return a, 1.0
    t = np.clip(2.0 * a / A_MIN - 1.0, 1e-9, 1.0 - 1e-9)
    s, r = np.exp(-1.0 / t), np.exp(-1.0 / (1.0 - t))
    lam = s / (s + r)
    dlam = (s / t ** 2 * r + s * r / (1.0 - t) ** 2) / (s + r) ** 2 * 2.0 / A_MIN
    return lam * a + (1.0 - lam) * A_MIN / 2, lam + dlam * (a - A_MIN / 2)


def sat(v, lo, hi):
    """Smooth saturation to [lo, hi] and its slope."""
    sig = lambda y: 0.5 * (1.0 + np.tanh(0.5 * y))                          # noqa: E731
    up = lambda s: mono(lambda t: sig(K_SAT * (t - lo)), s,                 # noqa: E731
                        lambda t: K_SAT * up(t) * (1.0 - up(t)))
    dn = lambda s: mono(lambda t: sig(K_SAT * (hi - t)), s,                 # noqa: E731
                        lambda t: -K_SAT * dn(t) * (1.0 - dn(t)), dec=True)
    w = mono(lambda t: t + np.logaddexp(0.0, K_SAT * (lo - t)) / K_SAT, v, up)
    out = mono(lambda t: t - np.logaddexp(0.0, K_SAT * (t - hi)) / K_SAT, w, dn)
    return out, dn(w) * up(v)


def control_law(X, gains, jac=False):
    """Thrust F and moment M, (23), and their gradients if jac."""
    gp, gd, yd, zd = gains
    y, z, th, vy, vz, om = (X[..., i] for i in range(6))
    ay = -gp * (y - yd) - gd * vy
    az, dpsi = psi(G - gp * (z - zd) - gd * vz)
    nn = sq(ay) + sq(az)
    n = mono(np.sqrt, nn, lambda t: 0.5 / mono(np.sqrt, t))
    F, dF = sat(M_V * n, 0.0, 12.0)
    thd, dthd = sat(mono(np.arctan, ay / az, lambda t: 1.0 / (1.0 + sq(t))), -np.pi / 4, np.pi / 4)
    M, dM = sat(J_V * (64.0 * (th - thd) + 16.0 * om), -12.0, 12.0)
    if not jac:
        return F, M
    fy, fz = dF * M_V * ay / n, dF * M_V * az * dpsi / n
    ty, tz = dthd * az / nn, -dthd * ay * dpsi / nn
    m = dM * J_V
    grad_F = [-gp * fy, -gp * fz, 0.0, -gd * fy, -gd * fz, 0.0]
    grad_M = [64 * gp * m * ty, 64 * gp * m * tz, 64 * m, 64 * gd * m * ty, 64 * gd * m * tz, 16 * m]
    return F, M, grad_F, grad_M


def f_b(X, jac=False):
    """Backup dynamics f_b, (4), and their Jacobian if jac."""
    out = control_law(X, BACKUP, jac)
    F, M, th = out[0], out[1], X[..., 2]
    s, c = sin(th), cos(th)
    v = stack([X[..., 3], X[..., 4], X[..., 5], F * s / M_V, F * c / M_V - G, -M / J_V])
    if not jac:
        return v
    gF, gM = out[2], out[3]
    e = lambda k: [1.0 if j == k else 0.0 for j in range(6)]  # noqa: E731
    J = stack([stack(e(3)), stack(e(4)), stack(e(5)),
               stack([s * gF[k] / M_V + (F * c / M_V if k == 2 else 0.0) for k in range(6)]),
               stack([c * gF[k] / M_V - (F * s / M_V if k == 2 else 0.0) for k in range(6)]),
               stack([-gM[k] / J_V for k in range(6)])], -2)
    return v, J


@njit(cache=True)
def _psi(a):
    t = min(max(2.0 * a / A_MIN - 1.0, 1e-9), 1.0 - 1e-9)
    s, r = np.exp(-1.0 / t), np.exp(-1.0 / (1.0 - t))
    lam = s / (s + r)
    dlam = (s / t ** 2 * r + s * r / (1.0 - t) ** 2) / (s + r) ** 2 * 2.0 / A_MIN
    return lam * a + (1.0 - lam) * A_MIN / 2, lam + dlam * (a - A_MIN / 2)


@njit(cache=True)
def _sat(v, lo, hi):
    w = v + np.logaddexp(0.0, K_SAT * (lo - v)) / K_SAT
    out = w - np.logaddexp(0.0, K_SAT * (w - hi)) / K_SAT
    return out, 0.25 * (1.0 + np.tanh(0.5 * K_SAT * (hi - w))) * (1.0 + np.tanh(0.5 * K_SAT * (v - lo)))


@njit(cache=True)
def f_b_nb(x):
    """f_b and its Jacobian for one state, compiled."""
    gp, gd, yd, zd = BACKUP
    th, om = x[2], x[5]
    ay = -gp * (x[0] - yd) - gd * x[3]
    az, dpsi = _psi(G - gp * (x[1] - zd) - gd * x[4])
    nn = ay * ay + az * az
    n = np.sqrt(nn)
    F, dF = _sat(M_V * n, 0.0, 12.0)
    thd, dthd = _sat(np.arctan(ay / az), -np.pi / 4, np.pi / 4)
    M, dM = _sat(J_V * (64.0 * (th - thd) + 16.0 * om), -12.0, 12.0)
    fy, fz, ty, tz, m = dF * M_V * ay / n, dF * M_V * az * dpsi / n, dthd * az / nn, -dthd * ay * dpsi / nn, dM * J_V
    gF = np.array([-gp * fy, -gp * fz, 0.0, -gd * fy, -gd * fz, 0.0])
    gM = np.array([64 * gp * m * ty, 64 * gp * m * tz, 64 * m, 64 * gd * m * ty, 64 * gd * m * tz, 16 * m])
    s, c = np.sin(th), np.cos(th)
    J = np.zeros((6, 6))
    J[0, 3] = J[1, 4] = J[2, 5] = 1.0
    J[3], J[4], J[5] = s * gF / M_V, c * gF / M_V, -gM / J_V
    J[3, 2] += F * c / M_V
    J[4, 2] -= F * s / M_V
    return np.array([x[3], x[4], om, F * s / M_V, F * c / M_V - G, -M / J_V]), J


def k_b(x):
    """Backup controller k_b."""
    return np.array(control_law(x, BACKUP))


def k_p(x):
    """Nominal controller k_p."""
    return np.array(control_law(x, NOMINAL))


def g(x):
    """g of (22)."""
    out = np.zeros((6, 2))
    out[3, 0], out[4, 0], out[5, 1] = np.sin(x[2]) / M_V, np.cos(x[2]) / M_V, -1.0 / J_V
    return out


def rk4(f, x, dt):
    """One RK4 step."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def plant(x, u, dt):
    """One RK4 step of (22), input held."""
    return rk4(lambda y: np.r_[y[3:6], 0.0, -G, 0.0] + g(y) @ u, x, dt)
