"""Closed loops of Section VI.

    python simulate.py [grown] [standard_T] [standard_1.5T] [standard_2T]
"""
import sys
from pathlib import Path

import numpy as np
from numba import njit
from family import Family, place_links
from model import (P, P_SQRT, R0, T, U_HI, U_LO, XI_STAR, alpha_1, alpha_hat, f_b, f_b_nb, g, grad_h,
                   h, k_b, k_p, plant, rho, rk4)

HORIZON = {"grown": T, "standard_T": T, "standard_1.5T": 1.5 * T, "standard_2T": 2.0 * T}
DT = 0.02                        # sampling step of the plant and of the rollout
DURATION = 20.0
START = np.array([-2.0, 11.0, 0.0, 0.0, 0.0, 0.0])
N = 640                          # flow steps per link, Theorem 2
BUDGET = 64                      # blocks of duration T per visit, Section V-B
# Visits per update, Section V-B.  Fixed to reproduce the paper's runs; with the
# time rule (no visit starts after a computation time of T) they depend on the machine.
VISITS = (4, 3, 3)


def flow(x, n, dt):
    """Backup flow phi_b from x, n RK4 steps."""
    X = [np.asarray(x, float)]
    for _ in range(n):
        X.append(rk4(f_b, X[-1], dt))
    return np.array(X)


@njit(cache=True)
def rollout(x, horizon):
    """phi_b(tau_j, x) and Phi_b(tau_j, x), tau_j = j DT."""
    n = int(round(horizon / DT))
    X, Phi = np.empty((n + 1, 6)), np.empty((n + 1, 6, 6))
    X[0], Phi[0] = x, np.eye(6)
    for i in range(n):
        x0, P0 = X[i], Phi[i]
        k1, J = f_b_nb(x0)
        K1 = J @ P0
        k2, J = f_b_nb(x0 + 0.5 * DT * k1)
        K2 = J @ (P0 + 0.5 * DT * K1)
        k3, J = f_b_nb(x0 + 0.5 * DT * k2)
        K3 = J @ (P0 + 0.5 * DT * K2)
        k4, J = f_b_nb(x0 + DT * k3)
        K4 = J @ (P0 + DT * K3)
        X[i + 1] = x0 + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        Phi[i + 1] = P0 + DT / 6.0 * (K1 + 2 * K2 + 2 * K3 + K4)
    return X, Phi


@njit(cache=True)
def qp(y, A, b):
    """QP (19) with W_Q = I: projection of y onto U and A u >= b."""
    box = np.array([[U_LO[0], U_LO[1]], [U_HI[0], U_LO[1]], [U_HI[0], U_HI[1]], [U_LO[0], U_HI[1]]])
    poly, b = box, b - 1e-10
    for k in range(len(b)):
        if min(A[k, 0] * U_LO[0], A[k, 0] * U_HI[0]) + min(A[k, 1] * U_LO[1], A[k, 1] * U_HI[1]) >= b[k]:
            continue
        s = poly[:, 0] * A[k, 0] + poly[:, 1] * A[k, 1] - b[k]
        if s.min() >= 0:
            continue
        n, m, out = len(poly), 0, np.empty((2 * len(poly), 2))
        for j in range(n):
            q = (j + 1) % n
            if s[j] >= 0:
                out[m], m = poly[j], m + 1
            if s[j] * s[q] < 0:
                out[m], m = poly[j] + s[j] / (s[j] - s[q]) * (poly[q] - poly[j]), m + 1
        if m == 0:
            return False, y
        poly = out[:m]
    if np.all(A[:, 0] * y[0] + A[:, 1] * y[1] >= b) and np.all((U_LO <= y) & (y <= U_HI)):
        return True, y
    best, dist = y, np.inf
    for j in range(len(poly)):
        e = poly[(j + 1) % len(poly)] - poly[j]
        t = ((y[0] - poly[j, 0]) * e[0] + (y[1] - poly[j, 1]) * e[1]) / max(e[0] ** 2 + e[1] ** 2, 1e-300)
        c = poly[j] + min(max(t, 0.0), 1.0) * e
        if (c[0] - y[0]) ** 2 + (c[1] - y[1]) ** 2 < dist:
            best, dist = c, (c[0] - y[0]) ** 2 + (c[1] - y[1]) ** 2
    return True, best


def safety_filter(x, horizon, fam=None):
    """Safety filter (17)-(19) of the family, or the standard bCBF with S_0 if fam is None."""
    X, Phi = rollout(x, horizon)
    gx, kb = g(x), k_b(x)
    hv = h(X)
    A = np.einsum("ni,nij,jk->nk", grad_h(X), Phi, gx)
    b = A @ kb - alpha_1(np.maximum(hv, 0) if fam else hv)
    if fam:
        beta = fam.beta(X[-1])
        H = beta.max()
        At = (X[-1] - fam.c) @ (-2 * P @ Phi[-1] @ gx) / fam.R[:, None]
        bt = At @ kb - alpha_hat(fam.R, max(H, 0.0)) - rho(H - beta)
    else:
        H = (R0 ** 2 - np.sum((P_SQRT @ (X[-1] - XI_STAR)) ** 2)) / R0
        grad = -2 * (X[-1] - XI_STAR) @ P / R0
        At = (grad @ Phi[-1] @ gx)[None]
        bt = At @ kb - alpha_hat(R0, H) - grad @ f_b_nb(X[-1])[0]
    left, u = qp(k_p(x), np.vstack([A, At]), np.r_[b, bt])
    return (u, False) if left else (kb, True)


def grow(fam, x, carried, cap):
    """One update of demand-driven growth, Section V-B."""
    S = [x]
    for _ in range(round(T / DT)):                      # nominal closed loop over [0, T]
        S.append(plant(S[-1], k_p(S[-1]), DT))
    queue = list(flow(np.array(S), round(T / DT), DT)[-1]) + list(carried)
    left, searches = [], 0
    for k, q in enumerate(queue):
        if searches == cap:
            return fam, left + queue[k:]
        if fam.H(q) >= 0:                               # not a coverage deficit
            continue
        searches += 1
        X = [q[None]]
        for _ in range(BUDGET):                         # blocks of T until the arc enters R_A
            X.append(flow(X[-1][-1], N, T / N)[1:])
            if fam.room(X[-1])[0].max() > 0:
                break
        else:                                           # budget spent
            left.append(q)
            continue
        links = place_links(np.vstack(X), T / N, fam, N)
        if links is None:
            left.append(q)
            continue
        new = fam.appended(links, T / N)
        fam = new if new is not None else fam
    return fam, left


def simulate(arm):
    """One closed-loop run."""
    horizon = HORIZON[arm]
    fam = Family() if arm == "grown" else None
    x, pending, appends, carried = START.copy(), None, [], []
    rec = {k: [] for k in ("t", "x", "u", "k_p", "empty")}
    every = round(T / DT)
    for i in range(round(DURATION / DT)):
        t = i * DT
        if fam and pending and i == pending[0]:
            appends.append((t, len(pending[1].R) - len(fam.R)))
            fam, pending = pending[1], None
        if fam and i % every == 0 and i + every < round(DURATION / DT):
            new, carried = grow(fam, x, carried, VISITS[i // every])
            pending = (i + every, new)                  # appended at t_j + T
        u, empty = safety_filter(x, horizon, fam)
        for k, v in zip(rec, (t, x, u, k_p(x), empty)):
            rec[k].append(v)
        x = plant(x, u, DT)
    out = {k: np.array(v) for k, v in rec.items()}
    if fam:
        out.update(c=fam.c, R=fam.R, appends=np.array(appends).reshape(-1, 2))
    return out


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    for arm in sys.argv[1:] or list(HORIZON):
        r = simulate(arm)
        np.savez_compressed(f"results/{arm}.npz", **r)
        print(f"{arm}: lowest z {r['x'][:, 1].min():.3f}, least h_1 {h(r['x']).min():.4f}, "
              f"empty {r['empty'].sum()}, balls {len(r.get('R', [1]))}, "
              f"appends {r.get('appends', np.zeros((0, 2))).tolist()}")
