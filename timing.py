"""Computation times of Section VI (results/timing.txt).

    python timing.py
"""
import os
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import time  # noqa: E402

import numpy as np  # noqa: E402
import simulate  # noqa: E402
from family import Family  # noqa: E402
from simulate import DT, HORIZON, safety_filter, rollout  # noqa: E402

PASSES, REPS = 5, 3


def clock(f, *a):
    s = time.perf_counter()
    f(*a)
    return time.perf_counter() - s


def families(r):
    """Family in force at each sample of the grown run."""
    n = 1 + np.cumsum(np.r_[0, r["appends"][:, 1]]).astype(int)
    k = np.searchsorted(np.r_[0, r["appends"][:, 0]], r["t"] + 1e-9, side="right") - 1
    fams = []
    for m in n:
        f = Family()
        f.c, f.R = r["c"][:m], r["R"][:m]
        fams.append(f)
    return [fams[j] for j in k], n[k]


if __name__ == "__main__":
    lines = [f"one thread, times in ms, sample period {1e3 * DT:.0f} ms",
             "run            window   terminal constraints  rollout  rest  step (median, 99%, max)"]
    arms = {}
    for arm in ("standard_T", "standard_1.5T", "standard_2T", "grown"):
        r = np.load(f"results/{arm}.npz")
        fam, n = families(r) if arm == "grown" else ([None] * len(r["t"]), np.ones(len(r["t"]), int))
        safety_filter(r["x"][0], HORIZON[arm], fam[0])     # compiles
        arms[arm] = (r, fam, n, HORIZON[arm], [], [])
    order = list(arms)
    for p in range(PASSES):                              # the order of the runs rotates
        for arm in order[p % 4:] + order[:p % 4]:
            r, fam, n, hz, roll, step = arms[arm]
            roll.append([clock(rollout, x, hz) for x in r["x"]])
            step.append([clock(safety_filter, x, hz, f) for x, f in zip(r["x"], fam)])
    for arm, (r, fam, n, hz, roll, step) in arms.items():
        roll, step = 1e3 * np.array(roll), 1e3 * np.array(step)
        for m in np.unique(n):
            s, t = n == m, r["t"][n == m]
            ro, st = np.median(roll[:, s], 0), np.median(step[:, s], 0)
            lines.append(f"{arm:14s} {t[0]:2.0f}-{t[-1] + DT:2.0f} s  {m:4d}  {np.median(ro):6.1f}  "
                         f"{np.median(st - ro):5.1f}  {np.median(st):5.1f} {np.percentile(step[:, s], 99):5.1f} "
                         f"{step[:, s].max():5.1f}")
    grow, spent = simulate.grow, []

    def timed(*a):
        s = time.perf_counter()
        out = grow(*a)
        spent.append(time.perf_counter() - s)
        return out
    simulate.grow = timed
    for _ in range(REPS):
        simulate.simulate("grown")
    g = np.array(spent).reshape(REPS, -1)
    lines.append("growth updates (s), median (range) over three runs: " + ", ".join(
        f"{np.median(c):.1f} ({c.min():.1f}-{c.max():.1f})" for c in g.T))
    open("results/timing.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
