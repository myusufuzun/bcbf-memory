"""Fig. 2 and results/summary.txt.

    python fig2_quadrotor.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Ellipse, Patch  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from family import Family  # noqa: E402
from model import P_SQRT_INV, R0, T, XI_STAR, h  # noqa: E402
from simulate import DT, DURATION, flow, rollout  # noqa: E402

ARMS = (("standard_T", r"standard bCBF, $T$", "#82939d", "--"),
        ("standard_1.5T", r"standard bCBF, $1.5T$", "#4a6fa5", ":"),
        ("standard_2T", r"standard bCBF, $2T$", "#bb3e03", "-."),
        ("grown", r"grown safety filter, $T$", "#006d77", "-"))
FILL = ("#cfe3f5", "#f6d57f", "#e8a6c8", "#94869f", "#a9d6b5", "#f2b58c", "#b7bfe6")
MARK = ("#2b6cb0", "#d98a1f", "#a8447f", "#3d3d3d", "#2f7d4f", "#c0561a", "#4a55a8")
SHAPE = (None, "o", "s", "D", "^", "v", "P")
ZS, ZDS = np.arange(-1.45, 13.61, 0.05), np.arange(-5.0, 0.951, 0.025)


def slice_values(run):
    """Operating margin (16) on the slice y = theta = ydot = thetadot = 0, per family in force."""
    cache = Path("results/fig2_slice_cache.npz")
    counts = 1 + np.cumsum(np.r_[0, run["appends"][:, 1]]).astype(int)
    if cache.exists() and np.array_equal(np.load(cache)["counts"], counts):
        return np.load(cache)["margin"]
    Z, ZD = np.meshgrid(ZS, ZDS, indexing="ij")
    X0 = np.zeros((Z.size, 6))
    X0[:, 1], X0[:, 4] = Z.ravel(), ZD.ravel()
    margin = np.empty((len(counts), Z.size))
    for i in range(0, len(X0), 4000):
        nodes = flow(X0[i:i + 4000], round(T / DT), DT)
        clear = h(nodes).min(0)
        for k, n in enumerate(counts):
            fam = Family()
            fam.c, fam.R = run["c"][:n], run["R"][:n]
            margin[k, i:i + 4000] = np.minimum(fam.H(nodes[-1]), clear)
    margin = margin.reshape(len(counts), *Z.shape)
    np.savez_compressed(cache, margin=margin, counts=counts)
    return margin


def ellipse_axes():
    """y-z projection of the unit ball of ||.||."""
    A = P_SQRT_INV[:2]
    w, V = np.linalg.eigh(A @ A.T)
    return np.sqrt(w), np.degrees(np.arctan2(V[1, 1], V[0, 1]))


def panel_a(ax, run):
    """(a) Instance, stored balls and path."""
    (a, b), ang = ellipse_axes()
    grey = "#c8c8c8"
    ax.add_patch(plt.Polygon([(-3, 10), (-0.5, 10), (-0.5, -2), (0.5, -2), (0.5, 10), (3, 10),
                              (3, -2.4), (-3, -2.4)], fc=grey, ec="0.35", lw=0.7, zorder=0))
    n = 1 + np.cumsum(np.r_[0, run["appends"][:, 1]]).astype(int)
    for k in range(1, len(n)):                         # colors of (b), older on top
        ax.add_collection(PatchCollection(
            [Ellipse(c[:2], 2 * r * b, 2 * r * a, angle=ang)
             for c, r in zip(run["c"][n[k - 1]:n[k]], run["R"][n[k - 1]:n[k]])],
            fc=FILL[k], ec="none", zorder=3 - 0.1 * k))
    ax.add_patch(Ellipse(XI_STAR[:2], 2 * R0 * b, 2 * R0 * a, angle=ang, fc=FILL[0], ec=MARK[0],
                         lw=0.9, zorder=3))
    x = run["x"]
    ax.plot(x[:, 0], x[:, 1], color=ARMS[3][2], lw=1.4, zorder=4)
    for t in (0.0, 5.0, 7.0, 9.0):                     # four rollouts
        X = rollout(x[round(t / DT)], T)[0]
        ax.plot(X[:, 0], X[:, 1], color="#444444", lw=0.7, alpha=0.55, ls=(0, (2.2, 1.6)), zorder=7)
        ax.plot(X[0, 0], X[0, 1], ".", color="#444444", ms=2.6, zorder=7)
        ax.plot(X[-1, 0], X[-1, 1], "x", color="#444444", ms=3.0, mew=0.8, zorder=7)
    ax.plot(x[0, 0], x[0, 1], "o", mfc="white", mec="k", ms=4, mew=0.9, zorder=6)
    ax.plot(0, 0, "s", color="k", ms=4, zorder=6)
    ax.set(xlim=(-3, 3), ylim=(-2.4, 13.9), xlabel="$y$ (m)", ylabel="$z$ (m)",
           xticks=[-2, 0, 2], yticks=[-2, 0, 2, 4, 6, 8, 10, 12], title="(a) instance and path")
    for side in ax.spines.values():
        side.set_visible(True)


def panel_b(ax, run, margin):
    """(b) Operating set on the slice."""
    for k in range(len(margin) - 1, -1, -1):
        ax.contourf(ZDS, ZS, margin[k], levels=[0, margin[k].max() + 1], colors=[FILL[k]], zorder=1 + len(margin) - k)
    for k in range(len(margin) - 1, 0, -1):
        ax.contour(ZDS, ZS, margin[k], levels=[0], colors=[MARK[k]], linewidths=0.45, zorder=6)
    ax.contour(ZDS, ZS, margin[0], levels=[0], colors=[ARMS[0][2]], linewidths=1.0, linestyles=["--"], zorder=7)
    x = run["x"]
    k = np.argmax(np.abs(x[:, [0, 2, 3, 5]]).max(1) < 0.02)      # the path reaches the slice
    i = round(run["appends"][0, 0] / DT)                         # dotted from the first append
    ax.plot(x[i:k + 1, 4], x[i:k + 1, 1], color=ARMS[3][2], lw=1.0, ls=(0, (1.0, 1.3)), zorder=8)
    ax.plot(x[k:, 4], x[k:, 1], color=ARMS[3][2], lw=1.3, zorder=8)
    for k, (ta, _) in enumerate(run["appends"], start=1):
        i = round(ta / DT)
        ax.plot(x[i, 4], x[i, 1], SHAPE[k], ms=4.2, mfc=MARK[k], mec="white", mew=0.7, zorder=11)
    ax.set(xlim=(ZDS[0], 0.7), ylim=(-1.45, 13.3), xlabel=r"$\dot z$ (m\,s$^{-1}$)", ylabel="$z$ (m)",
           xticks=[-5, -4, -3, -2, -1, 0], yticks=[0, 2, 4, 6, 8, 10, 12], title="(b) slice of the operating set")
    sets = [Patch(fc=FILL[k], ec=MARK[k], lw=0.45) for k in range(1, len(margin))]
    k = 3 if len(sets) <= 3 else 2
    ax.legend([Patch(fc=FILL[0], ec=ARMS[0][2], ls="--", lw=0.9)] + sets[:k] + [tuple(sets[k:])] * (len(sets) > k)
              + [Line2D([], [], color=ARMS[3][2], lw=1.3)],
              ["$S_0$ alone"] + ["from $%g$~s" % ta for ta, _ in run["appends"][:k]]
              + ["later appends"] * (len(sets) > k) + ["grown safety filter"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
              loc="lower left", frameon=False, fontsize=6.5, handlelength=1.6, borderpad=0.1,
              labelspacing=0.2, handletextpad=0.4, borderaxespad=0.2)


def main():
    runs = {a: dict(np.load(f"results/{a}.npz")) for a, *_ in ARMS if Path(f"results/{a}.npz").exists()}
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "text.usetex": True, "font.family": "serif",
                         "text.latex.preamble": r"\renewcommand{\rmdefault}{ptm}"})
    fig = plt.figure(figsize=(3.40, 3.95), layout="constrained")
    rows = fig.add_gridspec(2, 1, height_ratios=[2.45, 1.5])
    top, low = rows[0].subgridspec(1, 2, width_ratios=[0.8, 1.2]), rows[1].subgridspec(1, 2)
    ax = [fig.add_subplot(top[0]), fig.add_subplot(top[1]), fig.add_subplot(low[0])]
    ax.append(fig.add_subplot(low[1], sharex=ax[2]))
    grown = runs["grown"]
    grown["appends"] = grown["appends"][grown["appends"][:, 1] > 0]    # updates that add balls
    margin = slice_values(grown)
    panel_a(ax[0], grown)
    panel_b(ax[1], grown, margin)
    for arm, label, color, ls in ARMS:
        if arm in runs:
            r = runs[arm]
            ax[2].plot(r["t"], r["x"][:, 1], color=color, ls=ls, lw=1.1)
            ax[3].plot(r["t"], np.linalg.norm(r["u"] - r["k_p"], axis=1), color=color, ls=ls, lw=1.1, label=label)
    for k, (ta, _) in enumerate(grown["appends"], start=1):
        for a in ax[2:]:
            a.axvline(ta, color=MARK[k], ls=(0, (2.2, 2.2)), lw=0.8, zorder=0)
    ax[2].set(xlabel="$t$ (s)", ylabel="$z$ (m)", title="(c) height")
    ax[3].set(xlabel="$t$ (s)", ylabel=r"$\|u-\mathbf{k}_{\mathrm{p}}\|_{\mathbf{W}_{\mathrm{Q}}}$",
              title="(d) departure", xticks=np.arange(0, DURATION + 1, 5), ylim=(-0.5, None))
    fig.legend(*ax[3].get_legend_handles_labels(), loc="outside lower center", ncols=2, fontsize=6.5,
               frameon=False, borderpad=0.0, labelspacing=0.2, columnspacing=1.2, handlelength=2.2)
    fig.savefig("results/fig2_quadrotor.pdf")

    lines = []
    for arm, *_ in ARMS:
        if arm in runs:
            r = runs[arm]
            dev = np.linalg.norm(r["u"] - r["k_p"], axis=1) > 1e-9
            lines.append(f"{arm:14s} lowest z {r['x'][:, 1].min():8.3f}  final z {r['x'][-1, 1]:7.3f}  "
                         f"least h_1 {h(r['x']).min():.4f}  "
                         f"empty {int(r['empty'].sum())}  departs {dev.sum()} steps, last at "
                         f"{r['t'][dev].max() if dev.any() else 0:.2f} s")
    X0 = np.zeros((10500, 6))                           # at rest on the slice
    X0[:, 1] = np.arange(-1.0, 9.5, 0.001)
    nodes = flow(X0, round(T / DT), DT)
    rest = ["S_0 alone"] + ["after append at %g s" % ta for ta, _ in grown["appends"]]
    for n, name in zip(1 + np.cumsum(np.r_[0, grown["appends"][:, 1]]).astype(int), rest):
        fam = Family()
        fam.c, fam.R = grown["c"][:n], grown["R"][:n]
        ok = np.minimum(fam.H(nodes[-1]), h(nodes).min(0)) >= 0
        lines.append(f"operating set at rest reaches z = {X0[ok, 1].min():.3f}, {name}")
    lines.append(f"balls {len(grown['R'])}, appends (t, balls) {grown['appends'].tolist()}")
    Path("results/summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
