"""Fig. 1.

    python fig1_regions.py
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Patch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy.linalg import expm  # noqa: E402
from shapely import affinity  # noqa: E402
from shapely.geometry import LineString, Point, Polygon  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

A = np.diag([-1.0, -1.0])           # f_b(x) = A x
MU = -1.0                           # mu_a of (12)
T = 1.0                             # prediction horizon
THETA = 0.010                       # vartheta_a
N_J = N_K = 100                     # flow steps of links j and k
R0 = 0.29                           # S_0 = B(0, R0)
SAFE = (-0.95, 3.97, -1.10, 0.96)   # C_S: xmin, xmax, ymin, ymax
ENTRANCE_J = np.array([0.468, 0.078])
DOCK_RADIUS_J, DOCK_RADIUS_K = 0.105, 0.085
SUCCESSOR_K = 6                     # ball of link j containing the dock of link k

CONTRACTION = float(np.linalg.norm(expm(A * THETA), 2))
FLOW_T = expm(A * T)

COLORS = {
    "original_fill": "#F9EAC4", "original_edge": "#9C7019",
    "added_fill": "#DCEAF8", "expanded_edge": "#34689B",
    "terminal_fill": "#C1A9DF", "terminal_edge": "#4B3186",
    "ball_edge": "#7459AD", "dock_fill": "#F5F1FB", "flow": "#283D42",
    "endpoint": "#BC4B24", "safe_edge": "#929A9E",
}


def link(entrance, n, dock_radius, slack=0.998):
    """Centers on a backup arc and radii satisfying arrival and exposure (11)."""
    C = np.array([expm(A * (i * THETA)) @ np.asarray(entrance, float) for i in range(n + 1)])
    a = 1.0 + 2.0 * MU * THETA
    R = np.empty(n + 1)
    R[-1] = dock_radius
    for i in range(n - 1, -1, -1):
        d = C[i + 1] - C[i]
        E = np.linalg.norm(d - THETA * (A @ C[i]))
        disc = (2 * E) ** 2 - 4 * a * (np.dot(d, d) - R[i + 1] ** 2)
        if disc <= 0:
            raise RuntimeError(f"no radius satisfies the exposure condition at step {i}")
        R[i] = slack * min((-2 * E + math.sqrt(disc)) / (2 * a), R[i + 1] / CONTRACTION)
    return C, R


def margins(C, R):
    """Least margins of arrival and exposure over a link."""
    arrival, exposure = [], []
    for i in range(len(R) - 1):
        arrival.append(R[i + 1] - CONTRACTION * R[i])
        d = C[i + 1] - C[i]
        E = np.linalg.norm(d - THETA * (A @ C[i]))
        exposure.append(R[i + 1] ** 2 - (1 + 2 * MU * THETA) * R[i] ** 2 - np.dot(d, d) - 2 * R[i] * E)
    return min(arrival), min(exposure)


# link j docks into S_0, link k into a ball of link j
Cj, Rj = link(ENTRANCE_J, N_J, DOCK_RADIUS_J)
offset = 0.12 * np.array([-ENTRANCE_J[1], ENTRANCE_J[0]]) / np.linalg.norm(ENTRANCE_J)
Ck, Rk = link(np.linalg.solve(expm(A * (N_K * THETA)), Cj[SUCCESSOR_K] + offset), N_K, DOCK_RADIUS_K)

# rotate so that x below lies on the horizontal axis
ang = -math.atan2(*(np.linalg.inv(FLOW_T) @ Ck[0])[::-1])
Q = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
Cj, Ck = Cj @ Q.T, Ck @ Q.T
successor = Cj[SUCCESSOR_K]

# Definition 3
arr_j, exp_j = margins(Cj, Rj)
arr_k, exp_k = margins(Ck, Rk)
base_margin = R0 - (np.linalg.norm(Cj[-1]) + Rj[-1])
dock_margin = Rj[SUCCESSOR_K] - (np.linalg.norm(Ck[-1] - successor) + Rk[-1])
assert min(arr_j, exp_j, arr_k, exp_k, base_margin, dock_margin) > 0
assert 0 < N_J * THETA <= T + 1e-12 and 0 < N_K * THETA <= T + 1e-12

# R_A, O_T(S_0) and O_T(R_A)
RES = 180
S0 = Point(0, 0).buffer(R0, resolution=RES)
RA = unary_union([S0] + [Point(*c).buffer(r, resolution=RES)
                         for c, r in zip(np.vstack([Cj, Ck]), np.r_[Rj, Rk])])
SCALE = math.exp(T)
OT_S0 = affinity.scale(S0, xfact=SCALE, yfact=SCALE, origin=(0, 0))
OT_RA = affinity.scale(RA, xfact=SCALE, yfact=SCALE, origin=(0, 0))

# x with phi_b(T, x) at the entrance of link k
x_T = Ck[0]
x = np.linalg.inv(FLOW_T) @ x_T
minx, miny, maxx, maxy = OT_RA.bounds
assert SAFE[0] < minx and maxx < SAFE[1] and SAFE[2] < miny and maxy < SAFE[3]
assert not OT_S0.covers(Point(*x)) and not RA.covers(Point(*x))


def polygons(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif hasattr(geometry, "geoms"):
        for component in geometry.geoms:
            yield from polygons(component)


def draw_geometry(ax, geometry, face, edge="none", lw=0.7, zorder=1):
    for polygon in polygons(geometry):
        polygon = orient(polygon, sign=1.0)
        vertices, codes = [], []
        for ring in [polygon.exterior, *polygon.interiors]:
            coords = np.asarray(ring.coords)
            vertices.extend(coords)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(coords) - 2) + [MplPath.CLOSEPOLY])
        ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=face, edgecolor=edge,
                               linewidth=lw, zorder=zorder))


def draw_boundary(ax, geometry, **kwargs):
    if geometry.geom_type in ("LineString", "LinearRing"):
        coordinates = np.asarray(geometry.coords)
        ax.plot(coordinates[:, 0], coordinates[:, 1], **kwargs)
    elif hasattr(geometry, "geoms"):
        for component in geometry.geoms:
            draw_boundary(ax, component, **kwargs)


def callout(ax, text, xy, xytext, color, fontsize=8.5, ha="center"):
    return ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va="center", color=color,
                       fontsize=fontsize, zorder=15,
                       arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55,
                                   "shrinkA": 2.5, "shrinkB": 1.5})


def displayed(C, count=8):
    """Indices of the balls outlined."""
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    return np.unique([np.argmin(abs(distance - target))
                      for target in np.linspace(0, distance[-1], count)])


def main():
    plt.rcParams.update({"font.family": "serif", "text.usetex": True,
                         "text.latex.preamble": r"\renewcommand{\rmdefault}{ptm}",
                         "font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)

    height = 3.40 * (SAFE[3] - SAFE[2] + 0.07) / (SAFE[1] - SAFE[0] + 0.07)
    fig, ax = plt.subplots(figsize=(3.40, height))
    fig.subplots_adjust(left=0.008, right=0.992, bottom=0.008, top=0.992)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(SAFE[0] - 0.035, SAFE[1] + 0.035)
    ax.set_ylim(SAFE[2] - 0.035, SAFE[3] + 0.035)
    ax.axis("off")

    # C_S and operating sets
    ax.add_patch(Rectangle((SAFE[0], SAFE[2]), SAFE[1] - SAFE[0], SAFE[3] - SAFE[2],
                           facecolor="white", edgecolor=COLORS["safe_edge"], linewidth=0.55, zorder=0))
    ax.text(-0.85, 0.82, r"$\mathcal{C}_{\mathrm{S}}$", color="#576268", ha="left", va="center",
            fontsize=9.5)
    draw_geometry(ax, OT_S0, COLORS["original_fill"])
    draw_geometry(ax, OT_RA.difference(OT_S0), COLORS["added_fill"])
    ax.add_patch(Circle((0, 0), SCALE * R0, facecolor="none", edgecolor=COLORS["original_edge"],
                        linewidth=0.75, linestyle=(0, (3.0, 2.3)), zorder=9))
    draw_boundary(ax, OT_RA.boundary, color=COLORS["expanded_edge"], linewidth=1.05, linestyle="-",
                  zorder=11)

    # R_A, some of its balls and the docks
    draw_geometry(ax, RA, COLORS["terminal_fill"], COLORS["terminal_edge"], lw=0.75, zorder=4)
    for C, R in ((Cj, Rj), (Ck, Rk)):
        for i in displayed(C):
            ax.add_patch(Circle(C[i], R[i], facecolor="none", edgecolor=COLORS["ball_edge"],
                                linewidth=0.35, zorder=5))
    ax.add_patch(Circle(successor, Rj[SUCCESSOR_K], facecolor="none",
                        edgecolor=COLORS["terminal_edge"], linewidth=0.95, zorder=7))
    ax.add_patch(Circle(Ck[-1], Rk[-1], facecolor=COLORS["dock_fill"],
                        edgecolor=COLORS["terminal_edge"], linewidth=0.65, zorder=8))
    ax.add_patch(Circle(Cj[-1], Rj[-1], facecolor=COLORS["dock_fill"],
                        edgecolor=COLORS["terminal_edge"], linewidth=0.65, zorder=8))
    ax.add_patch(Circle((0, 0), R0, facecolor="none", edgecolor=COLORS["terminal_edge"],
                        linewidth=0.95, zorder=8))

    # labels
    ax.text(-0.18, 0.50, r"$\mathcal{O}_T(S_0)$", color=COLORS["original_edge"], fontsize=9,
            ha="center", va="center", zorder=15)
    ax.text(-0.08, 0.0, r"$S_0$", color=COLORS["terminal_edge"], fontsize=9, ha="center",
            va="center", zorder=15)
    callout(ax, r"$\mathcal{R}_{\mathcal{A}}$", (1.05, -0.10), (0.84, -0.99),
            COLORS["terminal_edge"], fontsize=9.5)
    cut = OT_RA.intersection(LineString([(3.00, -2), (3.00, 2)]))
    callout(ax, r"$\mathcal{O}_T(\mathcal{R}_{\mathcal{A}})$", (3.00, cut.bounds[3]), (3.25, 0.75),
            COLORS["expanded_edge"], fontsize=10)

    # links j and k, rollout from x
    for C in (Cj, Ck):
        ax.annotate("", xy=C[-1], xytext=C[0],
                    arrowprops={"arrowstyle": "-|>", "color": COLORS["flow"], "lw": 0.85,
                                "mutation_scale": 7, "shrinkA": 0, "shrinkB": 0}, zorder=10)
    callout(ax, r"$j$", (0.33, -0.09), (0.36, -0.50), COLORS["flow"])
    callout(ax, r"$k$", (0.85, 0.0), (0.72, 0.62), COLORS["flow"])
    ax.annotate("", xy=x_T, xytext=x,
                arrowprops={"arrowstyle": "-|>", "color": COLORS["flow"], "lw": 1.0,
                            "linestyle": (0, (3.5, 2.7)), "mutation_scale": 7,
                            "shrinkA": 2, "shrinkB": 2}, zorder=10)
    ax.scatter(*x, s=11, color=COLORS["flow"], zorder=12)
    ax.annotate(r"$x$", xy=x, xytext=(1, 5), textcoords="offset points", fontsize=9, ha="center",
                va="bottom", color=COLORS["flow"], zorder=15)
    ax.text(2.35, -0.14, r"$\varphi_{\mathrm{b}}(\tau,x)\in\mathcal{C}_{\mathrm{S}}$",
            color=COLORS["flow"], fontsize=8.5, ha="center", va="center", zorder=15)
    ax.scatter(*x_T, s=15, facecolor=COLORS["endpoint"], edgecolor="white", linewidth=0.45, zorder=13)
    callout(ax, r"$\varphi_{\mathrm{b}}(T,x)\in\mathcal{R}_{\mathcal{A}}$", x_T, (1.85, 0.70),
            COLORS["endpoint"], fontsize=8.5)

    handles = [
        Patch(facecolor=COLORS["original_fill"], edgecolor=COLORS["original_edge"], linewidth=0.6,
              label="Original operating set"),
        Patch(facecolor=COLORS["added_fill"], edgecolor=COLORS["expanded_edge"], linewidth=0.6,
              label="Added operating region"),
        Patch(facecolor=COLORS["terminal_fill"], edgecolor=COLORS["terminal_edge"], linewidth=0.6,
              label="Certified region"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.035), frameon=False,
              fontsize=6.8, handlelength=1.35, handleheight=0.85, labelspacing=0.5,
              handletextpad=0.6, borderaxespad=0)

    fig.savefig(out / "fig1_regions.pdf")
    plt.close(fig)
    print(f"margins of Definition 3: arrival {min(arr_j, arr_k):.2e}, exposure {min(exp_j, exp_k):.2e}, "
          f"base dock {base_margin:.2e}, dock of k {dock_margin:.2e}")


if __name__ == "__main__":
    main()
