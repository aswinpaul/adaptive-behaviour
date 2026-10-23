"""The generative model shared by every agent (fig_gen_model.pdf).

Three panels, drawn in the paper style:

    a   the Pong court to scale, with the discretisation each modality uses
    b   the generative model as a two-slice graph: three state factors, their
        transitions, the action and the identity likelihood
    c   the parameters of interest, their array shapes, and which scheme learns
        and uses each one

Every number in the figure is read from the code rather than typed in: the model
sizes and action sets from the runner scripts, the bin edges from the
environment's own state-to-observation functions, the court and paddle from the
game configuration, and the array shapes from pymdp's constructors. If the model
changes, the figure changes with it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from adaptive_behaviour.pong.analysis.plots import style
from adaptive_behaviour.pong.env import pong_statetoobs, readState
from adaptive_behaviour.pymdp.utils import random_A_matrix, random_B_matrix

RUNNERS = Path(__file__).resolve().parents[2] / "runners"
RUNNER_FILES = ("trial_si.py", "trial_dpefe.py", "trial_clmethod.py")
FACTORS = ("ball-x", "ball-y", "paddle-y")

FIG_H_MM = 80.0
INK = "#262626"
MUTED = "#595959"
LIGHT = "#E6E6E6"
COURT = "#F2F2F2"


# --- the model, as the code defines it ------------------------------------------------

def _literals(path: Path, names: set[str]) -> dict:
    """Module-level literal assignments of a script, read without executing it."""
    found = {}
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return found


def model_spec() -> dict:
    """State sizes and action sets, identical in all three runners (checked)."""
    names = {"s1_size", "s2_size", "s3_size", "s1_actions", "s2_actions", "s3_actions",
             "o1_size", "o2_size", "o3_size"}
    specs = [_literals(RUNNERS / f, names) for f in RUNNER_FILES]
    for f, spec in zip(RUNNER_FILES[1:], specs[1:]):
        if spec != specs[0]:
            raise ValueError(f"{f} defines a different generative model from {RUNNER_FILES[0]}")
    s = specs[0]
    num_states = [s["s1_size"], s["s2_size"], s["s3_size"]]
    actions = [s["s1_actions"], s["s2_actions"], s["s3_actions"]]
    num_controls = [len(a) for a in actions]
    return {
        "num_states": num_states,
        "num_obs": [s["o1_size"], s["o2_size"], s["o3_size"]],
        "actions": actions,
        "A": [a.shape for a in random_A_matrix([s["o1_size"], s["o2_size"], s["o3_size"]], num_states)],
        "B": [b.shape for b in random_B_matrix(num_states, num_controls)],
        # CFL melts the factors into one joint state and one joint action (agent_cl.py).
        "CL": (int(np.prod(num_controls)), int(np.prod(num_states))),
    }


def _bins(to_obs, lo: int, hi: int) -> list[tuple[int, int, int]]:
    """(bin, first pixel, last pixel) as the environment assigns pixels lo..hi."""
    runs: list[list[int]] = []
    for pixel in range(lo, hi + 1):
        label = to_obs(pixel)
        if runs and runs[-1][0] == label:
            runs[-1][2] = pixel
        else:
            runs.append([label, pixel, pixel])
    return [tuple(r) for r in runs]


# --- panel a: the court -------------------------------------------------------------------

def draw_court(ax, spec: dict) -> None:
    W, H = readState.AREA_WIDTH, readState.AREA_HEIGHT
    pw, ph = readState.PADDLE_WIDTH, readState.Pong_config().paddle_size
    ball = (372, 205)
    paddle_c = H / 2

    xbins = _bins(pong_statetoobs.bx_obs, 0, W)
    ybins = _bins(pong_statetoobs.by_obs, 0, H + 40)
    assert len({b for b, _, _ in xbins}) == spec["num_states"][0]
    assert len({b for b, _, _ in ybins}) == spec["num_states"][1]

    ax.set_aspect("equal")
    ax.set_xlim(-150, W + 175)
    ax.set_ylim(H + 175, -95)  # screen coordinates: y grows downwards
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), W, H, facecolor=COURT, edgecolor=INK, lw=0.75))
    ax.add_patch(Rectangle((0, paddle_c - ph / 2), pw, ph, facecolor=INK, lw=0))
    ax.add_patch(Circle(ball, 11, facecolor=INK, lw=0))

    def strip(cells, horizontal: bool, offset: float, depth: float, hit: int):
        for i, (b, a, z) in enumerate(cells):
            face = INK if b == hit else ("white" if i % 2 == 0 else LIGHT)
            outside = (z > (W if horizontal else H))
            if horizontal:
                rect = Rectangle((a, offset), z - a + 1, depth)
            else:
                rect = Rectangle((offset, a), depth, z - a + 1)
            rect.set(facecolor=face, edgecolor=MUTED, lw=0.3, hatch="////" if outside and b != hit else None)
            ax.add_patch(rect)

    bx_hit = pong_statetoobs.bx_obs(ball[0])
    by_hit = pong_statetoobs.by_obs(ball[1])
    py_hit = pong_statetoobs.by_obs(int(paddle_c))

    strip(xbins, True, H + 18, 30, bx_hit)             # ball-x, under the court
    strip(ybins, False, W + 18, 30, by_hit)            # ball-y, right of the court
    strip(ybins, False, -48, 30, py_hit)               # paddle-y, left of the court

    guide = dict(color=MUTED, lw=0.5, ls=(0, (2, 1.5)))
    ax.plot([ball[0], ball[0]], [ball[1] + 11, H + 18], **guide)
    ax.plot([ball[0] + 11, W + 18], [ball[1], ball[1]], **guide)
    ax.plot([-18, 0], [paddle_c, paddle_c], **guide)

    # Actions move the paddle by a fixed step.
    ax.add_patch(FancyArrowPatch((pw + 28, paddle_c - 70), (pw + 28, paddle_c + 70),
                                 arrowstyle="<|-|>", mutation_scale=5, lw=0.6, color=INK))
    ax.text(pw + 45, paddle_c, "Up · Stay · Down", va="center", ha="left")

    ax.add_patch(FancyArrowPatch((0, -30), (160, -30), arrowstyle="-|>", mutation_scale=5,
                                 lw=0.6, color=INK))
    ax.text(170, -30, "x: distance from paddle", va="center", ha="left")

    n1, n2, n3 = spec["num_states"]
    ax.text(W / 2, H + 60, f"ball-x: {n1} states of 16 px\nrate code, 4–41 Hz",
            ha="center", va="top", linespacing=1.3)
    ax.text(W + 118, H / 2, f"ball-y: {n2} states of 86 px\nplace code, 8 electrodes",
            ha="center", va="center", rotation=90, rotation_mode="anchor", linespacing=1.3)
    ax.text(-62, H / 2, f"paddle-y: {n3} states",
            ha="center", va="bottom", rotation=90, rotation_mode="anchor")


# --- panel b: the model as a graph ----------------------------------------------------------

def _arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=5, lw=0.6,
                                 color=INK, shrinkA=0, shrinkB=0, **kw))


def draw_graph(ax, spec: dict) -> None:
    ax.set_xlim(0, 52)
    ax.set_ylim(0, 70)
    ax.set_aspect("equal")
    ax.axis("off")

    r = 3.6
    x_label, x_t, x_t1, x_o = 0.5, 20, 34, 47
    rows = {name: y for name, y in zip(FACTORS, (46, 30, 14))}
    for k, (name, y) in enumerate(rows.items()):
        n = spec["num_states"][k]
        ax.text(x_label, y, f"{name}\n({n})", va="center", ha="left", linespacing=1.2)
        for x, sub in ((x_t, "t"), (x_t1, "t+1")):
            ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK, lw=0.75))
            ax.text(x, y, f"$s_{{{sub}}}$", ha="center", va="center")
        ax.add_patch(Rectangle((x_o - r, y - r), 2 * r, 2 * r, facecolor=LIGHT, edgecolor=INK, lw=0.75))
        ax.text(x_o, y, "$o_{t+1}$", ha="center", va="center")
        _arrow(ax, (x_t + r, y), (x_t1 - r, y))
        ax.text((x_t + x_t1) / 2, y + 1.2, "$\\mathbb{B}$", ha="center", va="bottom")
        _arrow(ax, (x_t1 + r, y), (x_o - r, y))
        ax.text((x_t1 + x_o) / 2, y + 1.2, "$\\mathbb{A}$", ha="center", va="bottom")

    # The action enters only the ball-x transition, as the runners define it.
    y_u = 60
    ax.add_patch(Rectangle(((x_t + x_t1) / 2 - r, y_u - r), 2 * r, 2 * r,
                           facecolor=INK, edgecolor=INK, lw=0.75))
    ax.text((x_t + x_t1) / 2, y_u, "$u_t$", ha="center", va="center", color="white")
    _arrow(ax, ((x_t + x_t1) / 2, y_u - r), (x_t1 - 0.72 * r, rows["ball-x"] + 0.72 * r))
    acts = ", ".join(a.replace("Play-", "") for a in spec["actions"][0])
    ax.text((x_t + x_t1) / 2, y_u + r + 1.0, f"{{{acts}}}", va="bottom", ha="center")

    n = spec["num_states"]
    ax.text(x_label, 3.5,
            f"$\\mathbb{{A}}$ = identity (fully observed)\n"
            f"joint state {n[0]} × {n[1]} × {n[2]} = {int(np.prod(n))}",
            va="center", ha="left", linespacing=1.3)


# --- panel c: parameters by scheme ----------------------------------------------------------

def _mark(ax, x, y, kind: str) -> None:
    """Drawn rather than typed, so no glyph falls back to a non-Arial font."""
    if kind == "used":
        ax.plot([x], [y], marker="o", ms=4.2, mfc=INK, mec=INK, mew=0.6)
    elif kind == "learned":
        ax.plot([x], [y], marker="o", ms=4.2, mfc="white", mec=INK, mew=0.8)
    else:
        ax.plot([x - 1.0, x + 1.0], [y, y], color=MUTED, lw=0.6)


def draw_table(ax, spec: dict) -> None:
    width = 55
    ax.set_xlim(0, width)
    ax.set_ylim(0, 70)
    ax.axis("off")

    B = spec["B"]
    b_shape = f"{'×'.join(map(str, B[0]))}\n{'×'.join(map(str, B[1]))} (×2)"
    c_shape = " · ".join(str(n) for n in spec["num_obs"])
    cl_shape = "×".join(map(str, spec["CL"]))
    rows = [
        ("$\\mathbb{B}$", "transitions", b_shape, ("used", "used", "learned")),
        ("$\\mathbb{C}$", "preferences", c_shape, ("used", "used", "none")),
        ("$\\mathbb{CL}$", "state–action", cl_shape, ("none", "none", "used")),
        ("$\\Gamma$", "risk", "$\\mathit{T}$ × 1", ("none", "none", "used")),
    ]
    x_sym, x_role, x_shape = 0.5, 6.0, 24.0
    x_marks = (42.0, 47.5, 53.0)
    top = 62
    for x, head in zip(x_marks, ("AIF-1", "DP-$\\mathit{T}$", "CFL-$\\mathit{T}$")):
        ax.text(x, top, head, ha="center", va="bottom", rotation=90)
    ax.text(x_shape, top, "shape", ha="left", va="bottom")
    ax.plot([0, width], [top - 1.5] * 2, color=INK, lw=0.5)

    y = top - 7
    for sym, role, shape, marks in rows:
        ax.text(x_sym, y, sym, ha="left", va="center")
        ax.text(x_role, y, role, ha="left", va="center")
        ax.text(x_shape, y, shape, ha="left", va="center", linespacing=1.2)
        for x, kind in zip(x_marks, marks):
            _mark(ax, x, y, kind)
        y -= 10.5
    ax.plot([0, width], [y + 5] * 2, color=INK, lw=0.5)

    for i, (kind, text) in enumerate((("used", "learned and used to choose actions"),
                                      ("learned", "learned, not used to choose actions"),
                                      ("none", "not part of the scheme"))):
        yy = y - 0.5 - 4.2 * i
        _mark(ax, 1.5, yy, kind)
        ax.text(4.0, yy, text, ha="left", va="center")


# --- figure ---------------------------------------------------------------------------------

def draw_figure() -> plt.Figure:
    spec = model_spec()
    W, H = style.FIG_WIDTH_MM, FIG_H_MM
    fig = plt.figure(figsize=(style.mm(W), style.mm(H)))

    def axes(left, bottom, width, height):
        return fig.add_axes((left / W, bottom / H, width / W, height / H))

    panels = {
        "a": (axes(0, 0, 58, 76), draw_court),
        "b": (axes(59, 3, 50, 70), draw_graph),
        "c": (axes(110, 3, 55, 70), draw_table),
    }
    for letter, (ax, draw) in panels.items():
        draw(ax, spec)
        fig.text(ax.get_position().x0, 1 - 1.0 / H, style.panel_letter(letter),
                 fontsize=style.PANEL_LABEL_PT, fontweight="bold", ha="left", va="top")
    return fig


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fig_gen_model_draft.pdf")
    with style.context():
        fig = draw_figure()
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"), dpi=300)
    print("wrote", out)
