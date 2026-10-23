#!/usr/bin/env python3
"""Build the paper's figures: the three results figures and the generative model.

Each page is one figure at print size, 165 mm wide and at most 170 mm tall, drawn
directly from figdata in the paper style. Rows, with the panel letters the captions
use:

    a b c     performance by trial window (box plots)
    d e f     performance over the trial (linear fits with 95% CI)
    g h i j   model quantities: NTE and risk (CFL page) or NTE of B and C per modality
    k l       relative improvement of the agents (k) and of the in-vitro groups (l)

Panels a-c and k-l share their y-range across the three pages, so the agent families
can be compared by eye. The same run also draws fig_gen_model.pdf (plots/gen_model.py),
the generative model every agent shares. Every figure is audited against the
journal's figure rules before anything is written (see audit()).
"""

from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass
from pathlib import Path

from adaptive_behaviour import paths

REPO_ROOT = paths.REPO_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import Collection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from adaptive_behaviour.pong.analysis import figdata
from adaptive_behaviour.pong.analysis.plots import gen_model, panels, style

PAPER_DIR = paths.OUTPUTS_FIGURES
README_DIR = paths.OUTPUTS_README


@dataclass(frozen=True)
class Page:
    name: str
    family: str


#: In fig_panels.pdf order; paper.tex includes them as page=1, 2 and 3.
PAGES = (
    Page("cfl_agents", "cfl"),
    Page("dpefe_agents", "dpefe"),
    Page("aif_agent", "si"),
)

# --- layout, in millimetres ---------------------------------------------------------

ROWS = ("abc", "def", "ghij", "kl")
TOP_MM = 5.5             # row-1 panel letters and the window legend
AXES_MM = (29.0, 29.0, 29.0, 26.0)
GAP_BELOW_MM = (13.0, 17.0, 13.0)  # tick labels, brackets or x label, next row's letters
BOTTOM_MM = 5.0
PAGE_H_MM = TOP_MM + sum(AXES_MM) + sum(GAP_BELOW_MM) + BOTTOM_MM
PAGE_W_MM = style.FIG_WIDTH_MM
CELL_PAD_MM = 0.8        # between a cell's left edge and its y label
AXES_RIGHT_PAD_MM = 1.5  # between an axes' rightmost text and the next cell
LETTER_RISE_MM = 1.8     # panel letter baseline above the axes top
LEGEND_STRIP_MM = 5.0    # series legend between rows 2 and 3

L = style.LABELS


def panel_options(family: str, letter: str, shared: dict) -> dict:
    """Axis label, limits and the uniform-entropy marker for one panel."""
    options = {
        "a": {"ylabel": L["long_rally"]},
        "b": {"ylabel": L["ace"]},
        "c": {"ylabel": L["rally_length"]},
        "d": {"ylabel": L["long_rally"]},
        "e": {"ylabel": L["ace"]},
        "f": {"ylabel": L["rally_length"]},
        "k": {"ylabel": L["relative_improvement_2l"], "ylim": (0, 180)},
        "l": {"ylabel": L["relative_improvement_2l"], "ylim": (0, None)},
    }
    if family == "cfl":
        # The CL mapping stays close to uniform, so zoom to where it moves; the uniform
        # line keeps the zoom from overstating how far that is.
        lo = min(float(figdata.cl_nte()["entropy_1"].min()) - 0.004, 0.985)
        options |= {
            "g": {"ylabel": L["nte_cl"], "ylim": (lo, 1.004), "uniform": True},
            "h": {"ylabel": L["risk"]},
            "i": {"ylabel": L["nte_cl"], "ylim": (lo, 1.004), "uniform": True},
            "j": {"ylabel": L["risk"]},
        }
    else:
        agent = figdata.FAMILIES[family].modality_agent
        nte_b, nte_c = f"{L['nte_b']} ({agent})", f"{L['nte_c']} ({agent})"
        options |= {
            "g": {"ylabel": nte_b, "ylim": (0, 1.06), "uniform": True},
            "h": {"ylabel": nte_c, "ylim": (0, 1.06), "uniform": True},
            "i": {"ylabel": nte_b, "ylim": (0, 1.06), "uniform": True},
            "j": {"ylabel": nte_c, "ylim": (0, 1.06), "uniform": True},
        }
    chosen = dict(options[letter])
    if letter in shared:
        chosen["ylim"] = shared[letter]
    return chosen


def draw_panel(ax, spec, options: dict) -> None:
    if isinstance(spec, figdata.Boxes):
        panels.draw_boxes(ax, spec, **options)
    elif isinstance(spec, figdata.Trends):
        panels.draw_trends(ax, spec, **options)
    else:
        panels.draw_bars(ax, spec, **{k: v for k, v in options.items() if k != "uniform"})


def shared_ylims(letters: str = "abc") -> dict[str, tuple[float, float]]:
    """The union of each panel's automatic y-range over the three pages."""
    limits: dict[str, tuple[float, float]] = {}
    with style.context():
        for page in PAGES:
            data = figdata.page_data(page.family)
            for letter in letters:
                fig, ax = plt.subplots()
                draw_panel(ax, data[letter], panel_options(page.family, letter, {}))
                lo, hi = ax.get_ylim()
                plt.close(fig)
                old = limits.get(letter, (lo, hi))
                limits[letter] = (min(lo, old[0]), max(hi, old[1]))
    return limits


# --- page assembly --------------------------------------------------------------------

def _rect(left_mm, bottom_mm, width_mm, height_mm):
    return (left_mm / PAGE_W_MM, bottom_mm / PAGE_H_MM, width_mm / PAGE_W_MM, height_mm / PAGE_H_MM)


def _row_bottoms() -> list[float]:
    bottoms, top = [], PAGE_H_MM - TOP_MM
    for r, height in enumerate(AXES_MM):
        bottoms.append(top - height)
        top = top - height - (GAP_BELOW_MM[r] if r < len(GAP_BELOW_MM) else 0)
    return bottoms


def _fit_to_cells(fig, axes: dict, cells: dict) -> None:
    """Widen each axes to fill its cell, leaving room for its own tick and axis labels."""
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    px_per_mm = fig.dpi / 25.4
    width_px = fig.get_figwidth() * fig.dpi
    for letter, ax in axes.items():
        pos = ax.get_position()
        tight = ax.get_tightbbox(renderer)
        need_left = (pos.x0 * width_px - tight.x0) / px_per_mm
        need_right = max(0.0, (tight.x1 - pos.x1 * width_px) / px_per_mm)
        cell_left, cell_right = cells[letter]
        left = cell_left + CELL_PAD_MM + need_left
        right = cell_right - AXES_RIGHT_PAD_MM - need_right
        ax.set_position(_rect(left, pos.y0 * PAGE_H_MM, right - left, pos.height * PAGE_H_MM))


def draw_page(page: Page, shared: dict) -> tuple[plt.Figure, dict]:
    data = figdata.page_data(page.family)
    fig = plt.figure(figsize=(style.mm(PAGE_W_MM), style.mm(PAGE_H_MM)))
    axes, cells = {}, {}
    for row, bottom, height in zip(ROWS, _row_bottoms(), AXES_MM):
        cell_w = PAGE_W_MM / len(row)
        for c, letter in enumerate(row):
            cells[letter] = (c * cell_w, (c + 1) * cell_w)
            ax = fig.add_axes(_rect(c * cell_w + 12, bottom, cell_w - 14, height))
            draw_panel(ax, data[letter], panel_options(page.family, letter, shared))
            axes[letter] = ax

    # Trial windows: one key for every box plot on the page, top right.
    fig.legend(handles=panels.window_handles(), loc="upper right", ncol=2,
               bbox_to_anchor=(1 - 1.0 / PAGE_W_MM, 1 - 0.3 / PAGE_H_MM))
    # Groups: one key between rows 2 and 3, for the fits in d-f (and i-j on the CFL page).
    groups = list(data["d"].frame[data["d"].series].unique())
    strip_mm = _row_bottoms()[2] + AXES_MM[2] + LETTER_RISE_MM + 3.5 + LEGEND_STRIP_MM / 2
    fig.legend(handles=panels.series_handles(groups), loc="center", ncol=len(groups),
               bbox_to_anchor=(0.5, strip_mm / PAGE_H_MM))
    if page.family != "cfl":
        # Modalities: the fits in i and j; their colours match the boxes in g and h.
        modalities = list(data["i"].frame[data["i"].series].unique())
        axes["i"].legend(handles=panels.series_handles(modalities), loc="upper left",
                         bbox_to_anchor=(0.0, 0.86))

    for _ in range(2):
        _fit_to_cells(fig, axes, cells)

    for letter, ax in axes.items():
        top_mm = ax.get_position().y1 * PAGE_H_MM
        fig.text(cells[letter][0] / PAGE_W_MM, (top_mm + LETTER_RISE_MM) / PAGE_H_MM,
                 style.panel_letter(letter), fontsize=style.PANEL_LABEL_PT,
                 fontweight="bold", ha="left", va="bottom")
    return fig, axes


# --- audit ------------------------------------------------------------------------------

def _drawn_ticklabels(axis) -> list:
    # get_ticklabels() also returns labels of ticks outside the view limits, which are
    # never drawn; keep only the ticks inside them.
    lo, hi = sorted(axis.get_view_interval())
    tol = 1e-9 * max(1.0, abs(hi - lo))
    return [t.label1 for t in axis.get_major_ticks() if lo - tol <= t.get_loc() <= hi + tol]


def _texts(fig) -> list:
    texts = list(fig.texts)
    for legend in fig.legends:
        texts += legend.get_texts()
    for ax in fig.axes:
        texts += _drawn_ticklabels(ax.xaxis) + _drawn_ticklabels(ax.yaxis)
        texts += [ax.xaxis.label, ax.yaxis.label, ax.title, *ax.texts]
        if ax.get_legend():
            texts += ax.get_legend().get_texts()
    return [t for t in texts if t.get_visible() and t.get_text().strip()]


def audit(fig) -> list[str]:
    """Problems that break the journal's figure rules or the page layout; empty if none.

    Checks: page size; every text 8-12 pt; every stroked line or edge at least 0.3 pt;
    every text inside the page; no two texts overlapping.
    """
    problems = []
    width_mm, height_mm = (v * 25.4 for v in fig.get_size_inches())
    if abs(width_mm - style.FIG_WIDTH_MM) > 0.01 or height_mm > style.FIG_MAX_HEIGHT_MM:
        problems.append(f"page is {width_mm:.1f} x {height_mm:.1f} mm")

    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    texts = _texts(fig)
    boxes = []
    for t in texts:
        size = t.get_fontsize()
        if not 8 <= size <= 12:
            problems.append(f"text {t.get_text()!r} is {size} pt")
        box = t.get_window_extent(renderer)
        if box.x0 < page.x0 - 0.5 or box.x1 > page.x1 + 0.5 or box.y0 < page.y0 - 0.5 or box.y1 > page.y1 + 0.5:
            problems.append(f"text {t.get_text()!r} runs off the page")
        # A fixed sub-pixel margin, so glyphs that merely touch still count as clear.
        boxes.append(box.padded(-0.75))
    for (t1, b1), (t2, b2) in itertools.combinations(zip(texts, boxes), 2):
        if b1.overlaps(b2):
            problems.append(f"texts {t1.get_text()!r} and {t2.get_text()!r} overlap")

    for artist in fig.findobj(lambda a: isinstance(a, (Line2D, Patch, Collection))):
        if not artist.get_visible():
            continue
        if isinstance(artist, Line2D):
            widths = []
            if artist.get_linestyle() not in ("None", "", " ", "none"):
                widths.append(artist.get_linewidth())
            if artist.get_marker() not in ("None", "", " ", "none", None):
                widths.append(artist.get_markeredgewidth())
        elif isinstance(artist, Patch):
            widths = [artist.get_linewidth()] if artist.get_edgecolor()[3] > 0 else []
        else:
            widths = list(artist.get_linewidths())
        thin = [w for w in widths if 0 < w < style.MIN_LINE_PT]
        if thin:
            problems.append(f"{type(artist).__name__} drawn at {min(thin):.2f} pt")
    return problems


# --- entry point ------------------------------------------------------------------------

def build(paper_dir: Path = PAPER_DIR, readme_dir: Path = README_DIR) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    readme_dir.mkdir(parents=True, exist_ok=True)
    shared = shared_ylims()

    figures, problems = [], []
    with style.context():
        for page in PAGES:
            fig, _ = draw_page(page, shared)
            problems += [f"{page.name}: {p}" for p in audit(fig)]
            figures.append((page, fig))
        model_fig = gen_model.draw_figure()
        problems += [f"fig_gen_model: {p}" for p in audit(model_fig)]
    if problems:
        raise SystemExit("Figure audit failed:\n  " + "\n  ".join(problems))

    pdf_path = paper_dir / "fig_panels.pdf"
    # Omitting CreationDate makes the PDF byte-reproducible, so sync_paper.py --check
    # only reports a difference when a panel actually changed.
    with style.context(), PdfPages(pdf_path, metadata={"CreationDate": None}) as pdf:
        for page, fig in figures:
            pdf.savefig(fig)
            fig.savefig(readme_dir / f"{page.name}.png", dpi=300)
            plt.close(fig)

    model_path = paper_dir / "fig_gen_model.pdf"
    with style.context():
        model_fig.savefig(model_path, metadata={"CreationDate": None})
        plt.close(model_fig)

    for path in (pdf_path, model_path, *(readme_dir / f"{p.name}.png" for p in PAGES)):
        shown = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        print(f"Wrote {shown}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR,
                        help="where fig_panels.pdf goes (default: outputs/figures)")
    parser.add_argument("--readme-dir", type=Path, default=README_DIR,
                        help="where the PNG previews go (default: outputs/github-readme)")
    args = parser.parse_args()
    build(args.paper_dir, args.readme_dir)


if __name__ == "__main__":
    main()
