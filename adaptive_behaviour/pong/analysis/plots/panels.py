"""Drawing functions for the three kinds of figure panel, in the paper style.

Each function draws one figdata panel description onto a given axes and nothing else:
no figure creation, no legends (the page adds shared ones), no global style changes.
The statistics are computed exactly as the legacy seaborn calls computed them:

- boxes: matplotlib's boxplot_stats with 1.5 IQR whiskers, which is what seaborn 0.13
  boxplot uses internally; drawn with ax.bxp so a box can be coloured by group and
  filled by trial window at the same time;
- trends: seaborn regplot, linear fit with a 95% CI bootstrapped with seed=0, shown
  as bars at the first and last minute rather than as a band;
- bars: seaborn barplot, mean with a 95% CI bootstrapped with seed=0.
"""

from __future__ import annotations

import re

import numpy as np
import seaborn as sns
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory, offset_copy

from adaptive_behaviour.pong.analysis import figdata
from adaptive_behaviour.pong.analysis.plots import style

BOX_WIDTH = 0.38
BOX_OFFSET = 0.2
BOX_LINE_PT = 0.75
TIME_TICKS = (0, 5, 10, 15, 20)
CI_DODGE_MIN = 0.32  # sideways offset between neighbouring CI bars, in minutes
CI_BAR_PT = 0.9
CI_CAP_PT = 1.2
UNIFORM_COLOR = "#404040"
#: Space above the uniform line for its label, as a fraction of the axis range.
UNIFORM_HEADROOM = 0.14


def _finish_y(ax, ylabel: str, ylim) -> None:
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if ylim is not None:
        ax.set_ylim(*ylim)


def _uniform_line(ax) -> None:
    """Mark NTE = 1, where the parameter carries no information, and label it inside."""
    lo, hi = ax.get_ylim()
    top = 1.0 + UNIFORM_HEADROOM * (1.0 - lo)
    if hi < top:
        ax.set_ylim(lo, top)
    ax.axhline(1.0, color=UNIFORM_COLOR, lw=0.5, ls=(0, (3, 2)), zorder=0)
    ax.text(0.99, 1.0, "uniform", transform=blended_transform_factory(ax.transAxes, ax.transData),
            ha="right", va="bottom", color=UNIFORM_COLOR)


#: Categories written PREFIX-SUFFIX are labelled by their suffix, with the shared prefix
#: set once under a bracket: six names at 8 pt do not fit side by side in a 45 mm panel.
_GROUPED = re.compile(r"^(CFL|DP|ball|paddle)-(\w+)$")
_BRACKET_TEXT = {"CFL": r"CFL-$\mathit{T}$", "DP": r"DP-$\mathit{T}$", "ball": "ball", "paddle": "paddle"}
#: Bracket position below the axis, in points: under the 8 pt tick labels.
_BRACKET_DROP_PT = 12.5


def _category_ticks(ax, categories: list[str]) -> None:
    labels, runs = [], {}
    for i, name in enumerate(categories):
        match = _GROUPED.match(name)
        if match:
            labels.append(match.group(2))
            runs.setdefault(match.group(1), []).append(i)
        else:
            labels.append(name)
    ax.set_xticks(range(len(categories)), labels)
    ax.tick_params(axis="x", length=0)
    base = blended_transform_factory(ax.transData, ax.transAxes)
    line_at = offset_copy(base, fig=ax.figure, y=-_BRACKET_DROP_PT, units="points")
    text_at = offset_copy(base, fig=ax.figure, y=-_BRACKET_DROP_PT - 1.0, units="points")
    for prefix, idx in runs.items():
        # Drawn as artists rather than plot lines so they never read as data.
        ax.add_artist(Line2D([idx[0] - 0.35, idx[-1] + 0.35], [0, 0], transform=line_at,
                             color="black", linewidth=style.AXIS_LINE_PT, clip_on=False))
        ax.text((idx[0] + idx[-1]) / 2, 0, _BRACKET_TEXT[prefix], transform=text_at,
                ha="center", va="top", clip_on=False)


def draw_boxes(ax, spec: figdata.Boxes, *, ylabel: str, ylim=None, uniform=False) -> None:
    """Two boxes per category: the first trial window open, the second filled."""
    frame = spec.frame
    categories = list(frame[spec.category].unique())
    for i, category in enumerate(categories):
        color = style.color_for(category)
        rows = frame[frame[spec.category] == category]
        for j, window in enumerate(style.WINDOW_ORDER):
            values = rows.loc[rows[spec.window] == window, spec.value].dropna().to_numpy()
            if values.size == 0:
                continue
            filled = style.WINDOW_FILLED[window]
            ax.bxp(
                cbook.boxplot_stats(values, whis=1.5),
                positions=[i + (j - 0.5) * 2 * BOX_OFFSET],
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=False,
                showmeans=True,
                meanprops=style.MEAN_PROPS,
                boxprops={"facecolor": color if filled else "white", "edgecolor": color,
                          "linewidth": BOX_LINE_PT},
                medianprops={"color": "white" if filled else color, "linewidth": style.DATA_LINE_PT},
                whiskerprops={"color": color, "linewidth": BOX_LINE_PT},
                capprops={"color": color, "linewidth": BOX_LINE_PT},
                manage_ticks=False,
            )
    _category_ticks(ax, categories)
    ax.set_xlim(-0.5, len(categories) - 0.5)
    _finish_y(ax, ylabel, ylim)
    if uniform:
        _uniform_line(ax)


def _ci_end_bars(ax, line, band, dodge: float) -> None:
    """Replace a fit's CI band by bars at its first and last minute.

    Overlapping bands of near-identical fits merge into one grey smear; bars at the two
    ends, where a linear fit's interval is widest, keep every series readable. The band
    stays on the axes, hidden, so the interval is the one regplot bootstrapped.
    """
    band.set_visible(False)
    vertices = band.get_paths()[0].vertices
    x = line.get_xdata()
    for x_end in (x[0], x[-1]):
        ys = vertices[np.isclose(vertices[:, 0], x_end), 1]
        half = (ys.max() - ys.min()) / 2
        ax.errorbar([x_end + dodge], [ys.min() + half], yerr=[[half], [half]], fmt="none",
                    ecolor=line.get_color(), elinewidth=CI_BAR_PT, capsize=CI_CAP_PT,
                    capthick=CI_BAR_PT)


def draw_trends(ax, spec: figdata.Trends, *, ylabel: str, ylim=None, uniform=False) -> None:
    """One linear fit per series over elapsed minutes, with its 95% CI at both ends."""
    frame = spec.frame
    groups = frame["group"].unique()
    for k, group in enumerate(groups):
        part = frame[frame["group"] == group]
        name = part[spec.series].unique()[0]
        sns.regplot(
            x=part["elapse_minute_rounded"], y=part[spec.value],
            x_estimator=np.mean, ci=95, seed=0, scatter=False, order=1,
            color=style.color_for(name), label=name, ax=ax,
            line_kws={"linestyle": style.linestyle_for(name), "linewidth": style.DATA_LINE_PT},
        )
        # regplot adds the fit line and then its band, so both are the latest artists.
        _ci_end_bars(ax, ax.lines[-1], ax.collections[-1],
                     dodge=(k - (len(groups) - 1) / 2) * CI_DODGE_MIN)
    ax.set_xlim(-1.5, 21.0)
    ax.set_xticks(TIME_TICKS)
    ax.set_xlabel(style.LABELS["time"])
    _finish_y(ax, ylabel, ylim)
    if uniform:
        _uniform_line(ax)


def draw_bars(ax, spec: figdata.Bars, *, ylabel: str, ylim=None) -> None:
    """Mean per group with a bootstrapped 95% CI."""
    frame = spec.frame
    order = list(frame[spec.category].unique())
    sns.barplot(
        data=frame, x=spec.category, y=spec.value, hue=spec.category,
        order=order, hue_order=order, palette={g: style.color_for(g) for g in order},
        errorbar=("ci", 95), seed=0, legend=False, width=0.7, saturation=1,
        edgecolor="black", linewidth=0.5,
        err_kws={"color": "black", "linewidth": BOX_LINE_PT}, ax=ax,
    )
    ax.set_xlabel("")
    ax.tick_params(axis="x", length=0)
    _finish_y(ax, ylabel, ylim)


# --- legend handles for the shared, page-level legends --------------------------------

def window_handles() -> list[Patch]:
    return [
        Patch(facecolor="#595959" if style.WINDOW_FILLED[w] else "white", edgecolor="#595959",
              linewidth=BOX_LINE_PT, label=style.WINDOW_LABELS[w])
        for w in style.WINDOW_ORDER
    ]


def series_handles(names) -> list[Line2D]:
    return [
        Line2D([], [], color=style.color_for(n), linestyle=style.linestyle_for(n),
               linewidth=style.DATA_LINE_PT, label=n)
        for n in names
    ]
