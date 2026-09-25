"""Shared figure style for the paper panels.

Sizes follow the Communications AI & Computing formatting guidelines: Arial or
Helvetica lettering at 8-12 pt with minimal size variation, lowercase panel letters,
lines of at least 0.3 pt, vector output with embedded fonts, and patterns in addition
to colour so that no distinction relies on colour alone.

Figures are drawn at the width they are printed, the preprint's text width, so that
8 pt in this module is 8 pt on the page and the journal copy needs no rescaling.

Colour scheme, by role rather than by panel:

- in-vitro cultures (MCC, HCC) are warm and drawn with dashed lines;
- agents are cool, from seaborn's perceptually uniform "mako" map, darker for a longer
  memory or planning horizon, and drawn with solid lines;
- controls (CTL, RST, IS) are greys.

Every colour that is drawn as a line has at least 3:1 contrast against white, and
colours that share a panel stay at least 19 CIE76 units apart under simulated
protanopia, deuteranopia and tritanopia.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl

MM = 1 / 25.4  # inches per millimetre

#: Width of every figure page, equal to the preprint's \textwidth (6.5 in).
FIG_WIDTH_MM = 165.0
#: Height limit, so a figure and its caption fit on one preprint page.
FIG_MAX_HEIGHT_MM = 170.0

TEXT_PT = 8.0
PANEL_LABEL_PT = 10.0
AXIS_LINE_PT = 0.5
DATA_LINE_PT = 1.0
MIN_LINE_PT = 0.3

#: Case of the panel letters; the journal asks for lowercase.
PANEL_LETTER_CASE = "lower"

RC = {
    # Text: Arial throughout, with math falling back to STIX only for glyphs Arial
    # lacks (the blackboard-bold CL, B and C).
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": TEXT_PT,
    "axes.labelsize": TEXT_PT,
    "axes.titlesize": TEXT_PT,
    "xtick.labelsize": TEXT_PT,
    "ytick.labelsize": TEXT_PT,
    "legend.fontsize": TEXT_PT,
    "legend.title_fontsize": TEXT_PT,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "mathtext.fallback": "stixsans",
    "mathtext.default": "regular",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    # Axes: open frame, outward ticks, no gridlines.
    "axes.linewidth": AXIS_LINE_PT,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 2.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": AXIS_LINE_PT,
    "ytick.major.width": AXIS_LINE_PT,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "lines.linewidth": DATA_LINE_PT,
    "patch.linewidth": AXIS_LINE_PT,
    "legend.frameon": False,
    "legend.borderpad": 0.2,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.2,
    "figure.facecolor": "white",
    # Output: editable, embedded TrueType fonts in vector files.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
}


def apply() -> None:
    """Install the style globally, for scripts that draw only paper panels."""
    mpl.rcParams.update(RC)


@contextmanager
def context():
    """Apply the style inside a with-block and restore the previous settings after."""
    with mpl.rc_context(RC):
        yield


def mm(value: float) -> float:
    """Millimetres to inches, matplotlib's unit for figure sizes."""
    return value * MM


def panel_letter(letter: str) -> str:
    return letter.lower() if PANEL_LETTER_CASE == "lower" else letter.upper()


# --- colours and line styles ----------------------------------------------------

#: Four steps of seaborn "mako" (0.60 to 0.05), light to dark, one per horizon.
HORIZON_RAMP = ("#3497a9", "#38629d", "#3c3162", "#180d16")

CULTURE_DASH = {
    "MCC": (0, (4.0, 1.5)),
    "HCC": (0, (1.2, 1.2)),
}

GROUP_COLORS = {
    "MCC": "#D55E00",
    "HCC": "#7A3B12",
    "CTL": "#4D4D4D",
    "RST": "#808080",
    "IS": "#B3B3B3",
    "AIF-1": HORIZON_RAMP[1],
    **{f"CFL-{t}": c for t, c in zip((1, 2, 3, 4), HORIZON_RAMP)},
    **{f"DP-{t}": c for t, c in zip((2, 5, 10, 20), HORIZON_RAMP)},
}

GROUP_LINESTYLES = {
    **{g: "-" for g in GROUP_COLORS},
    **CULTURE_DASH,
    "CTL": (0, (4.0, 1.2, 1.0, 1.2)),
}

#: Observation modalities in the entropy panels of the DP and AIF-1 pages.
MODALITY_COLORS = {
    "ball-x": "#009E73",
    "ball-y": "#7B3294",
    "paddle-y": "#8C510A",
}
MODALITY_LINESTYLES = {
    "ball-x": "-",
    "ball-y": (0, (4.0, 1.5)),
    "paddle-y": (0, (1.2, 1.2)),
}

#: Trial windows compared in the box plots. The first is drawn open and the second
#: filled, so the two are told apart by pattern as well as by position.
WINDOW_ORDER = ("0-5", "6-20")
WINDOW_LABELS = {"0-5": "0–5 min", "6-20": "6–20 min"}
WINDOW_FILLED = {"0-5": False, "6-20": True}

#: Mean marker, visible on both open and filled boxes.
MEAN_PROPS = {
    "marker": "^",
    "markersize": 3.0,
    "markerfacecolor": "white",
    "markeredgecolor": "black",
    "markeredgewidth": 0.5,
}


def color_for(name: str) -> str:
    return GROUP_COLORS.get(name) or MODALITY_COLORS[name]


def linestyle_for(name: str):
    return GROUP_LINESTYLES.get(name) or MODALITY_LINESTYLES[name]


# --- axis labels ----------------------------------------------------------------

#: One name per quantity, used on every page.
LABELS = {
    "long_rally": "Long rallies (%)",
    "ace": "Aces (%)",
    "rally_length": "Rally length (hits)",
    "time": "Time (min)",
    "relative_improvement": "Relative improvement (%)",
    # Row 4 panels are too short for the one-line label at 8 pt.
    "relative_improvement_2l": "Relative\nimprovement (%)",
    "nte_cl": r"NTE of $\mathbb{CL}$",
    "nte_b": r"NTE of $\mathbb{B}$",
    "nte_c": r"NTE of $\mathbb{C}$",
    "risk": r"Risk $\Gamma$ (log)",
    "window": "Trial window",
}
