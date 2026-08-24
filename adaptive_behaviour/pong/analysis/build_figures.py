#!/usr/bin/env python3
"""Assemble paper and README figure panels from generated plot PNGs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from adaptive_behaviour import paths

REPO_ROOT = paths.REPO_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAPER_DIR = paths.OUTPUTS_FIGURES
README_DIR = paths.OUTPUTS_README


@dataclass(frozen=True)
class PanelPage:
    name: str
    sources: tuple[tuple[str, Path], ...]


EXPERIMENT_BASELINE = paths.RESULTS_EXPERIMENT / "Rel_Improvement_SBI_exp.png"

PAGES = (
    PanelPage(
        name="cfl_agents",
        sources=(
            ("A", paths.RESULTS_CL / "long-rallies_CL_vs_SBI.png"),
            ("B", paths.RESULTS_CL / "aces_CL_vs_SBI.png"),
            ("C", paths.RESULTS_CL / "Avg_Rally_length_CL_vs_SBI.png"),
            ("D", paths.RESULTS_CL / "regression_lines_LongRallies_CL_v_SBI.png"),
            ("E", paths.RESULTS_CL / "regression_lines_Aces_CL_v_SBI.png"),
            ("F", paths.RESULTS_CL / "regression_lines_RL_v_SBI.png"),
            ("G", paths.RESULTS_CL / "entropy_cl_boxplot.png"),
            ("H", paths.RESULTS_CL / "gamma_CL_boxplot.png"),
            ("I", paths.RESULTS_CL / "entropy_cl_regression.png"),
            ("J", paths.RESULTS_CL / "gamma_regression.png"),
            ("K", paths.RESULTS_CL / "Rel_Improvement_CL_vs_SBI.png"),
            ("L", EXPERIMENT_BASELINE),
        ),
    ),
    PanelPage(
        name="dpefe_agents",
        sources=(
            ("A", paths.RESULTS_DPEFE / "long-rallies_AIF_vs_SBI.png"),
            ("B", paths.RESULTS_DPEFE / "aces_AIF_vs_SBI.png"),
            ("C", paths.RESULTS_DPEFE / "Avg_Rally_length_AIF_vs_SBI.png"),
            ("D", paths.RESULTS_DPEFE / "regression_lines_LongRallies_AIF_v_SBI.png"),
            ("E", paths.RESULTS_DPEFE / "regression_lines_Aces_AIF_v_SBI.png"),
            ("F", paths.RESULTS_DPEFE / "regression_lines_AIF_v_SBI.png"),
            ("G", paths.RESULTS_DPEFE / "entropy_b_dp5_boxplot.png"),
            ("H", paths.RESULTS_DPEFE / "entropy_c_dp5_boxplot.png"),
            ("I", paths.RESULTS_DPEFE / "entropy_b_dp5_regression.png"),
            ("J", paths.RESULTS_DPEFE / "entropy_c_dp5_regression.png"),
            ("K", paths.RESULTS_DPEFE / "Rel_Improvement_AIF_vs_SBI.png"),
            ("L", EXPERIMENT_BASELINE),
        ),
    ),
    PanelPage(
        name="aif_agent",
        sources=(
            ("A", paths.RESULTS_SI / "long-rallies_AIF_vs_SBI.png"),
            ("B", paths.RESULTS_SI / "aces_AIF_vs_SBI.png"),
            ("C", paths.RESULTS_SI / "Avg_Rally_length_AIF_vs_SBI.png"),
            ("D", paths.RESULTS_SI / "regression_lines_LongRallies_AIF_v_SBI.png"),
            ("E", paths.RESULTS_SI / "regression_lines_Aces_AIF_v_SBI.png"),
            ("F", paths.RESULTS_SI / "regression_lines_AIF_v_SBI.png"),
            ("G", paths.RESULTS_SI / "entropy_b_si_boxplot.png"),
            ("H", paths.RESULTS_SI / "entropy_c_si_boxplot.png"),
            ("I", paths.RESULTS_SI / "entropy_b_si_regression.png"),
            ("J", paths.RESULTS_SI / "entropy_c_si_regression.png"),
            ("K", paths.RESULTS_SI / "Rel_Improvement_AIF_vs_SBI.png"),
            ("L", EXPERIMENT_BASELINE),
        ),
    ),
)

PANEL_LEFTS = (0.010, 0.256, 0.502, 0.748)
PANEL_BOTTOMS = (0.672, 0.338, 0.004)
PANEL_WIDTH = 0.242
PANEL_HEIGHT = 0.320
LABEL_X_OFFSET = 0.004
LABEL_Y_OFFSET = 0.006
PANEL_LABEL_FONTSIZE = 21


def check_sources() -> None:
    missing = [source for page in PAGES for _, source in page.sources if not source.exists()]
    if missing:
        details = "\n".join(str(path.relative_to(REPO_ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing figure inputs:\n{details}")


def draw_page(page: PanelPage) -> plt.Figure:
    fig = plt.figure(figsize=(32, 18), dpi=100)
    for index, (label, source) in enumerate(page.sources):
        row, col = divmod(index, 4)
        left = PANEL_LEFTS[col]
        bottom = PANEL_BOTTOMS[row]
        ax = fig.add_axes((left, bottom, PANEL_WIDTH, PANEL_HEIGHT))
        image = mpimg.imread(source)
        ax.imshow(image, aspect="auto")
        ax.set_axis_off()
        fig.text(
            left - LABEL_X_OFFSET,
            bottom + PANEL_HEIGHT + LABEL_Y_OFFSET,
            label,
            ha="left",
            va="top",
            fontsize=PANEL_LABEL_FONTSIZE,
            fontweight="bold",
        )

    return fig


def save_readme_png(fig: plt.Figure, output_path: Path) -> None:
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    fig.savefig(tmp_path, dpi=180)
    tmp_path.replace(output_path)
    os.utime(output_path, None)


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    README_DIR.mkdir(parents=True, exist_ok=True)
    check_sources()

    pdf_path = PAPER_DIR / "fig_panels.pdf"
    # Omitting CreationDate makes the PDF byte-reproducible, so sync_paper.py --check
    # only reports a difference when a panel actually changed.
    with PdfPages(pdf_path, metadata={"CreationDate": None}) as pdf:
        for page in PAGES:
            fig = draw_page(page)
            pdf.savefig(fig)
            save_readme_png(fig, README_DIR / f"{page.name}.png")
            plt.close(fig)

    print(f"Wrote {pdf_path.relative_to(REPO_ROOT)}")
    for page in PAGES:
        print(f"Wrote {(README_DIR / f'{page.name}.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
