"""Data behind the paper's figure panels, separated from how the panels are drawn.

Each loader reproduces the data preparation of the per-family scripts in plots/
operation for operation: the same filters, the same grouping keys in the same order,
the same z-score cut. Row order is part of that contract. The confidence bands and
bar error bars are bootstrapped with seed=0, which resamples rows by position, so a
reordered frame would give different intervals even with identical values.

page_data(family) returns, for one figure page, what each panel a-l plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_behaviour import paths


@dataclass(frozen=True)
class Agent:
    csv: str
    group: int
    tag: str
    chip_id: int
    name: str


@dataclass(frozen=True)
class Family:
    key: str
    results: Path
    #: In-vitro groups kept for comparison, by their original code in the recordings.
    culture_groups: tuple[int, ...]
    agents: tuple[Agent, ...]
    #: For DP and AIF-1: the agent whose B and C entropies panels g-j show.
    modality_agent: str | None = None
    b_csv: str | None = None
    c_csv: str | None = None


_AGENT_DATE = "28.02.2024"

FAMILIES = {
    "cfl": Family(
        key="cfl",
        results=paths.RESULTS_CL,
        culture_groups=(0, 2),
        agents=tuple(
            Agent(f"data_clmethod_M{m}.csv", 6 + m, tag, 9 + m, f"CFL-{m}")
            for m, tag in zip((1, 2, 3, 4), ("cl-agent", "cl-agent-2", "cl-agent-3", "cl-agent-4"))
        ),
    ),
    "dpefe": Family(
        key="dpefe",
        results=paths.RESULTS_DPEFE,
        culture_groups=(0, 2),
        agents=tuple(
            Agent(f"p_data_{t}.csv", 6 + i, f"dp-agent-{i}", 9 + i, f"DP-{t}")
            for i, t in zip((1, 2, 3, 4), (2, 5, 10, 20))
        ),
        modality_agent="DP-5",
        b_csv="b_data_5.csv",
        c_csv="c_data_5.csv",
    ),
    "si": Family(
        key="si",
        results=paths.RESULTS_SI,
        culture_groups=(0, 1, 2),
        agents=(Agent("p_data.csv", 7, "aif-agent", 10, "AIF-1"),),
        modality_agent="AIF-1",
        b_csv="b_data.csv",
        c_csv="c_data.csv",
    ),
}


# --- panel descriptions -----------------------------------------------------------

@dataclass(frozen=True)
class Boxes:
    """Distribution per category and trial window (box plot)."""
    frame: pd.DataFrame
    value: str
    category: str = "group_name"
    window: str = "window"


@dataclass(frozen=True)
class Trends:
    """Per-minute values, one linear fit per group (regression lines with 95% CI)."""
    frame: pd.DataFrame
    value: str
    series: str = "group_name"


@dataclass(frozen=True)
class Bars:
    """Relative improvement per group (mean with bootstrapped 95% CI)."""
    frame: pd.DataFrame
    value: str = "normhc"
    category: str = "group_name"


# --- in-vitro recordings and performance --------------------------------------------

_CULTURE_NAMES = {0: "MCC", 1: "CTL", 2: "HCC", 3: "RST", 4: "IS"}
_CULTURE_ORDER = {"MCC": 0, "HCC": 1, "CTL": 2, "RST": 3, "IS": 4}
_EXCLUDED_CHIP = 7282


def _in_vitro(groups: tuple[int, ...] | None) -> pd.DataFrame:
    """Culture recordings with readable group names, renumbered MCC, HCC, CTL, RST, IS."""
    df = pd.read_csv(paths.require_in_vitro())
    df = df[df["chip_id"] != _EXCLUDED_CHIP]
    if groups is not None:
        df = df[df["group"].isin(groups)]
    df["group_name"] = 99
    for code, name in _CULTURE_NAMES.items():
        df["group_name"] = np.where(df["group"] == code, name, df["group_name"])
    for name, code in _CULTURE_ORDER.items():
        df["group"] = np.where(df["group_name"] == name, code, df["group"])
    return df


def _with_agents(df: pd.DataFrame, family: Family) -> pd.DataFrame:
    for agent in family.agents:
        new = pd.read_csv(family.results / agent.csv)
        new["group"] = agent.group
        new["tag"] = agent.tag
        new["date"] = _AGENT_DATE
        new["chip_id"] = agent.chip_id
        new["group_name"] = agent.name
        df = pd.concat([df, new])
    df["long_rally"] = np.where(df["hit_count"] >= 3, 1, 0)
    df["ace"] = np.where(df["hit_count"] == 0, 1, 0)
    df["number"] = df.index
    return df


def _add_window(df: pd.DataFrame) -> pd.DataFrame:
    # The legacy panels used half.astype(int) as the box hue; per-minute means of `half`
    # can be fractional where a minute straddles the split, and int() truncates them.
    df["window"] = np.where(df["half"].astype(int) == 0, "0-5", "6-20")
    return df


def _session_windows(df99: pd.DataFrame) -> pd.DataFrame:
    """Mean per session and trial window (panels a, b and the input to k, l)."""
    df2 = df99.groupby(["group", "tag", "chip_id", "date", "session_num",
                        "half", "elapse_minute_rounded", "group_name"]).mean(numeric_only=True)
    df5 = df2.reset_index().groupby(["group", "tag", "chip_id", "date",
                                     "session_num", "half", "group_name"]).mean(numeric_only=True)
    df4 = df5.reset_index()
    df4["%long_rally"] = df4["long_rally"] * 100
    df4["%ace"] = df4["ace"] * 100
    return _add_window(df4)


def _zscore_filter(lines: pd.DataFrame) -> pd.DataFrame:
    z = ((lines.hit_count - lines.hit_count.mean()) / lines.hit_count.std(ddof=0)).abs()
    return lines[z <= 2]


def _minute_windows(df99: pd.DataFrame) -> pd.DataFrame:
    """Per-minute means with |z| <= 2 on rally length, by window (panel c)."""
    lines = df99.groupby(["group", "group_name", "session_num", "date",
                          "tag", "chip_id", "elapse_minute_rounded"]).mean(numeric_only=True)
    return _add_window(_zscore_filter(lines.reset_index()))


def _minute_trends(df99: pd.DataFrame) -> pd.DataFrame:
    """Per-minute means with |z| <= 2 on rally length, minutes 0-19 (panels d-f)."""
    lines = df99.groupby(["group", "tag", "chip_id", "date", "session_num",
                          "elapse_minute_rounded", "group_name"]).mean(numeric_only=True)
    lines = _zscore_filter(lines.reset_index())
    lines["%long_rally"] = lines["long_rally"] * 100
    lines["%ace"] = lines["ace"] * 100
    return lines[lines["elapse_minute_rounded"] < 20]


def _improvement(df4: pd.DataFrame) -> pd.DataFrame:
    """Relative change in rally length from the first to the second window, per session."""
    filtdf = df4.groupby(["group_name", "group", "tag", "chip_id", "date",
                          "session_num", "half"]).mean(numeric_only=True)
    data = filtdf[["hit_count"]].copy().unstack(level=6)
    data = data.sort_values(by=["group"])
    data = data[data[("hit_count", 0)] != 0]
    data["normhc"] = (data[("hit_count", 1)] - data[("hit_count", 0)]) / data[("hit_count", 0)] * 100
    return data.reset_index()


@lru_cache(maxsize=None)
def performance(family_key: str) -> dict[str, pd.DataFrame]:
    family = FAMILIES[family_key]
    df99 = _with_agents(_in_vitro(family.culture_groups), family)
    sessions = _session_windows(df99)
    return {
        "sessions": sessions,
        "minutes": _minute_windows(df99),
        "trends": _minute_trends(df99),
        "improvement": _improvement(sessions),
    }


@lru_cache(maxsize=None)
def experiment_improvement() -> pd.DataFrame:
    """Relative improvement of all five in-vitro groups (panel l on every page)."""
    df99 = _in_vitro(groups=None)
    df99["long_rally"] = np.where(df99["hit_count"] >= 3, 1, 0)
    df99["ace"] = np.where(df99["hit_count"] == 0, 1, 0)
    df99["number"] = df99.index
    return _improvement(_session_windows(df99))


# --- model quantities -----------------------------------------------------------------

# Entropies are divided by their analytic maximum, reached when every distribution the
# parameter holds is uniform, so 1.0 means "carries no information" on every panel.
_CL_MAX = 2432 * np.log(3)  # 2432 states, one distribution over 3 actions each
_B_MAX = {"entropy_1": 38 * 3 * np.log(38),  # ball-x, 3 controllable actions
          "entropy_2": 8 * np.log(8),        # ball-y, uncontrolled
          "entropy_3": 8 * np.log(8)}        # paddle-y, uncontrolled
_C_MAX = {"entropy_1": np.log(38),           # one distribution over each modality
          "entropy_2": np.log(8),
          "entropy_3": np.log(8)}
_MODALITIES = ("ball-x", "ball-y", "paddle-y")


def _minute_means(df99: pd.DataFrame) -> pd.DataFrame:
    lines = df99.groupby(["group", "session_num", "group_name",
                          "elapse_minute_rounded"]).mean(numeric_only=True).reset_index()
    return lines[lines["elapse_minute_rounded"] < 21]


def _cl_agents(prefix: str) -> pd.DataFrame:
    frames = []
    for m in (1, 2, 3, 4):
        new = pd.read_csv(paths.RESULTS_CL / f"{prefix}_M{m}.csv")
        new["group_name"] = f"CFL-{m}"
        new["group"] = m - 1
        frames.append(new)
    return pd.concat(frames)


@lru_cache(maxsize=None)
def cl_nte() -> pd.DataFrame:
    df = _cl_agents("c_data")
    df["entropy_1"] = df["entropy_1"] / _CL_MAX
    return _add_window(df)


@lru_cache(maxsize=None)
def cl_risk() -> pd.DataFrame:
    """Risk term per step, in natural log with a 1e-15 floor."""
    df = _cl_agents("gamma_cl")
    df["gamma"] = np.log(df["gamma"] + 1e-15)
    return _add_window(df)


@lru_cache(maxsize=None)
def modality_entropy(family_key: str, parameter: str) -> pd.DataFrame:
    """NTE of B or C per observation modality, stacked long (ball-x, ball-y, paddle-y)."""
    family = FAMILIES[family_key]
    csv, maxima = {"b": (family.b_csv, _B_MAX), "c": (family.c_csv, _C_MAX)}[parameter]
    df = pd.read_csv(family.results / csv)
    for column, maximum in maxima.items():
        df[column] = df[column] / maximum
    parts = []
    for i, (column, name) in enumerate(zip(maxima, _MODALITIES)):
        part = df[[column, "session_num", "elapse_minute_rounded", "half"]].rename(
            columns={column: "entropy"})
        part["group"] = i
        part["group_name"] = name
        parts.append(part)
    return _add_window(pd.concat(parts))


# --- pages ----------------------------------------------------------------------------

def page_data(family_key: str) -> dict[str, Boxes | Trends | Bars]:
    """What each panel a-l of one figure page plots."""
    perf = performance(family_key)
    panels: dict[str, Boxes | Trends | Bars] = {
        "a": Boxes(perf["sessions"], "%long_rally"),
        "b": Boxes(perf["sessions"], "%ace"),
        "c": Boxes(perf["minutes"], "hit_count"),
        "d": Trends(perf["trends"], "%long_rally"),
        "e": Trends(perf["trends"], "%ace"),
        "f": Trends(perf["trends"], "hit_count"),
        "k": Bars(perf["improvement"]),
        "l": Bars(experiment_improvement()),
    }
    if family_key == "cfl":
        nte, risk = cl_nte(), cl_risk()
        panels |= {
            "g": Boxes(nte, "entropy_1"),
            "h": Boxes(risk, "gamma"),
            "i": Trends(_minute_means(nte), "entropy_1"),
            "j": Trends(_minute_means(risk), "gamma"),
        }
    else:
        b, c = modality_entropy(family_key, "b"), modality_entropy(family_key, "c")
        panels |= {
            "g": Boxes(b, "entropy"),
            "h": Boxes(c, "entropy"),
            "i": Trends(_minute_means(b), "entropy"),
            "j": Trends(_minute_means(c), "entropy"),
        }
    return dict(sorted(panels.items()))
