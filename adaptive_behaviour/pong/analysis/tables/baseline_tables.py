from __future__ import annotations

import glob
import os
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from adaptive_behaviour import paths

# Shared with 01_statistics.ipynb so that every table in the paper is drawn from one convention.
BASE_SEED = 20260817
N_BOOT = 100_000
BIO_GROUPS = {0: "MCC", 2: "HCC"}
EXCLUDED_CHIP = 7282
METRICS = ["rally", "long_rally", "ace"]

BASELINE_AGENTS = ["Random", "Heuristic", "DynaQ", "DynaQ-replay"]

AGENT_FILES = {
    "CFL-1": paths.RESULTS_CL / "data_clmethod_M1.csv",
    "CFL-2": paths.RESULTS_CL / "data_clmethod_M2.csv",
    "CFL-3": paths.RESULTS_CL / "data_clmethod_M3.csv",
    "CFL-4": paths.RESULTS_CL / "data_clmethod_M4.csv",
    "AIF-1": paths.RESULTS_SI / "p_data.csv",
    "DP-2":  paths.RESULTS_DPEFE / "p_data_2.csv",
    "DP-5":  paths.RESULTS_DPEFE / "p_data_5.csv",
    "DP-10": paths.RESULTS_DPEFE / "p_data_10.csv",
    "DP-20": paths.RESULTS_DPEFE / "p_data_20.csv",
}

# Ordered as they appear in the tables: cultures, then published agents, then the new baselines.
DISPLAY_ORDER = (["MCC", "HCC"] + list(AGENT_FILES) + BASELINE_AGENTS)


# --------------------------------------------------------------------------- loading

def load_episodes(root: str | Path = paths.RESULTS) -> pd.DataFrame:
    """One row per episode, for cultures, published agents and baselines alike."""
    root = Path(root)
    frames = []

    bio = pd.read_csv(paths.require_in_vitro())
    bio = bio[(bio.chip_id != EXCLUDED_CHIP) & (bio.group.isin(BIO_GROUPS))].copy()
    bio["group_name"] = bio.group.map(BIO_GROUPS)
    bio["source"] = "in-vitro"
    bio["replicate_id"] = bio.group_name + "|chip" + bio.chip_id.astype(str)
    bio["session_id"] = (bio.replicate_id + "|" + bio.date.astype(str)
                         + "|s" + bio.session_num.astype(str))
    frames.append(bio)

    for name, path in AGENT_FILES.items():
        d = pd.read_csv(path)
        d["group_name"] = name
        d["source"] = "in-silico"
        # One seed is one independent run, so a seed is both the session and the replicate.
        d["replicate_id"] = f"{name}|seed" + d.session_num.astype(int).astype(str)
        d["session_id"] = d["replicate_id"]
        frames.append(d)

    for f in sorted(glob.glob(str(paths.RESULTS_BASELINES / "*.csv"))):
        base = os.path.basename(f)
        agent = re.sub(r"_\d+\.csv$", "", base)
        seed = re.search(r"_(\d+)\.csv$", base).group(1)
        d = pd.read_csv(f)
        d["group_name"] = agent
        d["source"] = "baseline"
        d["replicate_id"] = f"{agent}|seed{seed}"
        d["session_id"] = d["replicate_id"]
        frames.append(d)

    cols = ["source", "group_name", "replicate_id", "session_id", "half", "hit_count"]
    ep = pd.concat([f[cols] for f in frames], ignore_index=True)

    ep["half"] = ep.half.astype(int)             # 0 = minutes 0-5, 1 = minutes 6-20
    ep["long_rally"] = (ep.hit_count >= 3).astype(int) * 100
    ep["ace"] = (ep.hit_count == 0).astype(int) * 100
    return ep.rename(columns={"hit_count": "rally"})


def session_means(ep: pd.DataFrame) -> pd.DataFrame:
    """Collapse episodes to one row per session and window, which is the unit we average over."""
    return ep.groupby(["source", "group_name", "replicate_id", "session_id", "half"],
                      as_index=False)[METRICS].mean()


# --------------------------------------------------------------------------- statistics

def cluster_boot(values, clusters, label, n_boot: int = N_BOOT):
    """Bootstrap a mean, resampling whole clusters rather than individual rows.

    Seeded from ``label`` so that a table does not depend on the order in which cells are run.
    """
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(label.encode()))
    df = pd.DataFrame({"v": values, "c": clusters})
    grouped = [g["v"].to_numpy() for _, g in df.groupby("c", sort=False)]
    sizes = np.array([len(g) for g in grouped])
    means = np.array([g.mean() for g in grouped])
    idx = rng.integers(0, len(grouped), (n_boot, len(grouped)))
    return (sizes[idx] * means[idx]).sum(axis=1) / sizes[idx].sum(axis=1)


def hedges_g(a, b):
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled * (1 - 3 / (4 * (n1 + n2) - 9))


def absolute_table(sess: pd.DataFrame, groups=None) -> pd.DataFrame:
    """Window-2 mean of each metric with a replicate-level bootstrap interval."""
    w2 = sess[sess.half == 1]
    groups = groups or [g for g in DISPLAY_ORDER if g in set(w2.group_name)]
    rows = []
    for g in groups:
        sub = w2[w2.group_name == g]
        row = dict(group=g, source=sub.source.iloc[0], n=sub.replicate_id.nunique())
        for m in METRICS:
            b = cluster_boot(sub[m].to_numpy(), sub.replicate_id.to_numpy(), f"{m}|{g}")
            lo, hi = np.percentile(b, [2.5, 97.5])
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = sub[m].mean(), lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def within_table(sess: pd.DataFrame, groups=None) -> pd.DataFrame:
    """Change in rally length from window 1 to window 2, paired within session."""
    wide = sess.pivot_table(index=["source", "group_name", "replicate_id", "session_id"],
                            columns="half", values="rally")
    wide = wide.rename(columns={0: "w1", 1: "w2"}).dropna().reset_index()
    wide["gain"] = wide.w2 - wide.w1
    groups = groups or [g for g in DISPLAY_ORDER if g in set(wide.group_name)]
    rows = []
    for g in groups:
        sub = wide[wide.group_name == g]
        b = cluster_boot(sub.gain.to_numpy(), sub.replicate_id.to_numpy(), f"gain|{g}")
        lo, hi = np.percentile(b, [2.5, 97.5])
        t, p = stats.ttest_1samp(sub.gain, 0)
        rows.append(dict(group=g, source=sub.source.iloc[0], n=len(sub),
                         w1=sub.w1.mean(), w2=sub.w2.mean(), gain=sub.gain.mean(),
                         lo=lo, hi=hi, p=p))
    out = pd.DataFrame(rows)
    out["p_adj"] = multipletests(out.p, method="fdr_bh")[1]
    return out


def contrast_table(sess: pd.DataFrame, pairs) -> pd.DataFrame:
    """Between-group differences in window-2 rally length, corrected within this family alone."""
    w2 = sess[sess.half == 1]

    def get(g):
        s = w2[w2.group_name == g]
        return s.rally.to_numpy(), s.replicate_id.to_numpy()

    rows = []
    for a, c in pairs:
        va, ca = get(a)
        vc, cc = get(c)
        ba = cluster_boot(va, ca, f"rally|{a}")
        bc = cluster_boot(vc, cc, f"rally|{c}")
        d = ba - bc
        lo, hi = np.percentile(d, [2.5, 97.5])
        # Two-sided bootstrap p, floored at the resolution the resample count can support.
        p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / N_BOOT)
        rows.append(dict(A=a, B=c, diff=va.mean() - vc.mean(), lo=lo, hi=hi,
                         g=hedges_g(va, vc), p=p))
    out = pd.DataFrame(rows)
    out["p_adj"] = multipletests(out.p, method="fdr_bh")[1]
    out["outcome"] = np.where(out.p_adj >= 0.05, "comparable",
                              np.where(out["diff"] > 0, "A higher", "B higher"))
    return out


# The comparisons that bear on the reviewer's question, from the most to the least direct.
CONTRAST_PAIRS = [
    ("DynaQ", "CFL-1"), ("DynaQ", "CFL-2"), ("DynaQ", "CFL-3"), ("DynaQ", "CFL-4"),
    ("DynaQ", "DP-5"), ("DynaQ", "AIF-1"),
    ("Random", "AIF-1"), ("Random", "MCC"), ("Random", "HCC"),
    ("Heuristic", "CFL-4"), ("DynaQ", "MCC"), ("DynaQ", "HCC"),
]


# --------------------------------------------------------------------------- LaTeX

def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ci(lo, hi, d=2):
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


def booktabs(colspec, header, rows, caption, label, note=None, small=False):
    out = ["\\begin{table}[ht]", "\\centering"]
    if small:
        out.append("\\small")
    out += [f"\\caption{{{caption}}}", f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    out += [r + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    if note:
        out.append(f"\\\\[4pt]\\footnotesize{{{note}}}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def write_tables(sess: pd.DataFrame, tab_dir: str | Path = paths.OUTPUTS_TABLES) -> dict:
    """Write the two baseline tables and return the frames they were built from."""
    tab_dir = Path(tab_dir)
    tab_dir.mkdir(parents=True, exist_ok=True)

    order = ["MCC", "HCC", "AIF-1", "CFL-1", "CFL-2", "CFL-3", "CFL-4",
             "DP-2", "DP-5", "DP-10", "DP-20"] + BASELINE_AGENTS
    absol = absolute_table(sess, order)
    within = within_table(sess, order)
    contr = contrast_table(sess, CONTRAST_PAIRS)

    merged = absol.merge(within[["group", "w1", "gain", "lo", "hi", "p_adj"]],
                         on="group", suffixes=("", "_gain"))
    rows = []
    for _, r in merged.iterrows():
        tag = {"in-vitro": "In vitro", "in-silico": "Active inference",
               "baseline": "Baseline"}[r.source]
        rows.append(f"{tag} & {r.group} & {r.rally:.2f} {ci(r.rally_lo, r.rally_hi)} & "
                    f"{r.gain:+.2f} {ci(r.lo, r.hi)} & {fmt_p(r.p_adj)}")
    (tab_dir / "tab_baselines.tex").write_text(booktabs(
        "lllll",
        "Class & Group & Rally length (6--20 min) & Change across trial & $p_{\\mathrm{adj}}$",
        rows,
        "Non-active-inference baselines beside the published agents and the cultures. "
        "Absolute rally length is the second-window mean; the change across the trial is the "
        "paired within-session difference. Intervals are replicate-level bootstraps.",
        "tab:baselines",
        "The random controller cannot learn, and its change across the trial is accordingly "
        "indistinguishable from zero, which calibrates the improvement metric. The reactive "
        "heuristic does not learn either, but is well matched to the task.",
        small=True))

    rows_c = [f"{r.A} & {r.B} & {r['diff']:+.2f} & {ci(r.lo, r.hi)} & {r.g:+.2f} & "
              f"{fmt_p(r.p_adj)} & {r.outcome}" for _, r in contr.iterrows()]
    (tab_dir / "tab_baseline_contrasts.tex").write_text(booktabs(
        "llrrrrl",
        "A & B & $\\Delta$ rally & 95\\% CI & Hedges' $g$ & $p_{\\mathrm{adj}}$ & Outcome",
        rows_c,
        "Baseline contrasts on second-window rally length. Benjamini-Hochberg correction is "
        "applied within this family of twelve comparisons, which is separate from the "
        "agent-versus-culture family of the main analysis.",
        "tab:baseline-contrasts",
        "``Comparable'' denotes an absence of a detectable difference, not evidence of "
        "equivalence.",
        small=True))

    return {"absolute": absol, "within": within, "contrasts": contr}
