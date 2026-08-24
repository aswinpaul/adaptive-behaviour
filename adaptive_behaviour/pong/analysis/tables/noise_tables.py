from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from adaptive_behaviour import paths

CONDITIONS = [(0.00, 0.00, "control"), (0.05, 0.00, "obs. $\\varepsilon = 0.05$"),
              (0.15, 0.00, "obs. $\\varepsilon = 0.15$"), (0.00, 0.10, "act. $\\eta = 0.10$")]
AGENTS = ["CFL-4", "DP-5"]


def _per_seed(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["half"] = d.half.astype(int)
    g = d.groupby("session_num").apply(
        lambda x: pd.Series({"w1": x.loc[x.half == 0, "hit_count"].mean(),
                             "w2": x.loc[x.half == 1, "hit_count"].mean()}),
        include_groups=False)
    return g.dropna()


def load(root: str | Path, agent: str, eps: float, eta: float) -> pd.DataFrame | None:
    root = Path(root)
    tag = f"{agent}_eps{eps:.2f}" + (f"_eta{eta:.2f}" if eta > 0 else "")
    if agent.startswith("CFL"):
        p = paths.RESULTS_NOISE / f"{tag}.csv"
        return _per_seed(pd.read_csv(p)) if p.exists() else None
    files = [f for f in sorted(glob.glob(str(paths.RESULTS_NOISE / f"{tag}_seed*.csv")))
             if re.search(r"seed\d+\.csv$", f)]
    if not files:
        return None
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["session_num"] = int(re.search(r"seed(\d+)", f).group(1))
        frames.append(d)
    return _per_seed(pd.concat(frames, ignore_index=True))


def sweep(root: str | Path = paths.RESULTS) -> pd.DataFrame:
    rows = []
    for agent in AGENTS:
        for eps, eta, label in CONDITIONS:
            g = load(root, agent, eps, eta)
            if g is None:
                continue
            gain = g.w2 - g.w1
            t, p = stats.ttest_1samp(gain, 0)
            rows.append(dict(agent=agent, condition=label, n=len(g),
                             w1=g.w1.mean(), w2=g.w2.mean(), median_w2=g.w2.median(),
                             gain=gain.mean(), p=p))
    return pd.DataFrame(rows)


def controls(root: str | Path = paths.RESULTS) -> dict:
    """Both control conditions must match the published runs; the standard differs by agent."""
    root = Path(root)
    out = {}
    ctrl = load(root, "CFL-4", 0.0, 0.0)
    pub = _per_seed(pd.read_csv(paths.RESULTS_CL / "data_clmethod_M4.csv"))
    if ctrl is not None:
        out["CFL-4"] = dict(control=ctrl.w2.mean(), published=pub.w2.mean(),
                            p=stats.mannwhitneyu(ctrl.w2, pub.w2).pvalue)
    same = []
    for s in range(6):
        a = paths.RESULTS_NOISE / f"DP-5_eps0.00_seed{s}.csv"
        b = paths.results_dpefe_raw(5) / f"data_{s}.csv"
        if a.exists() and b.exists():
            same.append(np.allclose(pd.read_csv(a).hit_count, pd.read_csv(b).hit_count))
    out["DP-5"] = dict(identical=bool(same) and all(same), n=len(same))
    return out


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write_table(root: str | Path = paths.RESULTS,
                tab_dir: str | Path = paths.OUTPUTS_TABLES) -> pd.DataFrame:
    s = sweep(root)
    rows = [f"{r.agent} & {r.condition} & {int(r.n)} & {r.w1:.2f} & {r.w2:.2f} & "
            f"{r.median_w2:.2f} & {r.gain:+.2f} & {fmt_p(r.p)}" for _, r in s.iterrows()]
    body = ["\\begin{table}[ht]", "\\centering", "\\small",
            "\\caption{Performance under observation noise and unreliable action execution. "
            "The first row of each block is the control condition, in which both manipulations "
            "are switched off and the published run is recovered.}",
            "\\label{tab:noise}",
            "\\begin{tabular}{llrrrrrr}", "\\toprule",
            "Agent & Condition & Seeds & 0--5 min & 6--20 min & Median & $\\Delta$ & $p$ \\\\",
            "\\midrule"]
    body += [r + " \\\\" for r in rows]
    body += ["\\bottomrule", "\\end{tabular}",
             "\\\\[4pt]\\footnotesize{The median is given because the distribution of rally "
             "length over seeds is heavy tailed for CFL-4, so the mean alone would overstate the "
             "typical run. The DP-5 rows rest on six seeds and are indicative only; no difference "
             "between its observation-noise conditions is distinguishable from zero.}",
             "\\end{table}"]
    tab_dir = Path(tab_dir)
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "tab_noise.tex").write_text("\n".join(body) + "\n")
    return s
