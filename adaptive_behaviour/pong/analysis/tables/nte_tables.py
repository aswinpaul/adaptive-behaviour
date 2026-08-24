from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from adaptive_behaviour import paths

N_BALL_X, N_BALL_Y, N_PADDLE, N_ACTIONS = 38, 8, 8, 3
N_STATES = N_BALL_X * N_BALL_Y * N_PADDLE

CL_MAX = N_STATES * np.log(N_ACTIONS)
B_MAX = {
    1: N_BALL_X * N_ACTIONS * np.log(N_BALL_X),
    2: N_BALL_Y * 1 * np.log(N_BALL_Y),
    3: N_PADDLE * 1 * np.log(N_PADDLE),
}

CL_FILES = {f"CFL-{t}": paths.RESULTS_CL / f"c_data_M{t}.csv" for t in (1, 2, 3, 4)}
B_FILES = {f"DP-{t}": paths.RESULTS_DPEFE / f"b_data_{t}.csv" for t in (2, 5, 10, 20)}
B_FILES["AIF-1"] = paths.RESULTS_SI / "b_data.csv"

BEHAVIOUR = {f"CFL-{t}": paths.RESULTS_CL / f"data_clmethod_M{t}.csv" for t in (1, 2, 3, 4)}
BEHAVIOUR.update({f"DP-{t}": paths.RESULTS_DPEFE / f"p_data_{t}.csv" for t in (2, 5, 10, 20)})
BEHAVIOUR["AIF-1"] = paths.RESULTS_SI / "p_data.csv"

ORDER = ["CFL-1", "CFL-2", "CFL-3", "CFL-4", "AIF-1", "DP-2", "DP-5", "DP-10", "DP-20"]


def _sources():
    for label, path in CL_FILES.items():
        yield label, path, "$\\mathbb{CL}$", CL_MAX
    for label, path in B_FILES.items():
        yield label, path, "$\\mathbb{B}$ (ball-x)", B_MAX[1]


def entropy_table() -> pd.DataFrame:
    """Window means of NTE against the analytic maximum, plus the fall across the whole trace."""
    rows = []
    for label, path, param, maximum in _sources():
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d["half"] = d.half.astype(int)
        d["nte"] = d.entropy_1 / maximum
        g = d.groupby("half").nte.mean()
        rows.append(dict(agent=label, parameter=param, maximum=maximum,
                         w1=g.get(0), w2=g.get(1), change=g.get(1) - g.get(0),
                         drop_pct=(d.entropy_1.max() - d.entropy_1.min()) / maximum * 100))
    out = pd.DataFrame(rows)
    return out.set_index("agent").loc[[a for a in ORDER if a in set(out.agent)]].reset_index()


def prediction_table() -> pd.DataFrame:
    """Does a seed whose parameters became more informed improve more?"""
    entropy = {**CL_FILES, **B_FILES}
    rows = []
    for agent in ORDER:
        bpath, epath = BEHAVIOUR.get(agent), entropy.get(agent)
        if bpath is None or epath is None:
            continue
        if not (bpath.exists() and epath.exists()):
            continue
        b, e = pd.read_csv(bpath), pd.read_csv(epath)
        b["half"] = b.half.astype(int)
        e["half"] = e.half.astype(int)
        bs = b.groupby(["session_num", "half"]).hit_count.mean().unstack()
        es = e.groupby(["session_num", "half"]).entropy_1.mean().unstack()
        m = pd.DataFrame({"d_behaviour": bs[1] - bs[0], "d_entropy": es[1] - es[0]}).dropna()
        if len(m) < 5:
            continue
        r, p = stats.pearsonr(m.d_entropy, m.d_behaviour)
        rho, prho = stats.spearmanr(m.d_entropy, m.d_behaviour)
        rows.append(dict(agent=agent, n=len(m), r=r, p=p, rho=rho, p_rho=prho))
    out = pd.DataFrame(rows)
    out["p_adj"] = multipletests(out.p, method="fdr_bh")[1]
    out["p_rho_adj"] = multipletests(out.p_rho, method="fdr_bh")[1]
    return out


# --------------------------------------------------------------------------- LaTeX

def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


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


def write_tables(tab_dir: str | Path = paths.OUTPUTS_TABLES) -> dict:
    ent, pred = entropy_table(), prediction_table()
    tab_dir = Path(tab_dir)
    tab_dir.mkdir(parents=True, exist_ok=True)

    rows = [f"{r.agent} & {r.parameter} & {r.maximum:.2f} & {r.w1:.4f} & {r.w2:.4f} & "
            f"{r.change:+.4f} & {r.drop_pct:.1f}" for _, r in ent.iterrows()]
    (tab_dir / "tab_nte.tex").write_text(booktabs(
        "lllrrrr",
        "Agent & Parameter & $\\mathrm{TE}_{\\max}$ & NTE 0--5 & NTE 6--20 & $\\Delta$ & "
        "Fall (\\% of max)",
        rows,
        "Normalised total entropy against the analytic maximum, by time window. NTE is one when "
        "the parameter is uniform and approaches zero as it becomes fully determined, so values "
        "are comparable across agents and parameters.",
        "tab:nte",
        "The final column is the fall from the highest to the lowest value attained over a trial, "
        "expressed as a percentage of the analytic maximum. The planning schemes move their "
        "transition parameters by an order of magnitude more than the counterfactual scheme moves "
        "its mapping, and the size of that movement does not order the agents by performance.",
        small=True))

    rows_p = [f"{r.agent} & {int(r.n)} & {r.r:+.3f} & {fmt_p(r.p_adj)} & {r.rho:+.3f} & "
              f"{fmt_p(r.p_rho_adj)}" for _, r in pred.iterrows()]
    (tab_dir / "tab_nte_predict.tex").write_text(booktabs(
        "lrrrrr",
        "Agent & Seeds & Pearson $r$ & $p_{\\mathrm{adj}}$ & Spearman $\\rho$ & "
        "$p_{\\mathrm{adj}}$",
        rows_p,
        "Across-seed association between the change in parameter entropy over a trial and the "
        "change in average rally length. A negative coefficient would indicate that seeds whose "
        "parameters became more informed improved more.",
        "tab:ntepredict",
        "No association survives correction for either coefficient. Parameter entropy change "
        "therefore does not predict behavioural improvement within any agent.",
        small=True))

    return {"entropy": ent, "prediction": pred}
