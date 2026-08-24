from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_behaviour import paths

N_BALL_X = 38
UNIFORM = np.log(N_BALL_X)
MAX_H = 20
HORIZON = 5
REPORT_DEPTHS = [1, 2, 3, 4, 5, 10, 15, 20]


def decay_curve(root: str | Path = paths.RESULTS_PLANNING_DEPTH, horizon: int = HORIZON, max_h: int = MAX_H) -> pd.DataFrame:
    """Normalised entropy of the h-step-ahead prediction, per seed and action."""
    out_dir = Path(root)
    rows = []
    for path in sorted(out_dir.glob(f"B_h{horizon}_seed*.npy")):
        seed = int(path.stem.split("seed")[-1])
        B = np.load(path)                               # (next_state, state, action)
        for a in range(B.shape[2]):
            M = B[:, :, a]
            M = M / M.sum(axis=0, keepdims=True)        # columns are distributions over next state
            P = np.eye(N_BALL_X)
            for h in range(1, max_h + 1):
                P = M @ P
                col = P / P.sum(axis=0, keepdims=True)
                ent = -(col * np.log(np.clip(col, 1e-16, None))).sum(axis=0)
                rows.append(dict(seed=seed, action=a, h=h, nte=ent.mean() / UNIFORM))
    if not rows:
        raise FileNotFoundError(f"no saved matrices in {out_dir}")
    return pd.DataFrame(rows)


def summary(root: str | Path = paths.RESULTS_PLANNING_DEPTH, **kw) -> pd.DataFrame:
    d = decay_curve(root, **kw)
    s = d.groupby("h").nte.agg(["mean", "std"]).reset_index()
    s["n_seeds"] = d.seed.nunique()
    return s


def write_table(root: str | Path = paths.RESULTS_PLANNING_DEPTH,
                tab_dir: str | Path = paths.OUTPUTS_TABLES) -> pd.DataFrame:
    s = summary(root)
    idx = s.set_index("h")
    rows = [f"{h} & {idx.loc[h, 'mean']:.3f} & {idx.loc[h, 'std']:.3f}" for h in REPORT_DEPTHS]

    body = ["\\begin{table}[ht]", "\\centering", "\\small",
            "\\caption{Predictive information of the learned transition model as a function of "
            "rollout depth. Entries are the entropy of the $h$-step-ahead prediction, normalised "
            "by the entropy of the uniform distribution over the same support, so that a value of "
            "one means the rollout carries no information about the future state.}",
            "\\label{tab:planningdepth}",
            "\\begin{tabular}{rrr}", "\\toprule",
            "Depth $h$ & Normalised entropy & SD \\\\", "\\midrule"]
    body += [r + " \\\\" for r in rows]
    body += ["\\bottomrule", "\\end{tabular}",
             "\\\\[4pt]\\footnotesize{Computed from the transition parameters of the DP-5 agent "
             f"on {int(s.n_seeds.iloc[0])} seeds, averaged over the three actions and over all "
             "starting states. The environment itself is deterministic; what decays with depth is "
             "the information carried by the agent's \\textit{estimate} of it.}",
             "\\end{table}"]

    tab_dir = Path(tab_dir)
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "tab_planning_depth.tex").write_text("\n".join(body) + "\n")
    return s
