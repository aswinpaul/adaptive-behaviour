from __future__ import annotations

import re
from pathlib import Path

from adaptive_behaviour import paths

RUNNERS = {
    "CFL-$T$": "trial_clmethod.py",
    "DP-$T$": "trial_dpefe.py",
    "AIF-1": "trial_si.py",
}

# label -> (regex, how to render the captured group). Order fixes the row order of the table.
PATTERNS = [
    ("Episodes per trial",            r"^n_trials\s*=\s*(\d+)",                  "{}"),
    ("Action precision",              r"action_precision\s*=\s*(\d+)",           "{}"),
    ("Transition learning rate",      r"lr_pB\s*=\s*(\S+?)\s*$",                 "{}"),
    ("Numerical floor",               r"^EPS_VAL\s*=\s*(\S+?)\s",                "{}"),
    ("Prior preferences $\\mathbb{C}$",   r"^C\s*=\s*(\w+)\(",                   "{}"),
    ("Initial state prior $\\mathbb{D}$", r"^D\s*=\s*(\w+)\(",                   "{}"),
]

PRETTY = {
    "obj_array_uniform": "uniform",
    "random_single_categorical": "random categorical",
    "1e+16": "$10^{16}$",
    "1e-16": "$10^{-16}$",
}

HORIZON = {
    "CFL-$T$": ("Horizon", "memory, $T \\in \\{1,2,3,4\\}$"),
    "DP-$T$": ("Horizon", "planning, $T \\in \\{2,5,10,20\\}$"),
    "AIF-1": ("Horizon", "planning, $T = 1$"),
}


def audit(root: str | Path = paths.RUNNERS) -> dict:
    root = Path(root)
    out = {}
    for agent, fname in RUNNERS.items():
        src = (root / fname).read_text()
        vals = {}
        for label, pat, fmt in PATTERNS:
            m = re.search(pat, src, re.M)
            assert m, f"AUDIT FAILED: {label!r} not found in {fname}"
            raw = m.group(1)
            vals[label] = PRETTY.get(raw, fmt.format(raw))
        vals[HORIZON[agent][0]] = HORIZON[agent][1]
        out[agent] = vals
    return out


def write_table(root: str | Path = paths.RUNNERS,
                tab_dir: str | Path = paths.OUTPUTS_TABLES) -> dict:
    a = audit(root)
    agents = list(RUNNERS)
    labels = [l for l, _, _ in PATTERNS] + ["Horizon"]

    rows = []
    for label in labels:
        cells = [a[ag][label] for ag in agents]
        shared = "shared" if len(set(cells)) == 1 else "differs"
        rows.append(" & ".join([label] + cells + [shared]))

    body = ["\\begin{table}[ht]", "\\centering", "\\small",
            "\\caption{Hyperparameters of the three decision-making schemes, read directly from "
            "the simulation code. All settings that the schemes share are identical; the "
            "quantities that differ are the ones that define the schemes, together with the prior "
            "specification of AIF-1.}",
            "\\label{tab:hyperparams}",
            "\\begin{tabular}{lllll}", "\\toprule",
            "Setting & " + " & ".join(agents) + " & Status \\\\", "\\midrule"]
    body += [r + " \\\\" for r in rows]
    body += ["\\bottomrule", "\\end{tabular}",
             "\\\\[4pt]\\footnotesize{The horizon is the quantity under study and differs by "
             "construction. The prior specification of AIF-1 is the one unintended asymmetry; its "
             "effect is measured in Sec.~\\ref{sec:priorcheck} and found not to account for the "
             "performance of that agent.}",
             "\\end{table}"]

    tab_dir = Path(tab_dir)
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "tab_hyperparams.tex").write_text("\n".join(body) + "\n")
    return a
