# Reproducing the results

Every table and figure in the manuscript, mapped to the command that produces it.

Work is grouped by cost. Tiers 0–1 run in minutes from a clone, tier 2 in seconds with
the private recordings. Tier 4 needs a cluster.
Anything marked **private** cannot be run from a clone at all, because it reads in-vitro
culture recordings that are not redistributable.

| Tier | What it produces | Cost | From a clone |
|---|---|---|---|
| 0 | the manuscript PDF | minutes | yes |
| 1 | 5 of 13 tables | minutes | yes |
| 2 | `fig_panels.pdf`, the three results figures | seconds | **private** |
| 3 | aggregated CSVs, from raw per-seed output | minutes | raw output not shipped |
| 4 | all raw per-seed output | ~8,600 CPU-hours | cluster |

## Setup

```bash
uv sync
```

Every path resolves from the package location, so nothing depends on the working
directory. `adaptive_behaviour/paths.py` is the single place any location is defined.

## Tier 0 — compile the manuscript

The paper is a submodule and carries its own generated tables and figures, so it
compiles without this repository.

```bash
cd paper-adaptive-behaviour && pdflatex paper.tex
```

## Tier 1 — regenerate tables

Tables are written by the modules in `adaptive_behaviour/pong/analysis/tables/`, each
callable with no arguments. All write into `outputs/tables/`. Copy them, with the figures
from tier 2, into the paper with:

```bash
python sync_paper.py            # copy outputs/ into the paper repository
python sync_paper.py --check    # compare only; exit 1 if anything differs
```

### Where each table comes from

| Table | Produced by | Needs private data |
|---|---|---|
| `tab_units` | `01_statistics.ipynb` | **yes** |
| `tab_contrasts` | `01_statistics.ipynb` | **yes** |
| `tab_contrasts_ri` | `01_statistics.ipynb` | **yes** |
| `tab_within` | `01_statistics.ipynb` | **yes** |
| `tab_level` | `01_statistics.ipynb` | **yes** |
| `tab_absolute` | `01_statistics.ipynb` | **yes** |
| `tab_baselines` | `baseline_tables.write_tables()` | **yes** |
| `tab_baseline_contrasts` | `baseline_tables.write_tables()` | **yes** |
| `tab_nte` | `nte_tables.write_tables()` | no |
| `tab_nte_predict` | `nte_tables.write_tables()` | no |
| `tab_noise` | `noise_tables.write_table()` | no |
| `tab_planning_depth` | `planning_depth_table.write_table()` | no |
| `tab_hyperparams` | `hyperparam_table.write_table()` | no |

`tab_hyperparams` reads no data at all: it parses the runner sources and fails loudly if
a hyperparameter in the paper no longer matches the code, so it cannot go stale.

## Tier 2 — draw the results figures

```bash
python adaptive_behaviour/pong/analysis/build_figures.py   # -> outputs/figures, outputs/github-readme
```

**Private:** panels a–f and k–l compare the agents with the cultures, so the build reads
the in-vitro recordings and cannot run from a clone. The paper repository carries the
built `fig_panels.pdf`, so tier 0 is unaffected.

`build_figures.py` draws `fig_panels.pdf` directly from the data, as vector graphics,
in about ten seconds. Three modules split the work:

| Module | Role |
|---|---|
| `analysis/figdata.py` | what each panel plots: filters, grouping, z-score cut, entropy normalisation |
| `analysis/plots/panels.py` | how a panel is drawn: box plots, linear fits with 95% CIs, bars |
| `analysis/plots/style.py` | sizes, fonts, colours and axis labels shared by every panel |

The manuscript references the pages by number.

| Page | Content | Paper |
|---|---|---|
| 1 | `cfl_agents` | `page=1`, `fig:results-cfl` |
| 2 | `dpefe_agents` | `page=2`, `fig:results-dpefe` |
| 3 | `aif_agent` | `page=3`, `fig:results-aif` |

Each page is 165 mm wide (the preprint's text width) and at most 170 mm tall, with rows
a–c, d–f, g–j and k–l. Panel l is the same in-vitro baseline on all three pages, and
panels a–c and k–l share their y-range across the pages so the agent families can be
compared by eye.

**The build checks its own output.** Before writing anything it audits every page
against the target journal's figure rules (Communications AI & Computing) and stops,
listing the problems, if any text is outside 8–12 pt, any line is thinner than 0.3 pt,
any text leaves the page or overlaps another, or the page is the wrong size.

The output is deterministic: every bootstrapped interval passes `seed=0`, and
`fig_panels.pdf` is written without a creation timestamp, so two builds are
byte-identical and `python sync_paper.py --check` only reports a real change. Row order
in `figdata` is part of that contract, since the bootstrap resamples rows by position.

The ten `plot_*.py` scripts in `analysis/plots/` are the earlier per-panel PNG
pipeline. `build_figures.py` no longer uses them; the redesigned figures were checked
to plot exactly the same 1,385 statistics (box quartiles, means, whiskers, fit end
points, slopes, confidence limits and bar heights) as those scripts.

## Tier 3 — rebuild aggregates from raw per-seed output

The per-seed CSVs are not distributed (~2,400 files); the aggregates computed from them
are. If you have regenerated the raw output, rebuild the aggregates with the scripts in
`adaptive_behaviour/pong/analysis/aggregate/`.

## Tier 4 — regenerate everything from scratch

### Published runs, on a cluster

`adaptive_behaviour/pong/hpc/` holds the SLURM files. Each job is 99 tasks × 2 CPUs and
took **6–10 hours**. Set `ADAPTIVE_BEHAVIOUR_ROOT` to your clone before submitting.

```bash
sbatch adaptive_behaviour/pong/hpc/si_job.sh        # AIF-1
sbatch adaptive_behaviour/pong/hpc/dpefe_job.sh     # DP-2
sbatch adaptive_behaviour/pong/hpc/dpefe_job_2.sh   # DP-5
sbatch adaptive_behaviour/pong/hpc/dpefe_job_3.sh   # DP-10
sbatch adaptive_behaviour/pong/hpc/dpefe_job_4.sh   # DP-20
```

### CFL runs, locally and serially

There is deliberately no job script for `trial_clmethod.py`, and one cannot be written.
The runner takes no arguments, seeds once with `random.seed(10)` before an internal
100-trial loop, and hardcodes `memory_horizon` near the top of the file. Results are
therefore **sequence-dependent**: splitting seeds across processes changes every trial,
and launching 99 parallel copies would have them all write the same output file.

To reproduce, edit `memory_horizon` (1–4) and run once per horizon:

```bash
python adaptive_behaviour/pong/runners/trial_clmethod.py
```

Roughly 40 minutes to 2.5 hours per horizon, so 3–10 hours for all four.

### Two things to expect

**CFL reproduces approximately, not exactly.** In the noise sweep, the DP-5 zero-noise
control reproduced the published number to the digit, while CFL-4 gave 2.390 against a
published 2.470. This follows from the sequence dependence above: any change to how the
random stream is consumed shifts every subsequent trial. Treat CFL numbers as
reproducible to within that tolerance.

**Seed counts differ by agent.** CFL used 100 trials from its internal loop; AIF-1 and
every DP-T used 99 SLURM tasks. This is why episode counts read 7,000 for CFL and 6,930
for the others.

## Measured runtimes

From the stored notebook outputs, on a 12-core laptop unless noted.

| Stage | Measured |
|---|---|
| AIF-1, one seed alone | 48.5 min |
| AIF-1, 12 seeds at 6 workers | 334 min |
| DPEFE `T=5`, per seed | 50.1 / 77.8 / 231.7 min |
| CFL, all 100 trials at one horizon | 38.5 – 150.8 min |
| Baselines, per seed | 15 – 51 s |
| Baselines, all 396 | ~50 min |
| Noise sweep, all 28 runs | ~13 h |
| Published runs, per SLURM job | 6 – 10 h (99 tasks × 2 CPUs) |
| `build_figures.py`, all three pages | ~10 s |

The DPEFE per-seed spread is real, not measurement noise: cost depends on how long
rallies run, which varies with the seed.

## The notebooks

In `adaptive_behaviour/pong/notebooks/`, numbered in reading order. They hold code and
computed outputs only; the argument they support is in the manuscript.

| Notebook | Question | Private |
|---|---|---|
| `01_statistics` | Unit of analysis, cluster-robust intervals, agents vs cultures | **yes** |
| `02_planning_depth` | Why does a deeper planning horizon not help? | no |
| `03_normalised_entropy` | How far do the model parameters move, and does that predict behaviour? | no |
| `04_baselines` | Do non-active-inference controls reach the same performance? | **yes** |
| `05_noise_robustness` | Do the conclusions survive observation and action noise? | no |
| `06_prior_initialisation` | Does AIF-1's random prior initialisation explain its performance? | no |

`01_statistics` and `04_baselines` read the in-vitro recordings and cannot be executed
from a clone. Their stored outputs are the record of what those analyses produced.

Three notebooks generate a patched copy of a published runner rather than modifying it:
`02_planning_depth`, `05_noise_robustness` and `06_prior_initialisation` write into
`adaptive_behaviour/pong/runners/generated/`. Each patch asserts that its anchor was
found and that the result never writes into a published results directory, so a silent
mismatch fails rather than corrupting data.

## Data availability

`data/private/in_vitro_cells_sentience.csv` holds the mouse and human cortical culture
recordings. It is not redistributable and is excluded from version control. Code that
needs it calls `paths.require_in_vitro()`, which explains what is missing and why rather
than raising a bare `FileNotFoundError`.
