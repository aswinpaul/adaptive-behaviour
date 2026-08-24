import os
import sys

# Pin BLAS to a single thread BEFORE numpy is imported. The matrices here are small (2432 states),
# so threaded BLAS provides no speed-up, and with several runners in flight the default of one
# thread per core oversubscribes the machine badly.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import adaptive_behaviour  # noqa: F401,E402  (establishes import paths for pymdp/agents)
from adaptive_behaviour import paths  # noqa: E402

import revision_runners as R  # noqa: E402  (import after the thread pinning above)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: {sys.argv[0]} <agent> <seed> [n_episodes]")
    agent = sys.argv[1]
    seed = int(sys.argv[2])
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else R.N_EPISODES

    if agent not in R.BASELINES:
        raise SystemExit(f"unknown agent {agent!r}; choose from {sorted(R.BASELINES)}")

    out_dir = paths.RESULTS_BASELINES
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = R.run_seed(R.BASELINES[agent](seed), seed, n_episodes=n_episodes)
    frame.to_csv(out_dir / f"{agent}_{seed}.csv", index=False)


if __name__ == "__main__":
    main()
