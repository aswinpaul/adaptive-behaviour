#!/bin/bash
#SBATCH --job-name=dpefe_job
#SBATCH --output eta_job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aswin.paul@monash.edu
#SBATCH --ntasks=99
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB

# Activate the project environment, then run from the repository root so the task
# paths in the .conf file resolve. Adjust these two lines for your cluster.
cd "${ADAPTIVE_BEHAVIOUR_ROOT:?set this to your clone of the repository}"
source .venv/bin/activate

srun --ntasks=99 -l --multi-prog adaptive_behaviour/pong/hpc/dpefe_file_10.conf
