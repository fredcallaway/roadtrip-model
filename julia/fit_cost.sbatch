#!/usr/bin/env bash
#SBATCH --job-name=fit_cost
#SBATCH --output=out/%A_%a
#SBATCH --array=1-101
#SBATCH --time=240
#SBATCH --mem-per-cpu=6000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-user=flc2@princeton.edu

module load julia/1.0.0
julia run_fit_cost.jl $SLURM_ARRAY_TASK_ID