#!/bin/bash
#SBATCH --job-name=defect-cpu-baselines
#SBATCH --account=imt526a
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
mkdir -p outputs/slurm
source ~/.bashrc
conda activate /gpfs/projects/imt526a/jiufuzh/conda/envs/imt526
python scripts/prepare_data.py --config configs/default.yaml
python scripts/run_baselines.py --config configs/default.yaml --eval-split validation
