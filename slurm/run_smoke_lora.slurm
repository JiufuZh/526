#!/bin/bash
#SBATCH --job-name=defect-smoke-lora
#SBATCH --account=imt526a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
mkdir -p outputs/slurm
source ~/.bashrc
conda activate /gpfs/projects/imt526a/jiufuzh/conda/envs/imt526
export WANDB_PROJECT=group8_defect_detection
export TOKENIZERS_PARALLELISM=false
python scripts/prepare_data.py --config configs/smoke_lora_512.yaml
python scripts/train_sft.py --config configs/smoke_lora_512.yaml
