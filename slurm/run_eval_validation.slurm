#!/bin/bash
#SBATCH --job-name=defect-eval-val
#SBATCH --account=imt526a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
mkdir -p outputs/slurm
source ~/.bashrc
conda activate /gpfs/projects/imt526a/jiufuzh/conda/envs/imt526
export TOKENIZERS_PARALLELISM=false
python scripts/evaluate_llm.py --config configs/qlora_512.yaml --split validation --adapter outputs/qwen25_7b_qlora_512/final_adapter --mode logprob
