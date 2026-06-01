#!/bin/bash
#SBATCH --job-name=defect-qlora512
#SBATCH --account=imt526a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
mkdir -p outputs/slurm
source ~/.bashrc
conda activate /gpfs/projects/imt526a/jiufuzh/conda/envs/imt526
export WANDB_PROJECT=group8_defect_detection
export TOKENIZERS_PARALLELISM=false
python scripts/prepare_data.py --config configs/qlora_512.yaml
python scripts/profile_tokens.py --config configs/qlora_512.yaml --split train
python scripts/train_sft.py --config configs/qlora_512.yaml
python scripts/evaluate_llm.py --config configs/qlora_512.yaml --split validation --adapter outputs/qwen25_7b_qlora_512/final_adapter
