#!/bin/bash
#SBATCH --job-name=eval_ldm
#SBATCH --output=logs/eval_ldm_%j.log
#SBATCH --error=logs/eval_ldm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro

# sorgente reali: "test" (102, default) oppure "all" (1007)
REAL_SOURCE=${1:-test}

echo "Start: $(date)"
echo "Reali di riferimento: $REAL_SOURCE"

# la valutazione usa 1 sola GPU (niente torchrun): FID/MMD/MS-SSIM non sono in DDP
python3 -m src.evaluation.eval --real_source $REAL_SOURCE

echo "End: $(date)"