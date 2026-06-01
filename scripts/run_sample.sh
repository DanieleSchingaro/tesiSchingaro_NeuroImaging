#!/bin/bash
#SBATCH --job-name=sample_ldm
#SBATCH --output=logs/sample_ldm_%j.log
#SBATCH --error=logs/sample_ldm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
 
source /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro/.venv/bin/activate
cd /mnt/data/home-ubuntu/work/medical-3D-Rflow-Maisi-Schingaro
 
# numero di campioni da generare (default 100, sovrascrivibile da CLI)
N_SAMPLES=${1:-100}
 
# stampa info ambiente
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "GPU disponibili: $(nvidia-smi --list-gpus | wc -l)"
echo "Campioni da generare: $N_SAMPLES"
 
MASTER_PORT=$((29000 + RANDOM % 2000))
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT -m src.inference.sample --n_samples $N_SAMPLES
 
echo "End: $(date)"