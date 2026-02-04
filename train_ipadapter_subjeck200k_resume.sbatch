#!/bin/bash
#SBATCH --job-name=ipflux4_resume            # change if needed
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/project/aip-jacobson/zling/ctrl-flux/logs/%x_%j.out
#SBATCH --error=/project/aip-jacobson/zling/ctrl-flux/logs/%x_%j.err
#SBATCH --chdir=/project/aip-jacobson/zling/ctrl-flux

echo "Job started on $(hostname)"
echo "PWD: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source ~/.bashrc
source venv/bin/activate

export CONFIG_PATH=./train/configs/ipadapter_subject200k_resume.yaml

accelerate launch \
    --num_processes 4 \
    --main_process_port 41353 \
    -m train_ip_adapter_flux expname=ipflux-4gpu-resume
