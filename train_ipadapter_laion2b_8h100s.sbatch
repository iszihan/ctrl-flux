#!/bin/bash
#SBATCH --job-name=ipflux_laion2b_8h100s            # change if needed
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64  # 8 GPUs × 8 workers = 64 workers + overhead
#SBATCH --mem=128G  # increased from 48G to prevent CPU OOM
#SBATCH --time=48:00:00
#SBATCH --output=/project/aip-jacobson/zling/ctrl-flux/logs/%x_%j.out
#SBATCH --error=/project/aip-jacobson/zling/ctrl-flux/logs/%x_%j.err
#SBATCH --chdir=/project/aip-jacobson/zling/ctrl-flux

echo "Job started on $(hostname)"
echo "PWD: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Start background memory monitor (logs every 30s)
(while true; do
    echo "======== $(date '+%Y-%m-%d %H:%M:%S') ========"
    echo "CPU Memory: $(free -h | grep Mem | awk '{print $3"/"$2}')"
    echo "GPU Memory:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %sMiB/%sMiB (util: %s%%)\n", $1, $2, $3, $4}'
    echo ""
    sleep 30
done) &
MONITOR_PID=$!

source ~/.bashrc
source venv/bin/activate

export CONFIG_PATH=./train/configs/ipadapter_laion2b.yaml

accelerate launch \
    --num_processes 8 \
    --main_process_port 41353 \
    -m train_ip_adapter_flux expname=ipflux-laion2b-8h100s

# Cleanup: kill memory monitor
kill $MONITOR_PID 2>/dev/null
