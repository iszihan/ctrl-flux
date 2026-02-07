#!/bin/bash
# Run this script inside an interactive srun session with 4 GPUs, e.g.:
#   srun --account=aip-jacobson --partition=gpubase_h --gres=gpu:h100:4 --mem=96G --cpus-per-task=32 --time=8:00:00 --chdir=/project/aip-jacobson/zling/ctrl-flux --pty bash
# Then: ./run_train_laion2b_bs8_interactive.sh

set -e
cd /project/aip-jacobson/zling/ctrl-flux

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

# Batch size 8, lr 2e-4 for effective batch size
accelerate launch \
    --num_processes 2 \
    --main_process_port 41353 \
    train_ip_adapter_flux.py \
    expname=ipflux-laion2b-4h100-nonzeroinit-debug \
    resume_from_checkpoint=true \
    resume_path=runs/ipflux-laion2b-4l40s-nonzeroinit/ip_adapter-040000 \
    train.batch_size=4 \
    dataset.train_prefetch_factor=2 \
    train.text_encoder_offload=false \
    train.optimizer.lr=2e-4

# Cleanup: kill memory monitor
kill $MONITOR_PID 2>/dev/null
