#!/bin/bash

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

# Increase batch size from 4 to 8 
# Increase learning rate from 1e-4 to 2e-4 to account for double effective batch size 
# Dataloader num workers and prefetch factor stay the same for now.
accelerate launch \
    --num_processes 2 \
    --main_process_port 41353 \
    train_ip_adapter_flux.py \
    expname=ipflux-laion2b-4h100-nonzeroinit-b8-lr2e-4 \
    resume_from_checkpoint=true \
    resume_path=runs/ipflux-laion2b-4l40s-nonzeroinit/ip_adapter-040000 \
    train.batch_size=8 \
    dataset.train_prefetch_factor=2 \
    train.optimizer.lr=2e-4

# Cleanup: kill memory monitor
kill $MONITOR_PID 2>/dev/null
