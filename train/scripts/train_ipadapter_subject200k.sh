export CONFIG_PATH=./train/configs/ipadapter_subject200k.yaml
accelerate launch --num_processes 1 --main_process_port 41353 -m train_ip_adapter_flux