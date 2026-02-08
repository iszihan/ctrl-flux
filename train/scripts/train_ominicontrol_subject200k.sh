export CONFIG_PATH=./train/configs/ominicontrol_subject200k.yaml
accelerate launch --num_processes 1 --main_process_port 41353 -m train_ominicontrol_lora_flux