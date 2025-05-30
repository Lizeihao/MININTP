#!/bin/bash
source activate conda/envs/minintp
cd home/.../MININTP

# input your api key first
export WANDB_API_KEY=""

CHECK_PATH="./output/pretrain/miniNTP-0/tokenizer.json"
if [ ! -f "$CHECK_PATH" ]; then
    echo "start train miniNTP-0"
    deepspeed --master_port 27500 --include localhost:0,1,2,3,4,5,6,7 ./trainer/ntp_pretrain.py \
        --deepspeed_config ./deepspeed_config/ds_config.json \
        --max_train_seq_len 512 \
        --hidden_size 512 \
        --num_hidden_layers 8 \
        --num_attention_heads 8 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-4 \
        --data_size 1.0 \
        --epochs 2 \
        --log_interval 16 \
        --save_interval 5000 \
        --model_number 0
else
    echo "the training of miniNTP-0 is ok"
fi