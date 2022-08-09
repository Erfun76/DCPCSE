#!/bin/bash

# In this example, we show how to train DCPCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

CUDA_VISIBLE_DEVICES=5 python3 train.py \
    --model_name_or_path ./models/zibert256 \
    --train_file ./data/dump.txt \
    --output_dir result/unsup-dcpcse-zibert256-base-avg-first-last\
    --num_train_epochs 8 \
    --per_device_train_batch_size 256 \
    --learning_rate 3e-2 \
    --max_seq_length 32 \
    --evaluation_strategy no \
    --eval_steps 4000 \
    --pooler_type avg_first_last \
    --mlp_only_train \
    --pre_seq_len 16 \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
