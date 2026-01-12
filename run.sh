#!/usr/bin/env bash

export MID_DATA_DIR="./data/mid_data"
export RAW_DATA_DIR="./data/raw_data"
export OUTPUT_DIR="./out"

export GPU_IDS="2"
export BERT_TYPE="roberta_wwm"  # roberta_wwm / roberta_wwm_large / uer_large
export BERT_DIR="/home/ubun/Jax/transformer/chinese-roberta"
# export BERT_TYPE="bert-base-chinese"  # roberta_wwm / roberta_wwm_large / uer_large
# export BERT_DIR="/home/ubuntu/Jax/transformer/bert-base-chinese"

export MODE="train"
# export MODE="stack"
export TASK_TYPE="crf"

python3.6 main_maml.py \
--gpu_ids=$GPU_IDS \
--output_dir=$OUTPUT_DIR \
--mid_data_dir=$MID_DATA_DIR \
--mode=$MODE \
--task_type=$TASK_TYPE \
--raw_data_dir=$RAW_DATA_DIR \
--bert_dir=$BERT_DIR \
--bert_type=$BERT_TYPE \
--train_epochs=500 \
--swa_start=5 \
--attack_train="" \
--train_batch_size=60 \
--dropout_prob=0.1 \
--max_seq_len=64 \
--lr=2e-5 \
--other_lr=2e-3 \
--seed=123 \
--weight_decay=0.01 \
--loss_type='ls_ce' \
--eval_model \
--task_num=5 \
--meta_lr=5e-5 \
--update_lr=2e-5 \
--update_step=5 \
--update_step_test=5 \
--T=5 \
--alpha=0.2 \
--shot=10 \
--query=15 \
#--use_fp16