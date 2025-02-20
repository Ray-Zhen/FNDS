#!/bin/bash
DATASET_NAME="cuhk_peds"

CUDA_VISIBLE_DEVICES=3 \
python train.py \
--name train_cuhk_with-lgloss_tsr_rp-0.90_m-0.4_tau-0.03_epo-30_p-5000 \
--batch_size 64 \
--dataset_name $DATASET_NAME \
--loss_names 'itc' \
--loss_type 'local' \
--root_dir '/ai/teacher/mwt/datasets/' \
--num_epoch 30 \
--num_workers 24 \
--lr 1e-5 \
--temperature 0.03 \
--queue_size 6144 \
--reversed_p 0.90 \
--margin 0.4 \
--img_aug \
--txt_aug \
--fnm \
