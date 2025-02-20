#!/bin/bash
DATASET_NAME="rstp_reid"

CUDA_VISIBLE_DEVICES=5 \
python train.py \
--name train_rstp_itc_with-fnr_label_smooth \
--img_aug \
--txt_aug \
--batch_size 64 \
--dataset_name $DATASET_NAME \
--loss_names 'itc' \
--root_dir '/vol8/home/std_zrg/dataset/' \
--num_epoch 20 \
--num_workers 12 \
--lr 1e-5 \
--temperature 0.05 \
--queue_size 2048 \
--weight_decay 1e-3 \
--reversed_p 0.8 \
--distill \
--alpha_distill 0.2 \
--fnm \
