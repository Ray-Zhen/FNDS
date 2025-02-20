#!/bin/bash
DATASET_NAME="icfg_peds"

CUDA_VISIBLE_DEVICES=2 \
python train.py \
--name train_icfg_with-lgloss_tsr_rp-0.90_m-0.4_tau-0.05_epo-25_p-5000_q-2048_loc_lr-0.005 \
--batch_size 64 \
--dataset_name $DATASET_NAME \
--loss_names 'itc' \
--loss_type 'fusion' \
--root_dir '/ai/teacher/mwt/datasets/' \
--num_epoch 25 \
--num_workers 24 \
--lr 1e-5 \
--temperature 0.05 \
--queue_size 2048 \
--reversed_p 0.90 \
--margin 0.4 \
--img_aug \
--txt_aug \
--fnm \
--weight_decay 1e-3 \
