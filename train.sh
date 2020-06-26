#!/bin/bash

python main.py --img_path "../GTTS/Samples" \
--train \
--train_csv_file "../GTTS/Labels/pq_labels_all.csv" \
--val_csv_file "../GTTS/Labels/pq_labels_val.csv" \
--conv_base_lr 3e-4 \
--dense_lr 3e-3 \
--decay \
--ckpt_path ../checkpoints \
--epochs 100 \
--early_stopping_patience 10
