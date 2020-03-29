#!/bin/bash

python stepb.py \
    --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip \
    --save_path /home/tianqinl/time-series-domain-adaptation/train_related \
    --epochs 300\
    --lbl_percentage 0.7 \
    --sclass 0.7 \
    --lr_FNN 1e-3\
    --lr_centerloss 5e-3 \
    --scent 1e-4 \

    # sclass optimal is 0.7
