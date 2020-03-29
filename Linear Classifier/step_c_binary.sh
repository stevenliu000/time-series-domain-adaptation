#!/bin/bash

echo $1
python step_c_prototype.py \
    --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip \
    --save_path /home/tianqinl/time-series-domain-adaptation/train_related \
    --epochs 500\
    --lbl_percentage 0.7 \
    --sclass 0.7 \
    --lr_FNN 1e-3\
    --lr_centerloss 5e-3 \
    --scent 1e-4 \
    --sbinary_loss $1\
