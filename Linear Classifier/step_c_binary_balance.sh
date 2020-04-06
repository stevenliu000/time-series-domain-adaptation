#!/bin/bash

echo $1
python step_c_binary_balance.py \
    --data_path /home/weixinli \
    --save_path /home/weixinli/time-series-domain-adaptation/train_related/3 \
    --epochs 500\
    --lbl_percentage 0.7 \
    --sclass 0.7 \
    --lr_FNN 1e-3\
    --lr_centerloss 5e-3 \
    --scent 1e-4 \
    --sbinary_loss $1\
    --epoch_begin_prototype 10
