#!/bin/bash

echo $1
python step_c_endtoend.py \
    --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip \
    --save_path /home/tianqinl/time-series-domain-adaptation/train_related \
    --sprototype $1 \
    --epochs 500\
    --lbl_percentage 0.7 \
    --sclass $2 \
    --lr_FNN 1e-3\
    --epoch_begin_prototype 0\
    --lr_centerloss 5e-3 \
    --scent 1e-4 \

    # sclass optimal is 0.7
