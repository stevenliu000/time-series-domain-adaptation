#!/bin/bash

echo $1
python step_c_wot_sourceFNN.py --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip --save_path /home/tianqinl/time-series-domain-adaptation/train_related --sprototype $1 --epochs 200 --lbl_percentage 0.7 --select_pretrain_epoch 55
