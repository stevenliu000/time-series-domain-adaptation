#!/bin/bash

python stage3.py --data_path /home/tianqinl --task 3E --epochs 200 --n_critic 4 --lbl_percentage 0.7 --seed 0 --model_save_period 5 --clip_value $1 --dglobal $2 --sclass 0.7 --batch_size 256 --save_path /home/tianqinl/time-series-domain-adaptation/train_related/spring_break