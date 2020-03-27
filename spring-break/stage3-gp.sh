#!/bin/bash

python stage3_local_gp.py --data_path /home/tianqinl --task 3E --epochs 200 --n_critic 4 --lbl_percentage 0.7 --seed 0 --model_save_period 5 --gpweight 20 --dlocal 0.05 --sclass 0.7 --batch_size 512 --save_path /home/tianqinl/time-series-domain-adaptation/train_related/spring_break