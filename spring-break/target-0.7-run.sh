#!/bin/bash


python FNN_main.py --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip/spring_break_data/ \
    --train_file train_target-with-label-0.7-{}.pkl \
    --vali_file validation_target-with-label-0-7-{}.pkl \
    --PATH /home/tianqinl/time-series-domain-adaptation/spring-break/FNN_data_results/ \
    --model_save_steps 3 \
    --task 3E \
    --batch_size 400 \
    --epochs 150 \
    --job_type target-07 \
    2>&1 | tee tmp/log/target-07.fnn.system.out
