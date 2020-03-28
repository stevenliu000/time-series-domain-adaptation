#!/bin/bash

python FNN_main.py --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip/spring_break_data/ \
    --train_file train_target-with-label-0.$1-{}.pkl \
    --vali_file validation_target-with-label-0-$1-{}.pkl \
    --test_file target-without-label-0.$1-{}.pkl \
    --PATH /home/tianqinl/time-series-domain-adaptation/spring-break/FNN_data_results/ \
    --model_save_steps 3 \
    --task 3E \
    --batch_size 400 \
    --epochs $2 \
    --job_type target-0$1 \
    2>&1 | tee tmp/log/target-0$1-e$2.fnn.system.out
