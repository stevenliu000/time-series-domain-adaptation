#!/bin/bash

#python FNN_main.py --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip/ --PATH /home/tianqinl/time-series-domain-adaptation/JDA/Final_data_results/ --task 3Av2 --batch_size 400 --epochs 150
#
#
python FNN_main.py --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip/ --PATH /home/tianqinl/time-series-domain-adaptation/spring-break/FNN_data_results/ --task 3E --batch_size 400 --epochs 150 --job_type source 2>&1 | tee tmp/log/source.fnn.system.out





