#!/bin/bash

python FNN_main.py --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip/ --PATH /home/tianqinl/time-series-domain-adaptation/JDA/data_results/ --task 3Av2 --batch_size 30 --epochs 500


python FNN_main.py --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip/ --PATH /home/tianqinl/time-series-domain-adaptation/JDA/data_results/ --task 3Av2 --batch_size 100 --epochs 500


python FNN_main.py --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip/ --PATH /home/tianqinl/time-series-domain-adaptation/JDA/data_results/ --task 3E --batch_size 30 --epochs 500


python FNN_main.py --data_path /home/tianqinl/time-series-domain-adaptation/data_unzip/ --PATH /home/tianqinl/time-series-domain-adaptation/JDA/data_results/ --task 3E --batch_size 100 --epochs 500


