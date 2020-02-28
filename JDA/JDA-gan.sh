!#/bin/bash

python JDA-gan.py --data_path /home/weixinli/ \
		  --task 3A \
		  --batch_size 400 \
                  --epochs 10 \
                  --gap 4 \
                  --lbl_percentage 0.2 \
                  --num_per_class -1 \
                  --classifier /home/weixinli/time-series-domain-adaptation/JDA/FNN_trained_model \
                  --save_path ../train_related/JDA_GAN \
                  --model_save_period 5 
