!#/bin/bash


python JDA-gan.py --data_path /home/weixinli/ \
                  --task 3A \
                  --batch_size 500 \
                  --epochs 800 \
                  --gap 4 \
                  --lbl_percentage 0 \
                  --num_per_class -1 \
                  --classifier /home/weixinli/time-series-domain-adaptation/JDA/FNN_trained_model \
                  --save_path ../train_related/JDA_GAN \
                  --model_save_period 5

python JDA-gan.py --data_path /home/weixinli/ \
                  --task 3A \
                  --batch_size 500 \
                  --epochs 800 \
                  --gap 4 \
                  --lbl_percentage 0.2 \
                  --num_per_class -1 \
                  --classifier /home/weixinli/time-series-domain-adaptation/JDA/FNN_trained_model \
                  --save_path ../train_related/JDA_GAN \
                  --model_save_period 5

python JDA-gan.py --data_path /home/weixinli/ \
                  --task 3A \
                  --batch_size 500 \
                  --epochs 800 \
                  --gap 4 \
                  --lbl_percentage 0.5 \
                  --num_per_class -1 \
                  --classifier /home/weixinli/time-series-domain-adaptation/JDA/FNN_trained_model \
                  --save_path ../train_related/JDA_GAN \
                  --model_save_period 5

python JDA-gan.py --data_path /home/weixinli/ \
                  --task 3A \
                  --batch_size 500 \
                  --epochs 800 \
                  --gap 4 \
                  --lbl_percentage 0.7 \
                  --num_per_class -1 \
                  --classifier /home/weixinli/time-series-domain-adaptation/JDA/FNN_trained_model \
                  --save_path ../train_related/JDA_GAN \
                  --model_save_period 5

python JDA-gan.py --data_path /home/weixinli/ \
                  --task 3A \
                  --batch_size 500 \
                  --epochs 800 \
                  --gap 4 \
                  --lbl_percentage 1 \
                  --num_per_class -1 \
                  --classifier /home/weixinli/time-series-domain-adaptation/JDA/FNN_trained_model \
                  --save_path ../train_related/JDA_GAN \
                  --model_save_period 5

python JDA-gan.py --data_path /home/weixinli/ \
                  --task 3E \
                  --batch_size 650 \
                  --epochs 800 \
                  --gap 4 \
                  --lbl_percentage 0 \
                  --num_per_class -1 \
                  --classifier /home/weixinli/time-series-domain-adaptation/JDA/FNN_trained_model \
                  --save_path ../train_related/JDA_GAN \
                  --model_save_period 5

python JDA-gan.py --data_path /home/weixinli/ \
                  --task 3E \
                  --batch_size 650 \
                  --epochs 800 \
                  --gap 4 \
                  --lbl_percentage 0.2 \
                  --num_per_class -1 \
                  --classifier /home/weixinli/time-series-domain-adaptation/JDA/FNN_trained_model \
                  --save_path ../train_related/JDA_GAN \
                  --model_save_period 5

python JDA-gan.py --data_path /home/weixinli/ \
                  --task 3E \
                  --batch_size 650 \
                  --epochs 800 \
                  --gap 4 \
                  --lbl_percentage 0.5 \
                  --num_per_class -1 \
                  --classifier /home/weixinli/time-series-domain-adaptation/JDA/FNN_trained_model \
                  --save_path ../train_related/JDA_GAN \
                  --model_save_period 5

python JDA-gan.py --data_path /home/weixinli/ \
                  --task 3E \
                  --batch_size 650 \
                  --epochs 800 \
                  --gap 4 \
                  --lbl_percentage 0.7 \
                  --num_per_class -1 \
                  --classifier /home/weixinli/time-series-domain-adaptation/JDA/FNN_trained_model \
                  --save_path ../train_related/JDA_GAN \
                  --model_save_period 5

python JDA-gan.py --data_path /home/weixinli/ \
                  --task 3E \
                  --batch_size 650 \
                  --epochs 800 \
                  --gap 4 \
                  --lbl_percentage 1 \
                  --num_per_class -1 \
                  --classifier /home/weixinli/time-series-domain-adaptation/JDA/FNN_trained_model \
                  --save_path ../train_related/JDA_GAN \
                  --model_save_period 5

