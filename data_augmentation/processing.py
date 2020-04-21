import argparse
import os
import numpy as np
# Parameters
parser = argparse.ArgumentParser(description='processing agumentated data')
parser.add_argument("--data_path", type=str, default="../train_related/asd/", help="augmented dataset folder path")
parser.add_argument('--subfolders', type=str, nargs="+", help='where to store data')
parser.add_argument('--task', type=str, default="source_labeled", help="source or target;labeled or unlabeled")
parser.add_argument('--subset_num', type=int, default=20, default="subset number in each class in order to generate new data")
parser.add_argument('--save_path', type=str, default = "../data_unzip/data_augment", help='where to store data')

args = parser.parse_args()

save_folder = os.path.join(args.save_path, args.task)
os.makedir(save_folder, exist_ok=True)

new_x = []
new_y = []
for i in args.subfolders:
    data_path_x = os.path.join(args.data_path, i, 'new_data_x.npy')
    data_path_y = os.path.join(args.data_path, i, 'new_data_y.npy')
    data_x = np.load(data_path_x)
    data_y = np.load(data_path_y)
    new_x.append(data_x)
    new_y.append(data_y)

new_data_x = np.concatenate(new_x, axis=0)
new_data_y = np.concatenate(new_y, axis=0)

# save 
np.save(os.path.join(args.save_path, args.task, )
