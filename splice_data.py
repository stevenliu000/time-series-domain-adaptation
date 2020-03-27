import numpy as np
import pickle
import os

import argparse


parser = argparse.ArgumentParser(description='Time series adaptation: Split data')
parser.add_argument("-n", type=int, default=1, help="split data into train and validate")
parser.add_argument("--data_prop", type=float, default=0.1, help="split proportion")
parser.add_argument("--path_train", type=str, default = "data_unzip/processed_file_{}.pkl", help='source data path')
parser.add_argument("--output_folder", type=str, default="/home/tianqinl/time-series-domain-adaptation/data_unzip/")

dataset = ["3Av2", "3E"]
args = parser.parse_args()
path_train = args.path_train
dataname = dataset[int(args.n)]
datafilename = path_train.format(dataname)
data_ = np.load(datafilename, allow_pickle=True)

name_class = 50 if dataname == '3Av2' else 65
vali_num = 5 if dataname == '3Av2' else 10
#vali_prop = float(args.data_prop)

tmp = {i:[] for i in range(name_class)}

np.random.seed(0)

train_data = data_['tr_data']
train_lbl = data_['tr_lbl']
#vali_num = np.round(train_data.shape[0] * vali_prop).astype(int)
print(vali_num)

index = np.random.permutation(train_data.shape[0])
print(index.shape, train_lbl.shape)
train_data = train_data[index, :]
train_lbl = train_lbl[index, :]
lbl = np.argmax(train_lbl, axis = 1)

train_data_ = []
train_lbl_ = []

for i in range(len(lbl)):
    if len(tmp[lbl[i]]) < vali_num:
        tmp[lbl[i]].append(train_data[i,:])
    else:
        train_data_.append(train_data[i,:])
        train_lbl_.append(train_lbl[i,:])

vali_data = []
vali_lbl = []

for key, value in tmp.items():
    vali_data.extend(value)
    one_hot_lbl = np.zeros(name_class)
    one_hot_lbl[key] = 1.
    vali_lbl.extend([one_hot_lbl.astype('float32')] * vali_num)

np.random.seed(0)
index = np.random.permutation(vali_num * name_class)
vali_data = np.array(vali_data)
vali_lbl = np.array(vali_lbl)
print(vali_data.shape, vali_lbl.shape)

vali_data = vali_data[index,:]
vali_lbl = vali_lbl[index,:]
train_data_ = np.array(train_data_)
train_lbl_ = np.array(train_lbl_)

print(train_data_.shape, train_lbl_.shape)

vali_data_dict = {'tr_data': vali_data, 'tr_lbl': vali_lbl}
train_data_dict = {'tr_data': train_data_, 'tr_lbl': train_lbl_}


save_vali_path = os.path.join(args.output_folder, "validation_{}.pkl".format("-".join(datafilename.split("/")[-1].split(".")[:-1])))
print("-".join(datafilename.split("/")[-1].split(".")[:-1]))
print(datafilename)



print(save_vali_path)

save_train_path = os.path.join(args.output_folder, "train_{}.pkl".format(".".join(datafilename.split("/")[-1].split(".")[:-1])))
print(save_train_path)

f = open(save_vali_path,"wb")
pickle.dump(vali_data_dict, f)
f.close()
f = open(save_train_path,"wb")
pickle.dump(train_data_dict, f)
f.close()

#f = open("data_unzip/split_{}_validation_{}.pkl".format(vali_prop, dataname),"wb")
#pickle.dump(vali_data_dict, f)
#f.close()
#f = open("data_unzip/split_{}_train_{}.pkl".format(vali_prop, dataname),"wb")
#pickle.dump(train_data_dict, f)
#f.close()

