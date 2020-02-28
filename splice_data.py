import numpy as np
import pickle

import argparse


parser = argparse.ArgumentParser(description='Time series adaptation: Split data')
parser.add_argument("-n", type=int, default=0, help="split data into train and validate")
path_train = "data_unzip/processed_file_{}.pkl"
dataset = ["3Av2", "3E"]
args = parser.parse_args()

dataname = dataset[int(args.n)]
data_ = np.load(path_train.format(dataname), allow_pickle=True)

name_class = 50 if dataname == '3Av2' else 65
vali_num = 2 if dataname == '3Av2' else 4

tmp = {i:[] for i in range(name_class)}

np.random.seed(0)

train_data = data_['tr_data']
train_lbl = data_['tr_lbl']

index = np.random.permutation(train_data.shape[0])

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


vali_data_dict = {'tr_data': vali_data, 'tr_lbl': vali_lbl}
train_data_dict = {'tr_data': train_data_, 'tr_lbl': train_lbl_}


f = open("data_unzip/validation_{}.pkl".format(dataname),"wb")
pickle.dump(vali_data_dict, f)
f.close()
f = open("data_unzip/train_{}.pkl".format(dataname),"wb")
pickle.dump(train_data_dict, f)
f.close()

