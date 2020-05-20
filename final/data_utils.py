import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import random

def read_data(args, domain):
    '''
    domain: source or target
    '''
    assert domain in ['source', 'target']
    labeled_x_filename = 'processed_file_not_one_hot_%s_%1.1f_%s_known_label_x.npy'%(args.task, args.lbl_percentage, args.domain)
    labeled_y_filename = 'processed_file_not_one_hot_%s_%1.1f_%s_known_label_y.npy'%(args.task, args.lbl_percentage, args.domain)
    unlabeled_x_filename = 'processed_file_not_one_hot_%s_%1.1f_%s_unknown_label_x.npy'%(args.task, args.lbl_percentage, args.domain)
    unlabeled_y_filename = 'processed_file_not_one_hot_%s_%1.1f_%s_unknown_label_y.npy'%(args.task, args.lbl_percentage, args.domain)
    labeled_x = np.load(os.path.join(args.data_path, labeled_x_filename))
    labeled_y = np.load(os.path.join(args.data_path, labeled_y_filename))
    unlabeled_x = np.load(os.path.join(args.data_path, unlabeled_x_filename))
    unlabeled_y = np.load(os.path.join(args.data_path, unlabeled_y_filename))

    return labeled_x, labeled_y, unlabeled_x, unlabeled_y

class JoinDataset(Dataset):
    def __init__(self, source_x, source_y, target_x, target_y, random=False):
        self.source_x = source_x
        self.source_y = source_y
        self.target_x = target_x
        self.target_y = target_y
        
        self.source_len = self.source_y.shape[0]
        self.target_len = self.target_y.shape[0]
    
        self.random = random
    def __len__(self):
        return self.target_len
    
    def __getitem__(self, index):
        if self.random:
            index_source = random.randrange(self.source_len)
            index_target = random.randrange(self.target_len)
        else:
            index_source = index
            index_target = index

        return (self.source_x[index_source], self.source_y[index_source]), (self.target_x[index_target], self.target_y[index_target])
    
    
class SingleDataset(Dataset):
    def __init__(self, x, y):
            self.x = x
            self.y = y
            self.len = self.y.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


def get_source_dict(file_path, num_class, data_len=None, seed=0):
    '''
    input:
        data_len: keep how many data points

    output:
        {class: [data]},
        data_len
    '''
    data_ = np.load(file_path, allow_pickle=True)
    train_data = data_['tr_data']
    train_lbl = data_['tr_lbl']
    np.random.seed(seed)
    index = np.random.permutation(train_data.shape[0])
    train_data = train_data[index]
    train_lbl = train_lbl[index]
    if data_len:
        train_data = data_['tr_data'][:data_len]
        train_lbl = data_['tr_lbl'][:data_len]
    data_dict = get_class_data_dict(train_data, train_lbl, num_class)

    return data_dict, train_data.shape[0]

def get_target_dict(file_path, num_class, lbl_percentage, seed=0):
    '''
    split target domain data

    output:
        with label dict:
            {class: [data]}
        without label:
            [data], [lbl]
        with label:
            [data], [lbl]
        data_len
    '''
    data_ = np.load(file_path, allow_pickle=True)
    train_data = data_['te_data']
    train_lbl = data_['te_lbl']

    np.random.seed(seed)
    index = np.random.permutation(train_data.shape[0])
    train_data = train_data[index]
    train_lbl = train_lbl[index]

    with_label = {i:[] for i in range(num_class)}
    labeled_index = []
    for i in with_label:
        index = np.argwhere(train_lbl==i).flatten()
        np.random.seed(seed)
        index = np.random.choice(index, int(lbl_percentage*train_lbl.shape[0]/num_class))
        labeled_index.extend(index)
        with_label[i] = train_data[index]

    return with_label, (np.delete(train_data,labeled_index,axis=0), np.delete(train_lbl,labeled_index,axis=0)), (train_data[labeled_index], train_lbl[labeled_index]), train_data.shape[0]

def get_class_data_dict(data, lbl, num_class):
    '''
    construct a dict {label: data}
    '''
    lbl_not_one_hot = lbl
    result = {i:[] for i in range(num_class)}
    for i in result:
        index = np.argwhere(lbl_not_one_hot==i).flatten()
        result[i] = data[index]

    return result

def get_batch_source_data_on_class(class_dict, num_per_class):
    '''
    get batch from source data given a required number of sample per class
    '''
    batch_x = []
    batch_y = []
    keys = [key for key in class_dict]
    keys.sort()
    for key in keys:
        index = random.sample(range(len(class_dict[key])), num_per_class)
        batch_x.extend(class_dict[key][index])
        batch_y.extend([key] * num_per_class)
    # for key, value in class_dict.items():
    #     index = random.sample(range(len(value)), num_per_class)
    #     batch_x.extend(value[index])
    #     batch_y.extend([key] * num_per_class)

    return np.array(batch_x), np.array(batch_y)

def get_batch_target_data_on_class(real_dict, num_per_class, pesudo_dict, unlabel_data=None, real_weight=1, pesudo_weight=0.1, no_pesudo=False):
    '''
    get batch from target data given a required number of sample per class

    if totoal number sample in this class is less than the required number of sample
    then fetch the remainding data duplicatly from the labeled set
    '''
    batch_x = []
    batch_y = []
    batch_weight = []
    keys = [key for key in real_dict]
    keys.sort()
    for key in keys:
        real_num = len(real_dict[key])
        pesudo_num = len(pesudo_dict[key])
        num_in_class = real_num + pesudo_num

        if no_pesudo:
            # known label data only
            index = random.sample(range(100000000000), num_per_class)
            index = [i % real_num for i in index]
            batch_x.extend(real_dict[key][index])
            batch_y.extend([key] * num_per_class)
            batch_weight.extend([real_weight] * num_per_class)

        else:
            not_enough_pesudo = num_per_class - pesudo_num
            if not_enough_pesudo <= 0:
                # have enough label data in this class
                index = np.random.permutation(pesudo_num)[:num_per_class]
                batch_x.extend(pesudo_dict[key][index])
                batch_y.extend([key] * num_per_class)
                batch_weight.extend([pesudo_weight] * num_per_class)
            elif pesudo_num == 0:
                # no pesudo label at this class
                # use all known label data
                index = random.sample(range(100000000000), num_per_class)
                index = [i % real_num for i in index]
                batch_x.extend(real_dict[key][index])
                batch_y.extend([key] * num_per_class)
                batch_weight.extend([real_weight] * num_per_class)
            else:
                # have pesudo label at this class, but not enough
                # use known label data to fill the rest
                index_pesudo = np.random.permutation(pesudo_num)
                batch_x.extend(pesudo_dict[key][index_pesudo])
                batch_weight.extend([pesudo_weight] * pesudo_num)
                index_known = random.sample(range(100000000000), not_enough_pesudo)
                index_known = [i % real_num for i in index_known]
                batch_x.extend(real_dict[key][index_known])
                batch_weight.extend([real_weight] * not_enough_pesudo)
                batch_y.extend([key] * num_per_class)
                

    return np.array(batch_x), np.array(batch_y), np.array(batch_weight)
