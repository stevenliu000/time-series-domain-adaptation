import numpy as np
import random

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
    for key, value in class_dict.items():
        index = random.sample(range(len(value)), num_per_class)
        batch_x.extend(value[index])
        batch_y.extend([key] * num_per_class)

    return np.array(batch_x), np.array(batch_y)

def get_batch_target_data_on_class(real_dict, pesudo_dict, unlabel_data, num_per_class, real_weight=1, pesudo_weight=0.1):
    '''
    get batch from target data given a required number of sample per class

    if totoal number sample in this class is less than the required number of sample
    then fetch the remainding data duplicatly from the labeled set
    '''
    batch_x = []
    batch_y = []
    batch_weight = []
    for key in real_dict:
        real_num = len(real_dict[key])
        pesudo_num = len(pesudo_dict[key])
        num_in_class = real_num + pesudo_num

        if num_in_class < num_per_class:
            # if totoal number sample in this class is less than the required number of sample
            # then fetch the remainding data duplicatly from the labeled set
            
            num_fetch_unlabeled = (num_in_class - num_per_class)
            index = np.random.choice(real_num, num_fetch_unlabeled)
            batch_x.extend(real_dict[key][index])
            batch_y.extend([key] * num_fetch_unlabeled)
            batch_weight.extend([real_weight] * num_fetch_unlabeled)

            batch_x.extend(real_dict[key])
            batch_weight.extend([real_weight] * real_num)
            batch_x.extend(pesudo_dict[key])
            batch_weight.extend([pesudo_weight] * pesudo_num)
            batch_y.extend([key] * num_in_class)

        else:
            index = random.sample(range(num_in_class), num_per_class)
            index_in_real = []
            index_in_pesudo = []
            for i in index:
                if i >= real_num:
                    index_in_pesudo.append(i-real_num)
                else:
                    index_in_real.append(i)

            batch_x.extend(real_dict[key][index_in_real])
            batch_weight.extend([real_weight] * len(index_in_real))
            batch_x.extend(pesudo_dict[key][index_in_pesudo])
            batch_weight.extend([pesudo_weight] * len(index_in_pesudo))
            batch_y.extend([key] * num_per_class)

    return np.array(batch_x), np.array(batch_y), np.array(batch_weight)
