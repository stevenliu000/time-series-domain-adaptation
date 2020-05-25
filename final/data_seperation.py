import numpy as np
import os
import random
import argparse

parser = argparse.ArgumentParser(description='Data Seperation')
parser.add_argument("--data_path", type=str, required=True, help="dataset path")
parser.add_argument("--save_path", type=str, required=True, help="save path")
parser.add_argument("--task", type=str, required=True, help='3Av2 or 3E')
parser.add_argument("--percentage", nargs= '+', type=float, required=True)
args = parser.parse_args()

data_path = args.data_path
save_path = args.save_path
task = args.task
percentage = args.percentage
assert task in ['3Av2', '3E']
assert all(i > 0 for i in percentage)
assert all(i < 1 for i in percentage)

# Load data
seed = 0
file_name = "processed_file_%s.pkl"%task

total_data = np.load(os.path.join(data_path, file_name), allow_pickle=True)
source_all_x = total_data['tr_data']
source_all_y = total_data['tr_lbl']
target_all_x = total_data['te_data']
target_all_y = total_data['te_lbl']

source_all_y_not_one_hot = np.argmax(source_all_y,axis=1)
target_all_y_not_one_hot = np.argmax(target_all_y,axis=1)

# shuffle data
np.random.seed(seed)
shuffled_indice_source = np.random.permutation(source_all_x.shape[0])
np.random.seed(seed)
shuffled_indice_target = np.random.permutation(target_all_x.shape[0])

source_all_x_shuffled = source_all_x[shuffled_indice_source]
source_all_y_shuffled = source_all_y_not_one_hot[shuffled_indice_source]
target_all_x_shuffled = target_all_x[shuffled_indice_target]
target_all_y_shuffled = target_all_y_not_one_hot[shuffled_indice_target]

# Construct class dict
def get_class_data_dict(data, lbl, num_class):
    '''
    construct a dict {label: data}
    '''
    lbl_not_one_hot = lbl
    result = {i:[] for i in range(num_class)}
    for label in result:
        index = np.argwhere(lbl_not_one_hot==label)[:,0] # np.argwhere return (num_lbl_in_this_class, 1)
        result[label] = data[index]

    return result

source_dict = get_class_data_dict(source_all_x_shuffled, source_all_y_shuffled, source_all_y.shape[1])
target_dict = get_class_data_dict(target_all_x_shuffled, target_all_y_shuffled, target_all_y.shape[1])

# Separate Percentage Function
def separate_data_by_percentage(data_dict, known_percentage):
    unknown_label_x = []
    unknown_label_y = []
    known_label_x = []
    known_label_y = []
    for label, data in data_dict.items():
        num_in_this_label = data.shape[0]
        known_num_in_this_label = int(known_percentage * num_in_this_label)
        unknown_num_in_this_label = num_in_this_label - known_num_in_this_label
        np.random.seed(seed)
        known_indices = np.random.choice(num_in_this_label, known_num_in_this_label, replace=False)
        unknown_indices = np.delete(np.arange(0, num_in_this_label), known_indices)
        
        assert unknown_indices.shape[0]+known_indices.shape[0] == num_in_this_label
        assert known_indices.shape[0] == known_num_in_this_label
        assert unknown_indices.shape[0] == unknown_num_in_this_label
        assert any(np.isin(unknown_indices,known_indices)) == False
        
        known_label_x.extend(data[known_indices])
        unknown_label_x.extend(data[unknown_indices])
        known_label_y.extend([label] * known_num_in_this_label)
        unknown_label_y.extend([label] * unknown_num_in_this_label)
        
    return (np.array(known_label_x), np.array(known_label_y)), (np.array(unknown_label_x), np.array(unknown_label_y))

# Separate Percentage
for labeled_target_percentage in percentage: 
    (target_known_label_x, target_known_label_y), (target_unknown_label_x, target_unknown_label_y) = separate_data_by_percentage(target_dict, labeled_target_percentage)
    assert target_known_label_x.shape[0] + target_unknown_label_x.shape[0] == target_all_x_shuffled.shape[0]
    assert target_unknown_label_x.shape[0] == target_unknown_label_y.shape[0]
    assert target_known_label_x.shape[0] == target_known_label_y.shape[0]
    assert target_all_x_shuffled.shape[0] == target_all_y_shuffled.shape[0]
    np.save(save_path+"processed_file_not_one_hot_%s_%1.1f_target_known_label_x.npy"%(task, labeled_target_percentage), target_known_label_x)
    np.save(save_path+"processed_file_not_one_hot_%s_%1.1f_target_known_label_y.npy"%(task, labeled_target_percentage), target_known_label_y)
    np.save(save_path+"processed_file_not_one_hot_%s_%1.1f_target_unknown_label_x.npy"%(task, labeled_target_percentage), target_unknown_label_x)
    np.save(save_path+"processed_file_not_one_hot_%s_%1.1f_target_unknown_label_y.npy"%(task, labeled_target_percentage), target_unknown_label_y)

for labeled_source_percentage in percentage: 
    (source_known_label_x, source_known_label_y), (source_unknown_label_x, source_unknown_label_y) = separate_data_by_percentage(source_dict, labeled_source_percentage)
    assert source_known_label_x.shape[0] + source_unknown_label_x.shape[0] == source_all_x_shuffled.shape[0]
    assert source_unknown_label_x.shape[0] == source_unknown_label_y.shape[0]
    assert source_known_label_x.shape[0] == source_known_label_y.shape[0]
    assert source_all_x_shuffled.shape[0] == source_all_y_shuffled.shape[0]
    np.save(save_path+"processed_file_not_one_hot_%s_%1.1f_source_known_label_x.npy"%(task, labeled_source_percentage), source_known_label_x)
    np.save(save_path+"processed_file_not_one_hot_%s_%1.1f_source_known_label_y.npy"%(task, labeled_source_percentage), source_known_label_y)
    np.save(save_path+"processed_file_not_one_hot_%s_%1.1f_source_unknown_label_x.npy"%(task, labeled_source_percentage), source_unknown_label_x)
    np.save(save_path+"processed_file_not_one_hot_%s_%1.1f_source_unknown_label_y.npy"%(task, labeled_source_percentage), source_unknown_label_y)


        