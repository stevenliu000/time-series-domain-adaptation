#!/usr/bin/env python
# coding: utf-8

# # Data augmentation
# 
# method used here is based on the following paper
# 
# Germain Forestier et al. "Generating synthetic time series to augment sparse datasets". ICDM 2017.

# In[2]:


import sys, os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir,'spring-break'))
sys.path.insert(0, os.path.join(parent_dir,'Linear Classifier'))


# In[3]:


import numpy as np
import random
import copy
import math
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from martins.complex_transformer import ComplexTransformer
from FNNLinear import FNNLinear
from FNNSeparated import FNNSeparated
from GAN import Generator, Discriminator
from data_utils import *
import argparse
import logging
import logging.handlers
import pickle
from centerloss import CenterLoss
from DataSetLoader import JoinDataset, SingleDataset
from torch.autograd import Variable
from binaryloss import BinaryLoss

from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import dtw
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend, Memory
import time
from datetime import datetime, timedelta


# In[1]:


# Parameters
parser = argparse.ArgumentParser(description='JDA Time series adaptation')
parser.add_argument("--data_path", type=str, default="/projects/rsalakhugroup/complex/domain_adaptation", help="dataset path")
parser.add_argument("--task", type=str, default="3E", help="task type 3E or 3Av2")
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of which target data has label')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of which source data has label')

parser.add_argument("--num_class", type=int, default="65", help="num class")
parser.add_argument("--class_split", type=str, default="0-4", help="class generated")
parser.add_argument("--subset_count", type=int, default="20", help="select number of subset from each class")
parser.add_argument("--duplicate_time", type=int, default="1", help="numer of duplication")
parser.add_argument("--save_path", type=str, default="../train_related/asd", help="save path")

args = parser.parse_args()


# In[5]:


# # local only
# class local_args:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
        
# args = local_args(**{
#     'data_path': '../data_unzip',
#     'task': '3E',
#     'num_class': 65,
#     'class_split': "0-4",
#     'subset_count': 20,
#     'duplicate_time': 1,
#     'source_lbl_percentage': 0.7,
#     'target_lbl_percentage': 0.7,
#     'save_path': '..\train_related\asd',
# })


# In[6]:


labeled_target_x_filename = '/processed_file_not_one_hot_%s_%1.1f_target_known_label_x.npy'%(args.task, args.target_lbl_percentage)
labeled_target_y_filename = '/processed_file_not_one_hot_%s_%1.1f_target_known_label_y.npy'%(args.task, args.target_lbl_percentage)
unlabeled_target_x_filename = '/processed_file_not_one_hot_%s_%1.1f_target_unknown_label_x.npy'%(args.task, args.target_lbl_percentage)
unlabeled_target_y_filename = '/processed_file_not_one_hot_%s_%1.1f_target_unknown_label_y.npy'%(args.task, args.target_lbl_percentage)
labeled_target_x = np.load(args.data_path+labeled_target_x_filename)
labeled_target_y = np.load(args.data_path+labeled_target_y_filename)
unlabeled_target_x = np.load(args.data_path+unlabeled_target_x_filename)
unlabeled_target_y = np.load(args.data_path+unlabeled_target_y_filename)

labeled_source_x_filename = '/processed_file_not_one_hot_%s_%1.1f_source_known_label_x.npy'%(args.task, args.source_lbl_percentage)
labeled_source_y_filename = '/processed_file_not_one_hot_%s_%1.1f_source_known_label_y.npy'%(args.task, args.source_lbl_percentage)
unlabeled_source_x_filename = '/processed_file_not_one_hot_%s_%1.1f_source_unknown_label_x.npy'%(args.task, args.source_lbl_percentage)
unlabeled_source_y_filename = '/processed_file_not_one_hot_%s_%1.1f_source_unknown_label_y.npy'%(args.task, args.source_lbl_percentage)
labeled_source_x = np.load(args.data_path+labeled_source_x_filename)
labeled_source_y = np.load(args.data_path+labeled_source_y_filename)
unlabeled_source_x = np.load(args.data_path+unlabeled_source_x_filename)
unlabeled_source_y = np.load(args.data_path+unlabeled_source_y_filename)


# # ASD for source labeled

# In[7]:


def dba_parallel(class_x, verbose=False):
    
    # randomly chose one from class_x
    t_star_ind = np.random.choice(class_x.shape[0], 1)
    t_star = class_x[t_star_ind,][0]

    # dba
    dtw_class_t = np.empty((class_x.shape[0],))
    dnn = float('inf')
    dnn_ind = float('inf')
    for i in tqdm(range(class_x.shape[0])):
        dist = dtw(class_x[i], t_star)
        dtw_class_t[i] = dist
        if dist < dnn and i != t_star_ind:
            dnn = dist
            dnn_ind = i
    weight = np.exp(np.log(0.5) * dtw_class_t / dnn)
    dba_avg_t_star = dtw_barycenter_averaging(class_x, weights=weight, max_iter=5, verbose=verbose)
    return dba_avg_t_star


# In[8]:


cachedir = os.path.join(args.save_path, r'\cachedir')
memory = Memory(cachedir, verbose=0)
@memory.cache
def dba_parallel_warp(class_x, iter_num, core_used = mp.cpu_count() - 2):
    # parallel
    print("Number of processors used: ", core_used)
    start_time = time.time()
    with parallel_backend("loky", inner_max_num_threads=core_used):
        results = Parallel(n_jobs=core_used)(delayed(dba_parallel)(class_x) for i in range(iter_num))
    end_time = time.time()
    print("time used: ", end_time - start_time)
    memory.clear(warn=False)
    return np.array(results)


# In[ ]:





# In[ ]:


new_data_x = []
new_data_y = []

start_class, end_class = [int(m) for m in args.class_split.split("-")]

mission_left = [0]
overall_start = time.time()
print("###### BEGIN ########")
while len(mission_left) > 0:
    print("--- trying --- ")
    print("mission left: ", mission_left)
    for i in mission_left:
        try:
            class_ind = np.where(labeled_source_y == i)
            class_ind_subset = class_ind[0][np.random.choice(class_ind[0].shape[0], args.subset_count)]
            class_x = labeled_source_x[class_ind_subset]
            iter_num_class = round(args.duplicate_time * class_ind[0].shape[0])
            print("number of data generated for class {}: {}".format(i, iter_num_class))
            results_class = dba_parallel_warp(class_x, iter_num_class) # [iter_num_class, 2]
            new_data_x.append(results_class)
            new_data_y.extend([np.array([i] * iter_num_class)])
            mission_left.pop(mission_left.index(i))
            del class_x, class_ind, class_ind_subset, results_class
        except:
            print("error in class {}, skip for now.".format(i))
        
new_data_x = np.concatenate(new_data_x, axis=0)
new_data_y = np.concatenate(new_data_y, axis=0)
overall_end = time.time()
duration = timedelta(-1, overall_end - overall_start)
print("###### END ########")
print("Duriation: {} hrs; {} mins; {} s".format(duration.seconds//3600, duration.seconds//60, duration.seconds))



np.save('new_data_x.npy', new_data_x)
np.save('new_data_y', new_data_y)


# In[ ]:




