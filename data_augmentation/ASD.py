#!/usr/bin/env python
# coding: utf-8

# # Data augmentation
#
# method used here is based on the following paper
#
# Germain Forestier et al. "Generating synthetic time series to augment sparse datasets". ICDM 2017.

# In[3]:


import sys, os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir,'spring-break'))
sys.path.insert(0, os.path.join(parent_dir,'Linear Classifier'))


# In[4]:


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
import logging

# In[5]:


# Parameters
parser = argparse.ArgumentParser(description='JDA Time series adaptation')
parser.add_argument("--data_path", type=str, default="../data_unzip", help="dataset path")
parser.add_argument("--task", type=str, default="3E", help="task type 3E or 3Av2")
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of which target data has label')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of which source data has label')

parser.add_argument("--num_class", type=int, default="65", help="num class")
parser.add_argument("--class_split", type=str, default="0-4", help="class generated")
parser.add_argument("--subset_count", type=int, default="20", help="select number of subset from each class")
parser.add_argument("--duplicate_time", type=float, default="1", help="numer of duplication")
parser.add_argument("--save_path", type=str, default="../train_related/asd/", help="save path")

args = parser.parse_args()

# logging file
subfolder = os.path.join(args.save_path, "class_{}_subset_count_{}".format(args.class_split, args.subset_count))
os.makedirs(subfolder, exist_ok=True)
logging.basicConfig(filename=os.path.join(subfolder, 'logfile.log'), filemode='w', level=logging.INFO)
# In[11]:


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
#     'duplicate_time': 0.01,
#     'source_lbl_percentage': 0.7,
#     'target_lbl_percentage': 0.7,
#     'save_path': '..\train_related\asd',
# })


# In[8]:

logging.info("Loading data")
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

logging.info('Loading Finished!')

# # ASD for source labeled

# In[9]:


def dba_parallel(class_x, verbose=True):

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
        if dist < dnn and i != t_star_ind and dist != 0:
            dnn = dist
            dnn_ind = i

    weight = np.exp(np.log(0.5) * dtw_class_t / dnn)
    dba_avg_t_star = dtw_barycenter_averaging(class_x, weights=weight, max_iter=5, verbose=verbose)
    return dba_avg_t_star


# In[10]:



def dba_parallel_warp(class_x, iter_num, core_used = mp.cpu_count()):
    # parallel
    r = []
    start_time = time.time()
    # r.append(dba_parallel(class_x, verbose=True))
    with parallel_backend("loky", inner_max_num_threads=core_used):
        results = Parallel(n_jobs=core_used)(delayed(dba_parallel)(class_x) for i in range(iter_num))
    r.extend(results)
    end_time = time.time()
    logging.info("Finished indv class: time used: {} s.".format(round(end_time - start_time)))
    return np.array(r)


# In[ ]:





# In[12]:


new_data_x = []
new_data_y = []

start_class, end_class = [int(m) for m in args.class_split.split("-")]

# mission_left = [0, 1]
mission_left = [i for i in range(start_class, end_class + 1)]
overall_start = time.time()
logging.info("###### BEGIN ######## @ {}".format(datetime.now()))
logging.info("Number/class: {}; total class: {}; total DBA times: {}; cpu cores: {};".format(round(args.duplicate_time * 152), mission_left, round(args.duplicate_time * 152) * len(mission_left), mp.cpu_count()))
while len(mission_left) > 0:
    logging.info("\n--- trying --- ")
    logging.info("mission left: {}".format(mission_left))
    i = mission_left[0]
    try:
        logging.info("### START class {} @ {}".format(i, datetime.now()))
        class_ind = np.where(labeled_target_y == i)
        logging.debug("In class {}".format(i))
        class_ind_subset = class_ind[0][np.random.choice(class_ind[0].shape[0], args.subset_count)]
        class_x = labeled_target_x[class_ind_subset]
        iter_num_class = round(args.duplicate_time * class_ind[0].shape[0])
        logging.info("Number of data generated for class {}: {}".format(i, iter_num_class))
        results_class = dba_parallel_warp(class_x, iter_num_class) # [iter_num_class, 2]
        new_data_x.append(results_class)
        new_data_y.extend([np.array([i] * iter_num_class)])
        mission_left.pop(mission_left.index(i))
        logging.debug("misson left after pop: {}".format(mission_left))

        # save so far
        new_data_x_total = np.concatenate(new_data_x, axis=0)
        new_data_y_total = np.concatenate(new_data_y, axis=0)
        logging.info("new_data_x_now: {}".format(new_data_x_total.shape))
        logging.info('new_data_y_now: {}'.format(new_data_y_total.shape))
        np.save(os.path.join(subfolder, 'new_data_x.npy'), new_data_x_total)
        np.save(os.path.join(subfolder, 'new_data_y.npy'), new_data_y_total)
        del new_data_x_total, new_data_y_total
        logging.info('Saved for class {}'.format(i))
    except Exception as e:
        logging.info("error in class {}, skip for now: {}".format(i, e))



overall_end = time.time()
duration = timedelta(-1, overall_end - overall_start)
logging.info("###### END ########")
logging.info("Duriation: {} hrs; {} mins; {} s".format(duration.seconds//3600, duration.seconds//60, duration.seconds % 60))



logging.info("Data saved at {}".format(os.path.abspath(subfolder)))
logging.info("DONE!!! for class_{}".format(args.class_split))
# In[ ]:




