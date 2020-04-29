
# coding: utf-8

# In[1]:


import sys, os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir,'spring-break'))
sys.path.insert(0, os.path.join(parent_dir,'Linear Classifier'))


# In[2]:


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


# # Parser

# In[3]:


# Parameters
parser = argparse.ArgumentParser(description='JDA Time series adaptation')
parser.add_argument("--data_path", type=str, default="../data_unzip", help="dataset path")
parser.add_argument("--task", type=str, help='3A or 3E')
parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--lr_centerloss', type=float, default=0.005, help='learning rate for centerloss')
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of which target data has label')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of which source data has label')
parser.add_argument('--num_per_class', type=int, default=-1, help='number of sample per class when training local discriminator')
parser.add_argument('--save_path', type=str, default='../train_related/', help='where to store data')
parser.add_argument('--gfunction_epoch', type=int, default=5000, help='epoch of which gfunction is trained for')
parser.add_argument('--KL', type=bool, default=False, help="if calculate KL divergence")
parser.add_argument('--JS', type=bool, default=False, help="if calculate JS divergence")
parser.add_argument('--data', type=str, required=True, help='source or target to run')


args = parser.parse_args()


# In[7]:


device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
assert args.data in ['source', 'target']

# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True


args.task = '3Av2' if args.task == '3A' else '3E'
num_class = 50 if args.task == "3Av2" else 65
device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')

if args.num_per_class == -1:
    args.num_per_class = math.ceil(args.batch_size / num_class)
    
model_sub_folder = '/f-gan/%s'%args.data
if args.KL: model_sub_folder += '_KL'
if args.JS: model_sub_folder += '_JS'   

if not os.path.exists(args.save_path+model_sub_folder):
    os.makedirs(args.save_path+model_sub_folder)


# # Logger

# In[8]:


logger = logging.getLogger()
logger.setLevel(logging.INFO)

if os.path.isfile(args.save_path+model_sub_folder+ '/logfile.log'):
    os.remove(args.save_path+model_sub_folder+ '/logfile.log')
    
file_log_handler = logging.FileHandler(args.save_path+model_sub_folder+ '/logfile.log')
logger.addHandler(file_log_handler)

stdout_log_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_log_handler)

attrs = vars(args)
for item in attrs.items():
    logger.info("%s: %s"%item)


# # Data Loading
# 

# In[ ]:


labeled_x_filename = '/processed_file_not_one_hot_%s_%1.1f_%s_known_label_x.npy'%(args.task, args.target_lbl_percentage, args.data)
labeled_y_filename = '/processed_file_not_one_hot_%s_%1.1f_%s_known_label_y.npy'%(args.task, args.target_lbl_percentage, args.data)
unlabeled_x_filename = '/processed_file_not_one_hot_%s_%1.1f_target_unknown_label_x.npy'%(args.task, args.target_lbl_percentage, args.data)
unlabeled_y_filename = '/processed_file_not_one_hot_%s_%1.1f_target_unknown_label_y.npy'%(args.task, args.target_lbl_percentage, args.data)

labeled_x = np.load(args.data_path+labeled_x_filename)
labeled_y = np.load(args.data_path+labeled_y_filename)
unlabeled_x = np.load(args.data_path+unlabeled_x_filename)
unlabeled_y = np.load(args.data_path+unlabeled_y_filename)

join_dataset = JoinDataset(labeled_x, labeled_y, unlabeled_x, unlabeled_y, random=True)
join_dataloader = DataLoader(join_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


# # Weight initialize

# In[10]:


def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LayerNorm:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# # Model creation

# In[ ]:


class Gfunction(nn.Sequential):
    def __init__(self):
        super(Gfunction, self).__init__(
            nn.Linear(3200,1600),
            nn.ELU(),
            nn.Linear(1600,800),
            nn.ELU(),
            nn.Linear(800,1)
        )


# In[ ]:


def log_mean_exp(x, device):
    max_score = x.max()
    batch_size = torch.Tensor([x.shape[0]]).to(device)
    stable_x = x - max_score
    return max_score - batch_size.log() + stable_x.exp().sum(dim=0).log()

a = torch.rand([100,1]).to(device)
assert torch.all(log_mean_exp(a, device) - a.exp().mean(dim=0).log() < 1e-6)


# In[276]:


def KLDiv(g_x_source, g_x_target, device):
    # clipping
#     g_x_source = torch.clamp(g_x_source, -1e3, 1e3)
#     g_x_target = torch.clamp(g_x_target, -1e3, 1e3)
    return g_x_source.mean(dim=0) - log_mean_exp(g_x_target, device)


# In[ ]:


def JSDiv(g_x_source, g_x_target, device):
    return -F.softplus(-g_x_source).mean(dim=0) - F.softplus(g_x_target).mean(dim=0)


# In[ ]:


device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)

if args.KL:
    gfunction_KL_div = Gfunction().to(device)
    gfunction_KL_div.apply(weights_init)
    optimizer_gfunction_KL_div = torch.optim.Adam(gfunction_KL_div.parameters(), lr=args.lr)

if args.JS:
    gfunction_JS_div = Gfunction().to(device)
    gfunction_JS_div.apply(weights_init)
    optimizer_gfunction_JS_div = torch.optim.Adam(gfunction_JS_div.parameters(), lr=args.lr)


# # Train

# In[16]:


labeled_x = torch.Tensor(labeled_x.reshape(-1, 3200)).float().to(device)
unlabeled_x = torch.Tensor(unlabeled_x.reshape(-1, 3200)).float().to(device)

KL_div = []
JS_div = []
for i in tqdm(range(args.gfunction_epoch)):
#     for j, (labeled_x, labeled_y, unlabeled_x, unlabeled_y) in enumerate(join_dataloader):
#     labeled_x = labeled_x.to(device).float().view(-1, 3200)
#     unlabeled_x = unlabeled_x.to(device).float().view(-1, 3200)
    if args.KL:
        optimizer_gfunction_KL_div.zero_grad()
        x_labeled_g = gfunction_KL_div(labeled_x)
        x_unlabeled_g = gfunction_KL_div(unlabeled_x)
        loss_KL_labeled = - KLDiv(x_labeled_g, x_unlabeled_g, device) # maximize
        loss_KL_labeled.backward()
        optimizer_gfunction_KL_div.step()
        loss_KL_labeled = -loss_KL_labeled.item()
        KL_div.append(loss_KL_labeled)

    if args.JS:
        optimizer_gfunction_JS_div.zero_grad()
        x_labeled_g = gfunction_JS_div(labeled_x)
        x_unlabeled_g = gfunction_JS_div(unlabeled_x)
        loss_JS_labeled = - JSDiv(x_labeled_g, x_unlabeled_g, device) # maximize
        loss_JS_labeled.backward()
        optimizer_gfunction_JS_div.step()
        loss_JS_labeled = -loss_JS_labeled.item()
        JS_div.append(loss_JS_labeled)
            
    if i % 200 == 0:
        log_string = "Epoch %i: "%i
        if args.KL: log_string += "KL: %f; "%(loss_KL_labeled)
        if args.JS: log_string += "JS: %f; "%(loss_JS_labeled)   
        logger.info(log_string)
    
    if args.KL: 
        np.save(args.save_path+model_sub_folder+'/KL_div.npy', KL_div)
        
    if args.JS:
        np.save(args.save_path+model_sub_folder+'/JS_div.npy', JS_div)
        
log_string = "Epoch %i: "%i
if args.KL: log_string += "KL: %f; "%(loss_KL_labeled)
if args.JS: log_string += "JS: %f; "%(loss_JS_labeled)   
logger.info(log_string)
        

