
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
from scipy.linalg import block_diag
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
from BinaryLossNewImplementation import BinaryLossNewImplementation


# # Parser

# In[3]:


# Parameters
parser = argparse.ArgumentParser(description='JDA Time series adaptation')
parser.add_argument("--data_path", type=str, default="/projects/rsalakhugroup/complex/domain_adaptation", help="dataset path")
parser.add_argument("--task", type=str, help='3A or 3E')
parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of which target data has label')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of which source data has label')
parser.add_argument('--num_per_class', type=int, default=-1, help='number of sample per class when training local discriminator')
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--save_path', type=str, help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')
args = parser.parse_args()


# In[13]:


# # local only
# class local_args:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
        
# args = local_args(**{
#     'data_path': '../data_unzip',
#     'task': '3E',
#     'num_class': 50,
#     'batch_size': 100,
#     'num_per_class': -1,
#     'gap': 5,
#     'lbl_percentage':0.7,
#     'source_lbl_percentage': 0.7,
#     'target_lbl_percentage': 0.7,
#     'lr_gan': 1e-4,
#     'lr_FNN': 1e-4,
#     'lr_encoder': 1e-4,
#     'epochs': 2,
#     'clip_value': 0.01,
#     'n_critic': 4,
#     'sclass': 0.7,
#     'scent': 1e-2,
#     'seed': None,
#     'save_path': '../train_related',
#     'model_save_period': 1,
#     'lr_centerloss': 1e-3,
#     'lr_prototype': 1e-3,
#     'sprototype': 1e-2,
#     'seed': 0,
#     'select_pretrain_epoch': 77,
#     'epoch_begin_prototype': 0,
#     'sbinary_loss': 1,
# })


# In[6]:


device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')

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
    
model_sub_folder = '/checkpoint2/simple_MLP/task_%s_slp_%f_tlp_%f'%(args.task, args.source_lbl_percentage, args.target_lbl_percentage)

if not os.path.exists(args.save_path+model_sub_folder):
    os.makedirs(args.save_path+model_sub_folder)
    
pesudo_dict = {i:[] for i in range(num_class)}


# # Logger

# In[7]:


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

# In[8]:


labeled_target_x_filename = '/processed_file_not_one_hot_%s_%1.1f_target_known_label_x.npy'%(args.task, args.target_lbl_percentage)
labeled_target_y_filename = '/processed_file_not_one_hot_%s_%1.1f_target_known_label_y.npy'%(args.task, args.target_lbl_percentage)
unlabeled_target_x_filename = '/processed_file_not_one_hot_%s_%1.1f_target_unknown_label_x.npy'%(args.task, args.target_lbl_percentage)
unlabeled_target_y_filename = '/processed_file_not_one_hot_%s_%1.1f_target_unknown_label_y.npy'%(args.task, args.target_lbl_percentage)
labeled_target_x = np.load(args.data_path+labeled_target_x_filename)
labeled_target_y = np.load(args.data_path+labeled_target_y_filename)
unlabeled_target_x = np.load(args.data_path+unlabeled_target_x_filename)
unlabeled_target_y = np.load(args.data_path+unlabeled_target_y_filename)
labeled_target_dataset = SingleDataset(labeled_target_x, labeled_target_y)
unlabeled_target_dataset = SingleDataset(unlabeled_target_x, unlabeled_target_y)
labeled_target_dataloader = DataLoader(labeled_target_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
unlabeled_target_dataloader = DataLoader(unlabeled_target_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

labeled_source_x_filename = '/processed_file_not_one_hot_%s_%1.1f_source_known_label_x.npy'%(args.task, args.source_lbl_percentage)
labeled_source_y_filename = '/processed_file_not_one_hot_%s_%1.1f_source_known_label_y.npy'%(args.task, args.source_lbl_percentage)
unlabeled_source_x_filename = '/processed_file_not_one_hot_%s_%1.1f_source_unknown_label_x.npy'%(args.task, args.source_lbl_percentage)
unlabeled_source_y_filename = '/processed_file_not_one_hot_%s_%1.1f_source_unknown_label_y.npy'%(args.task, args.source_lbl_percentage)
labeled_source_x = np.load(args.data_path+labeled_source_x_filename)
labeled_source_y = np.load(args.data_path+labeled_source_y_filename)
unlabeled_source_x = np.load(args.data_path+unlabeled_source_x_filename)
unlabeled_source_y = np.load(args.data_path+unlabeled_source_y_filename)
labeled_source_dataset = SingleDataset(labeled_source_x, labeled_source_y)
unlabeled_source_dataset = SingleDataset(unlabeled_source_x, unlabeled_source_y)
labeled_source_dataloader = DataLoader(labeled_source_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
unlabeled_source_dataloader = DataLoader(unlabeled_source_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

join_dataset = JoinDataset(labeled_source_x, labeled_source_y, labeled_target_x, labeled_target_y, random=True)
join_dataloader = DataLoader(join_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

source_labeled_dict = get_class_data_dict(labeled_source_x, labeled_source_y, num_class)
target_labeled_dict = get_class_data_dict(labeled_target_x, labeled_target_y, num_class)


# # Weight initialize

# In[9]:


def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LayerNorm:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# In[ ]:


class SimpleMLP1(nn.Sequential):
    def __init__(self):
        super(SimpleMLP1, self).__init__(
            nn.Linear(3200,1600),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(1600,800),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(800,800),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(800,400),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(400,400),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(400,400),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(400,200),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(200,200),
            nn.ELU(),
        )
        
class SimpleMLP2(nn.Sequential):
    def __init__(self):
        super(SimpleMLP2, self).__init__(
            nn.Dropout(0.2),
            nn.Linear(200,num_class),
        )


# # Model creation

# In[10]:


device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)

encoder = SimpleMLP1().to(device)
CNet = SimpleMLP2().to(device)

criterion_classifier = nn.CrossEntropyLoss().to(device)

CNet.apply(weights_init)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(CNet.parameters()), lr=args.lr)


# # Train

# In[14]:


source_acc_label_ = []
source_acc_unlabel_ = []
target_acc_label_ = []
target_acc_unlabel_ = []

logger.info('Started Training')
for epoch in range(args.epochs):
    # update classifier
    # on source domain
    encoder.train()
    CNet.train()
 
    source_acc_label = 0.0
    num_datas = 0.0
    optimizerCNet.zero_grad()
    for batch_id, (source_x, source_y) in tqdm(enumerate(labeled_source_dataloader), total=len(labeled_source_dataloader)):
        optimizer.zero_grad()
        source_x = source_x.to(device).view(-1,3200).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)
        pred = CNet(encoder(source_x))
        source_acc_label += (pred.argmax(-1) == source_y).sum().item()
        loss = criterion_classifier(pred, source_y)
        loss.backward()
        optimizer.step()
        
    source_acc_label = source_acc_label / num_datas
    source_acc_label_.append(source_acc_label)
    
    # on target domain
    encoder.train()
    CNet.train()

    target_acc_label = 0.0
    num_datas = 0.0
    for batch_id, (target_x, target_y) in tqdm(enumerate(labeled_target_dataloader), total=len(labeled_target_dataloader)):
        optimizer.zero_grad()
        target_x = target_x.to(device).view(-1,3200).float()
        target_y = target_y.to(device)
        num_datas += target_x.size(0)
        pred = CNet(encoder(target_x))
        target_acc_label += (pred.argmax(-1) == target_y).sum().item()
        loss = criterion_classifier(pred, target_y) 
        loss.backward()
        optimizer.step()
        
    target_acc_label = target_acc_label / num_datas
    target_acc_label_.append(target_acc_label)
    
    
    # eval
    # source_domain
    source_acc_unlabel = 0.0
    num_datas = 0.0
    encoder.eval()
    CNet.eval()
    for batch_id, (source_x, source_y) in tqdm(enumerate(unlabeled_source_dataloader), total=len(unlabeled_source_dataloader)):
        source_x = source_x.to(device).view(-1,3200).float()
        source_y = source_y.to(device)
        num_datas += source_x.shape[0]
        pred = CNet(encoder(source_x))
        source_acc_unlabel += (pred.argmax(-1) == source_y).sum().item()
        
    source_acc_unlabel = source_acc_unlabel/num_datas
    source_acc_unlabel_.append(source_acc_unlabel)
    
    # target_domain
    target_acc_unlabel = 0.0
    num_datas = 0.0
    encoder.eval()
    CNet.eval()
    for batch_id, (target_x, target_y) in tqdm(enumerate(unlabeled_target_dataloader), total=len(unlabeled_target_dataloader)):
        target_x = target_x.to(device).view(-1,3200).float()
        target_y = target_y.to(device)
        num_datas += target_x.shape[0]
        pred = CNet(encoder(target_x))
        target_acc_unlabel += (pred.argmax(-1) == target_y).sum().item()
        
    target_acc_unlabel = target_acc_unlabel/num_datas
    target_acc_unlabel_.append(target_acc_unlabel)
    
    if epoch % args.model_save_period == 0:
        torch.save(CNet.state_dict(), args.save_path+model_sub_folder+ '/CNet_%i.t7'%(epoch+1))
        torch.save(encoder.state_dict(), args.save_path+model_sub_folder+ '/encoder_%i.t7'%(epoch+1))

    logger.info('Epochs %i: src labeled acc: %f; src unlabeled acc: %f; tgt labeled acc: %f; tgt unlabeled acc: %f'%(epoch+1, source_acc_label, source_acc_unlabel, target_acc_label, target_acc_unlabel))
    np.save(args.save_path+model_sub_folder+'/target_acc_label_.npy', target_acc_label_)
    np.save(args.save_path+model_sub_folder+'/target_acc_unlabel_.npy', target_acc_unlabel_)
    np.save(args.save_path+model_sub_folder+'/source_acc_label_.npy', source_acc_label_)
    np.save(args.save_path+model_sub_folder+'/source_acc_unlabel_.npy', source_acc_unlabel_)
    
    

