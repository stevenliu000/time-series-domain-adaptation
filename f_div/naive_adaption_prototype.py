
# coding: utf-8

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


# # Parser

# In[3]:


# Parameters
parser = argparse.ArgumentParser(description='JDA Time series adaptation')
parser.add_argument("--data_path", type=str, default="/projects/rsalakhugroup/complex/domain_adaptation", help="dataset path")
parser.add_argument("--task", type=str, help='3A or 3E')
parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of which target data has label')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of which source data has label')
parser.add_argument('--num_per_class', type=int, default=-1, help='number of sample per class when training local discriminator')
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--save_path', type=str, help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')
parser.add_argument('--model_path', type=str, help='where the data is stored')
parser.add_argument('--intervals', type=int, default=2, help='freq of compute f-div')


args = parser.parse_args()


# In[6]:


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
#     'gpu_num': 0,
#     'source_lbl_percentage': 0.7,
#     'target_lbl_percentage': 0.7
# })


# In[7]:


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
    
model_sub_folder = '/f-gan/naive_adaption/'

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

# In[9]:


labeled_target_x_filename = '/processed_file_not_one_hot_%s_%1.1f_target_known_label_x.npy'%(args.task, args.target_lbl_percentage)
labeled_target_y_filename = '/processed_file_not_one_hot_%s_%1.1f_target_known_label_y.npy'%(args.task, args.target_lbl_percentage)
unlabeled_target_x_filename = '/processed_file_not_one_hot_%s_%1.1f_target_unknown_label_x.npy'%(args.task, args.target_lbl_percentage)
unlabeled_target_y_filename = '/processed_file_not_one_hot_%s_%1.1f_target_unknown_label_y.npy'%(args.task, args.target_lbl_percentage)
labeled_target_x = np.load(args.data_path+labeled_target_x_filename)
labeled_target_y = np.load(args.data_path+labeled_target_y_filename)
unlabeled_target_x = np.load(args.data_path+unlabeled_target_x_filename)
unlabeled_target_y = np.load(args.data_path+unlabeled_target_y_filename)
labeled_target_dataset = SingleDataset(labeled_target_x, labeled_target_y)
unlabled_target_dataset = SingleDataset(unlabeled_target_x, unlabeled_target_y)
labeled_target_dataloader = DataLoader(labeled_target_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
unlabeled_target_dataloader = DataLoader(unlabled_target_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

labeled_source_x_filename = '/processed_file_not_one_hot_%s_%1.1f_source_known_label_x.npy'%(args.task, args.source_lbl_percentage)
labeled_source_y_filename = '/processed_file_not_one_hot_%s_%1.1f_source_known_label_y.npy'%(args.task, args.source_lbl_percentage)
unlabeled_source_x_filename = '/processed_file_not_one_hot_%s_%1.1f_source_unknown_label_x.npy'%(args.task, args.source_lbl_percentage)
unlabeled_source_y_filename = '/processed_file_not_one_hot_%s_%1.1f_source_unknown_label_y.npy'%(args.task, args.source_lbl_percentage)
labeled_source_x = np.load(args.data_path+labeled_source_x_filename)
labeled_source_y = np.load(args.data_path+labeled_source_y_filename)
unlabeled_source_x = np.load(args.data_path+unlabeled_source_x_filename)
unlabeled_source_y = np.load(args.data_path+unlabeled_source_y_filename)
labeled_source_dataset = SingleDataset(labeled_source_x, labeled_source_y)
unlabled_source_dataset = SingleDataset(unlabeled_source_x, unlabeled_source_y)
labeled_source_dataloader = DataLoader(labeled_source_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
unlabeled_source_dataloader = DataLoader(unlabled_source_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)


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
            nn.Linear(128,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,1)
        )


# In[177]:


def log_mean_exp(x):
    max_score = x.max()
    batch_size = torch.Tensor([x.shape[0]])
    stable_x = x - max_score
    return max_score - batch_size.log() + stable_x.exp().sum(dim=0).log()

a = torch.rand([100,128])
assert torch.all(log_mean_exp(a) - a.exp().mean(dim=0).log() < 1e-6)


# In[180]:


def fDiv(g_x_source, g_x_target):
    # clipping
    g_x_source = torch.clamp(g_x_source, -1e4, 1e4)
    g_x_target = torch.clamp(g_x_target, -1e4, 1e4)
    return g_x_source.mean(dim=0) - g_x_target.exp().mean(dim=0).log()


# In[52]:


device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)

seq_len = 10
feature_dim = 160
encoder = ComplexTransformer(layers=3,
                               time_step=seq_len,
                               input_dim=feature_dim,
                               hidden_size=64,
                               output_dim=64,
                               num_heads=8,
                               out_dropout=0.2,
                               leaky_slope=0.2).to(device)
encoder_MLP = FNNSeparated(d_in=64 * 2 * 1, d_h1=64*4, d_h2=64*2, dp=0.2).to(device)
GNet = Generator(dim=64*2).to(device)
gfunction1 = Gfunction().to(device)
gfunction2 = Gfunction().to(device)

encoder.apply(weights_init)
encoder_MLP.apply(weights_init)
GNet.apply(weights_init)
gfunction1.apply(weights_init)
gfunction2.apply(weights_init)


# In[12]:


def classifier_inference(encoder, CNet, x):
    CNet.eval()
    encoder.eval()
    with torch.no_grad():
        embedding = encoder_inference(encoder, x)
        pred = CNet(embedding)
    return pred


# In[13]:


def encoder_inference(encoder, encoder_MLP, x):
    real = x[:,:,0].reshape(x.size(0), seq_len, feature_dim).float()
    imag = x[:,:,1].reshape(x.size(0), seq_len, feature_dim).float()
    real, imag = encoder(real, imag)
    cat_embedding = torch.cat((real[:,-1,:], imag[:,-1,:]), -1).reshape(x.shape[0], -1)
    cat_embedding = encoder_MLP(cat_embedding)
    return cat_embedding


# # Train

# In[16]:


logger.info('Started loading')
source_acc_label_ = np.load(os.path.join(args.model_path, 'source_acc_label_.npy'))
source_acc_unlabel_ = np.load(os.path.join(args.model_path, 'source_acc_unlabel_.npy'))
target_acc_label_ = np.load(os.path.join(args.model_path, 'target_acc_label_.npy'))
target_acc_unlabel_ = np.load(os.path.join(args.model_path, 'target_acc_unlabel_.npy'))

labeled_f_div = []
unlabeled_f_div = []

source_acc_label = []
source_acc_unlabel = []
target_acc_label = []
target_acc_unlabel = []

epochs = []

for epoch in range(3, source_acc_label_.shape[0], args.intervals*args.model_save_period):
    # initialize 
    optimizerGfunction1 = torch.optim.Adam(gfunction1.parameters(), lr=args.lr)
    gfunction1.apply(weights_init)
    optimizerGfunction2 = torch.optim.Adam(gfunction1.parameters(), lr=args.lr)
    gfunction2.apply(weights_init)

    # load weight
    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder_%i.t7'%epoch)))
    encoder_MLP.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder_MLP%i.t7'%epoch)))
    GNet.load_state_dict(torch.load(os.path.join(args.model_path, 'GNet_%i.t7'%epoch)))
    
    # inferencing
    GNet.eval()
    encoder.eval()
    encoder_MLP.eval()
    
    # get source/target embedding
    source_x_labeled_embedding = torch.empty(0).to(device)
    source_x_unlabeled_embedding = torch.empty(0).to(device)
    target_x_labeled_embedding = torch.empty(0).to(device)
    target_x_unlabeled_embedding = torch.empty(0).to(device)
    with torch.no_grad():
        for batch_id, (source_x, source_y) in tqdm(enumerate(labeled_source_dataloader), total=len(labeled_source_dataloader)):
            source_x = source_x.to(device).float()
            source_x_embedding = encoder_inference(encoder, encoder_MLP, source_x).detach()
            source_x_labeled_embedding = torch.cat([source_x_labeled_embedding, source_x_embedding])
            
        for batch_id, (source_x, source_y) in tqdm(enumerate(unlabeled_source_dataloader), total=len(unlabeled_source_dataloader)):
            source_x = source_x.to(device).float()
            source_x_embedding = encoder_inference(encoder, encoder_MLP, source_x).detach()
            source_x_unlabeled_embedding = torch.cat([source_x_unlabeled_embedding, source_x_embedding])
            
        for batch_id, (target_x, target_y) in tqdm(enumerate(labeled_target_dataloader), total=len(labeled_target_dataloader)):
            target_x = target_x.to(device).float()
            target_x_embedding = encoder_inference(encoder, encoder_MLP, target_x)
            fake_x_embedding = GNet(target_x_embedding).detach()
            target_x_labeled_embedding = torch.cat([target_x_labeled_embedding, fake_x_embedding])     
            
        for batch_id, (target_x, target_y) in tqdm(enumerate(unlabeled_target_dataloader), total=len(unlabeled_target_dataloader)):
            target_x = target_x.to(device).float()
            target_x_embedding = encoder_inference(encoder, encoder_MLP, target_x)
            fake_x_embedding = GNet(target_x_embedding).detach()
            target_x_unlabeled_embedding = torch.cat([target_x_unlabeled_embedding, fake_x_embedding])    
        
    # for loop to train the gfunction 
    for i in tqdm(range(1000)):
        optimizerGfunction1.zero_grad()
        source_x_labeled_g = gfunction1(source_x_labeled_embedding)
        target_x_labeled_g = gfunction1(target_x_labeled_embedding)
        loss1 = fDiv(source_x_labeled_g, target_x_labeled_g)
        loss1.backward()
        optimizerGfunction1.step()
#         if i % 20 == 0:
#             logger.info("Epoch %i, iter %i, labeled f-div: %f"%(epoch, i, loss1.item()))
    loss1 = loss1.item()
    labeled_f_div.append(loss1)
    
    for i in tqdm(range(1000)):
        optimizerGfunction2.zero_grad()
        source_x_unlabeled_g = gfunction2(source_x_unlabeled_embedding)
        target_x_unlabeled_g = gfunction2(target_x_unlabeled_embedding)
        loss2 = fDiv(source_x_unlabeled_g, target_x_unlabeled_g)
        loss2.backward()
        optimizerGfunction2.step()
#         if i % 20 == 0:
#             logger.info("Epoch %i, iter %i, unlabeled f-div: %f"%(epoch, i, loss2.item()))
    loss2 = loss2.item()
    unlabeled_f_div.append(loss2)
        
    # save corresponding acc
    source_acc_label.append(source_acc_label_[epoch-1])
    source_acc_unlabel.append(source_acc_unlabel_[epoch-1])
    target_acc_label.append(target_acc_label_[epoch-1])
    target_acc_unlabel.append(target_acc_unlabel_[epoch-1])
    epochs.append(epoch)
    
    logger.info("-----------------------------------------")
    logger.info("Epoch %i, labeled f-div: %f, unlabeled f-div: %f"%(epoch, loss1, loss2))
    logger.info("-----------------------------------------")
    
    np.save(args.save_path+model_sub_folder+'/epochs.npy', epochs)
    np.save(args.save_path+model_sub_folder+'/labeled_f_div.npy', labeled_f_div)
    np.save(args.save_path+model_sub_folder+'/unlabeled_f_div.npy', unlabeled_f_div)
    np.save(args.save_path+model_sub_folder+'/source_acc_label.npy', source_acc_label)
    np.save(args.save_path+model_sub_folder+'/source_acc_unlabel.npy', source_acc_unlabel)
    np.save(args.save_path+model_sub_folder+'/target_acc_label.npy', target_acc_label)
    np.save(args.save_path+model_sub_folder+'/target_acc_unlabel.npy', target_acc_unlabel)
    

