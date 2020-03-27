#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir,'spring-break'))


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
from FNN import FNN
from GAN import Generator, Discriminator
from data_utils import *
import argparse
import logging
import logging.handlers
import pickle
from centerloss import CenterLoss
from prototype_loss import PrototypeLoss
from DataSetLoader import JoinDataset, SingleDataset
from torch.autograd import Variable


# # Parser

# In[3]:


# Parameters
parser = argparse.ArgumentParser(description='JDA Time series adaptation')
parser.add_argument("--data_path", type=str, default="/projects/rsalakhugroup/complex/domain_adaptation", help="dataset path")
parser.add_argument("--task", type=str, help='3A or 3E')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr_gan', type=float, default=1e-3, help='learning rate for adversarial')
parser.add_argument('--lr_centerloss', type=float, default=0.5, help='learning rate for centerloss')
parser.add_argument('--lr_prototype', type=float, default=0.5, help='learning rate for prototype')
parser.add_argument('--lr_FNN', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--lbl_percentage', type=float, default=0.7, help='percentage of which target data has label')
parser.add_argument('--num_per_class', type=int, default=-1, help='number of sample per class when training local discriminator')
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--save_path', type=str, help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')
parser.add_argument('--sclass', type=float, default=0.7, help='source domain classification weight on loss function')
parser.add_argument('--scent', type=float, default=0.01, help='source domain classification weight on centerloss')
parser.add_argument('--sprototype', type=float, default=0.01, help='prototype weight on target doamin loss')
parser.add_argument('--select_pretrain_epoch', type=int, default=77, help='select epoch num for pretrained medel weight')
parser.add_argument('--epoch_begin_prototype', type=int, default=10, help='starting point to train on prototype loss.')


args = parser.parse_args()


# In[6]:


# # local only
# class local_args:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
        
# args = local_args(**{
#     'data_path': '/Users/tianqinli/Code/Russ/time-series-domain-adaptation/data_unzip',
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
#     'save_path': '/Users/tianqinli/Code/Russ/time-series-domain-adaptation/train_related',
#     'model_save_period': 1,
#     'lr_centerloss': 1e-3,
#     'lr_prototype': 1e-3,
#     'sprototype': 1e-2,
#     'seed': 0,
#     'select_pretrain_epoch': 77,
#     'epoch_begin_prototype': 20
# })


# In[7]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True


args.task = '3Av2' if args.task == '3A' else '3E'
num_class = 50 if args.task == "3Av2" else 65
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.num_per_class == -1:
    args.num_per_class = math.ceil(args.batch_size / num_class)
    
model_sub_folder = '/stepc_endToend/task_%s_sprototype_%f_lrFNN_%f_sclass_%f'%(args.task, args.sprototype, args.lr_FNN, args.sclass)

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


# # Data Loading

# In[9]:


raw_data = np.load(args.data_path+'/processed_file_not_one_hot_%s.pkl'%args.task, allow_pickle=True)
target_dict, (target_unlabel_x, target_unlabel_y),(target_label_x,target_label_y), target_len  = get_target_dict(args.data_path+'/processed_file_not_one_hot_%s.pkl'%args.task, num_class, args.lbl_percentage)
source_dict, source_len = get_source_dict(args.data_path+'/processed_file_not_one_hot_%s.pkl'%args.task, num_class, data_len=target_len)
join_dataset = JoinDataset(raw_data['tr_data'],raw_data['tr_lbl'],raw_data['te_data'],raw_data['te_lbl'], random=True)
join_dataloader = DataLoader(join_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

source_dataset = SingleDataset(raw_data['tr_data'], raw_data['tr_lbl'])
source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
target_lbl_dataset = SingleDataset(target_label_x, target_label_y)
target_dataloader = DataLoader(target_lbl_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)


# # Weight initialize

# In[10]:


def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LayerNorm:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# # Model creation

# In[11]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

CNet = FNN(d_in=64 * 2 * 1, d_h1=500, d_h2=500, d_out=num_class, dp=0.2).to(device)
GNet = Generator(dim=64*2).to(device)

criterion_classifier = nn.CrossEntropyLoss().to(device)
criterion_centerloss = CenterLoss(num_classes=num_class, feat_dim=64 * 2 * 1, use_gpu=torch.cuda.is_available()).to(device)
criterion_prototype = PrototypeLoss(torch.cuda.is_available(), device).to(device)

GNet.apply(weights_init)
encoder.apply(weights_init)
CNet.apply(weights_init)

optimizerG = torch.optim.Adam(GNet.parameters(), lr=args.lr_gan)
optimizerFNN = torch.optim.Adam(CNet.parameters(), lr=args.lr_FNN)
optimizerEncoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)
optimizerCenterLoss = torch.optim.Adam(criterion_centerloss.parameters(), lr=args.lr_centerloss)


# In[12]:


def classifier_inference(encoder, CNet, x):
    CNet.eval()
    encoder.eval()
    with torch.no_grad():
        embedding = encoder_inference(encoder, x)
        pred = CNet(embedding)
    return pred


# In[13]:


def encoder_inference(encoder, x):
    real = x[:,:,0].reshape(x.size(0), seq_len, feature_dim).float()
    imag = x[:,:,1].reshape(x.size(0), seq_len, feature_dim).float()
    real, imag = encoder(real, imag)
    return torch.cat((real[:,-1,:], imag[:,-1,:]), -1).reshape(x.shape[0], -1)


# In[14]:


def compute_mean(samples, labels):
    assert samples.size(0) == labels.size(0)
    """
    samples = torch.Tensor([
                         [0.1, 0.1],    #-> group / class 1
                         [0.2, 0.2],    #-> group / class 2
                         [0.4, 0.4],    #-> group / class 2
                         [0.0, 0.0]     #-> group / class 0
                  ])
    labels = torch.LongTensor([1, 2, 2, 0])
    return 
        tensor([[0.0000, 0.0000],
                [0.1000, 0.1000],
                [0.3000, 0.3000]])
    """
    M = torch.zeros(labels.max()+1, len(samples)).to(device)
    M[labels, torch.arange(len(samples))] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)
    res = torch.mm(M, samples)
    return res


# # Train

# In[29]:


target_acc_label_ = []
source_acc_ = []
target_acc_unlabel_ = []




logger.info('Started Training')
for epoch in range(args.epochs):
    # update classifier
    # on source domain
    CNet.train()
    encoder.train()
    GNet.train()
    source_acc = 0.0
    num_datas = 0.0
    
    
    source_x_embeddings = torch.empty(0)
    source_ys = torch.empty(0, dtype=torch.long)
    
    
    for batch_id, (source_x, source_y) in tqdm(enumerate(source_dataloader), total=len(source_dataloader)):
        optimizerFNN.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerCenterLoss.zero_grad()
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)
        source_x_embedding = encoder_inference(encoder, source_x)
        source_x_embeddings = torch.cat([source_x_embeddings, source_x_embedding])
        source_ys = torch.cat([source_ys, source_y])
        pred = CNet(source_x_embedding)
        source_acc += (pred.argmax(-1) == source_y).sum().item()
        loss = (criterion_classifier(pred, source_y) +
                criterion_centerloss(source_x_embedding, source_y) * args.scent) * args.sclass
        loss.backward()
        optimizerFNN.step()
        optimizerCenterLoss.step()
        optimizerEncoder.step()
        
    
    
    source_acc = source_acc / num_datas
    source_acc_.append(source_acc)
    
    # get center
    

    print(source_x_embeddings.shape)
    print(source_y.sahpe)
    print("hi")
    source_centers = compute_mean(source_x_embeddings, source_ys) # (65, 128)
    
    print(source_centers.shape)
    
    # on target domain
    target_acc = 0.0
    num_datas = 0.0
    CNet.train()
    encoder.train()
    GNet.train()
    
    for batch_id, (target_x, target_y) in tqdm(enumerate(target_dataloader), total=len(target_dataloader)):
        
    
        optimizerFNN.zero_grad()
        optimizerG.zero_grad()
        optimizerEncoder.zero_grad()
        target_x = target_x.to(device).float()
        target_y = target_y.to(device)
        num_datas += target_x.size(0)
        target_x_embedding = encoder_inference(encoder, target_x)
        fake_target_embedding = GNet(target_x_embedding)
        pred = CNet(fake_target_embedding)
        target_acc += (pred.argmax(-1) == target_y).sum().item()
        
        if epoch >= args.epoch_begin_prototype: 
            # prototype loss calculate
            center_batch = source_centers[target_y, ]
            dist = torch.sum(torch.pow(fake_target_embedding - center_batch, 2), axis=1)
            unique_class_y, class_count = torch.unique(target_y, return_counts=True)
            prototype_loss = torch.mean(compute_mean(dist.view(-1,1), target_y))
            # add prototype loss
            loss = criterion_classifier(pred, target_y) + args.sprototype * prototype_loss
        else:
            loss = criterion_classifier(pred, target_y)
            
        loss.backward()
        optimizerFNN.step()
        optimizerG.step()
        optimizerEncoder.step()

    target_acc = target_acc / num_datas
    target_acc_label_.append(target_acc)
        
    correct_target = 0.0
    num_datas = 0.0
    CNet.eval()
    encoder.eval()
    GNet.eval()
    for batch in range(math.ceil(target_unlabel_x.shape[0]/args.batch_size)):
        target_unlabel_x_batch = torch.Tensor(target_unlabel_x[batch*args.batch_size:(batch+1)*args.batch_size]).to(device).float()
        target_unlabel_y_batch = torch.Tensor(target_unlabel_y[batch*args.batch_size:(batch+1)*args.batch_size]).to(device)
        num_datas += target_unlabel_x_batch.shape[0]
        target_unlabel_x_embedding = encoder_inference(encoder, target_unlabel_x_batch)
        fake_source_embedding = GNet(target_unlabel_x_embedding)
        pred = CNet(fake_source_embedding)
        correct_target += (pred.argmax(-1) == target_unlabel_y_batch).sum().item()
        
    target_unlabel_acc = correct_target/num_datas
    target_acc_unlabel_.append(target_unlabel_acc)
    
    
    if epoch % args.model_save_period == 0:
        torch.save(GNet.state_dict(), args.save_path+model_sub_folder+ '/GNet_%i.t7'%(epoch+1))
        torch.save(encoder.state_dict(), args.save_path+model_sub_folder+ '/encoder_%i.t7'%(epoch+1))
        torch.save(CNet.state_dict(), args.save_path+model_sub_folder+ '/CNet_%i.t7'%(epoch+1))
    if epoch == args.epoch_begin_prototype:
            logger.info('Epochs %i: Pass naive: source acc: %f; target labled acc: %f; target unlabeled acc: %f'%(epoch+1, source_acc, target_acc, target_unlabel_acc))
    logger.info('Epochs %i: source acc: %f; target labled acc: %f; target unlabeled acc: %f'%(epoch+1, source_acc, target_acc, target_unlabel_acc))
    np.save(args.save_path+model_sub_folder+'/source_acc_.npy',source_acc_)
    
    np.save(args.save_path+model_sub_folder+'/target_acc_label_.npy',target_acc_label_)
    np.save(args.save_path+model_sub_folder+'/target_acc_unlabel_.npy',target_acc_unlabel_)
    


# In[28]:


a = torch.empty(0, dtype=torch.long)
b = torch.LongTensor([2,3])
print(b)
m = [a,b]
torch.cat(m, dim=0)


# In[ ]:




