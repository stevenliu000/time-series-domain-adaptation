#!/usr/bin/env python
# coding: utf-8

# In[16]:


import sys, os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


# In[17]:


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


# # DataLoader

# In[3]:


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
            index_source = random.randrange(source_len)
            index_target = random.randrange(target_len)
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


# # Parser

# In[4]:


# Parameters
parser = argparse.ArgumentParser(description='JDA Time series adaptation')
parser.add_argument("--data_path", type=str, default="/projects/rsalakhugroup/complex/domain_adaptation", help="dataset path")
parser.add_argument("--task", type=str, help='3A or 3E')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr_gan', type=float, default=1e-3, help='learning rate for adversarial')
parser.add_argument('--lr_FNN', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--n_critic', type=int, default=4, help='gap: Generator train GAP times, discriminator train once')
parser.add_argument('--lbl_percentage', type=float, default=0.2, help='percentage of which target data has label')
parser.add_argument('--num_per_class', type=int, default=-1, help='number of sample per class when training local discriminator')
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--save_path', type=str, help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')
parser.add_argument('--clip_value', type=float, default=0.01, help='clip_value for WGAN')
parser.add_argument('--dlocal', type=float, default=0.01, help='local GAN weight on loss function')


args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# seed
if args.seed is None:
    args.seed = random.randint(1, 10000)
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
    
model_sub_folder = '/stage3_local/task_%s_clip_%f_dglobal_%f_critic_%i_sclass_%f'%(args.task, args.clip_value, args.dlocal, args.n_critic, args.sclass)

if not os.path.exists(args.save_path+model_sub_folder):
    os.makedirs(args.save_path+model_sub_folder)


# In[4]:


# # local only
# class local_args:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
        
# args = local_args(**{
#     'data_path': '/Users/stevenliu/time-series-adaption/time-series-domain-adaptation/data_unzip',
#     'task': '3E',
#     'num_class': 50,
#     'batch_size': 100,
#     'num_per_class': -1,
#     'gap': 5,
#     'lbl_percentage':0.2,
#     'lr_gan': 1e-4,
#     'lr_FNN': 1e-4,
#     'lr_encoder': 1e-4,
#     'epochs': 2,
#     'clip_value': 0.01,
#     'n_critic': 4
# })


# In[5]:


# args.task = '3Av2' if args.task == '3A' else '3E'
# num_class = 50 if args.task == "3Av2" else 65
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# if args.num_per_class == -1:
#     args.num_per_class = math.ceil(args.batch_size / num_class)
    
# model_sub_folder = '/task_%s_gap_%s_lblPer_%i_numPerClass_%i'%(args.task, args.gap, args.lbl_percentage, args.num_per_class)
    


# # Logger

# In[6]:


logger = logging.getLogger()
logger.setLevel(logging.INFO)

if os.path.isfile(args.save_path+model_sub_folder+ '/logfile.log'):
    os.remove(args.save_path+model_sub_folder+ '/logfile.log')

file_log_handler = logging.FileHandler(args.save_path+model_sub_folder+ '/logfile.log')
logger.addHandler(file_log_handler)

stdout_log_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_log_handler)


# # Data loading

# In[7]:


raw_data = np.load(args.data_path+'/processed_file_not_one_hot_%s.pkl'%args.task, allow_pickle=True)


# In[8]:


target_dict, (target_unlabel_x, target_unlabel_y),(target_label_x,target_label_y), target_len  = get_target_dict(args.data_path+'/processed_file_not_one_hot_%s.pkl'%args.task, num_class, args.lbl_percentage)
source_dict, source_len = get_source_dict(args.data_path+'/processed_file_not_one_hot_%s.pkl'%args.task, num_class, data_len=target_len)


# In[9]:


join_dataset = JoinDataset(raw_data['tr_data'],raw_data['tr_lbl'],raw_data['te_data'],raw_data['te_lbl'], random=True)
join_dataloader = DataLoader(join_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

source_dataset = SingleDataset(raw_data['tr_data'], raw_data['tr_lbl'])
source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
target_lbl_dataset = SingleDataset(target_label_x, target_label_y)
target_dataloader = DataLoader(target_lbl_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)


# # weight Initialize

# In[15]:


def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LayerNorm:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# # model creation

# In[11]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seq_len = 10
feature_dim = 160
encoder = ComplexTransformer(layers=3,
                               time_step=seq_len,
                               input_dim=feature_dim,
                               hidden_size=64,
                               output_dim=64,
                               num_heads=8,
                               out_dropout=0.2,
                               leaky_slope=0.2)
encoder.to(device)

CNet = FNN(d_in=64 * 2 * 1, d_h1=500, d_h2=500, d_out=num_class, dp=0.2)
CNet.to(device)

DNet_global = Discriminator(feature_dim=64*2, d_out=1).to(device)
DNet_local = Discriminator(feature_dim=64*2, d_out=num_class).to(device)
GNet = Generator(dim=64*2).to(device)
DNet_global.apply(weights_init)
DNet_local.apply(weights_init)
GNet.apply(weights_init)
encoder.apply(weights_init)
CNet.apply(weights_init)
optimizerD_global = torch.optim.Adam(DNet_global.parameters(), lr=args.lr_gan)
optimizerD_local = torch.optim.Adam(DNet_local.parameters(), lr=args.lr_gan)
optimizerG = torch.optim.Adam(GNet.parameters(), lr=args.lr_gan)
optimizerFNN = torch.optim.Adam(CNet.parameters(), lr=args.lr_FNN)
optimizerEncoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)
criterion_classifier = nn.CrossEntropyLoss().to(device)


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


# # Train

# In[14]:


target_acc_label_ = []
source_acc_ = []
target_acc_unlabel_ = []
error_D_global = []
error_G_global = []
error_D_local = []
error_G_local = []


# pre-trained
model_PATH = '../train_related/spring_break/stage2/task_3E_SClassWeight_0.700000_old'
CNet.load_state_dict(torch.load(model_PATH+'/CNet_pre_trained.t7', map_location=device))
encoder.load_state_dict(torch.load(model_PATH+'/encoder_pre_trained.t7', map_location=device))
GNet.load_state_dict(torch.load(model_PATH+'/GNet_pre_trained.t7', map_location=device))
print('Model Loaded!')
'''
print('Started Pre training')
for epoch in range(30):
    # update classifier
    # on source domain
    CNet.train()
    encoder.train()
    GNet.train()
    source_acc = 0.0
    num_datas = 0.0
    for batch_id, (source_x, source_y) in tqdm(enumerate(source_dataloader), total=len(source_dataloader)):
        optimizerFNN.zero_grad()
        optimizerEncoder.zero_grad()
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)
        source_x_embedding = encoder_inference(encoder, source_x)
        pred = CNet(source_x_embedding)
        source_acc += (pred.argmax(-1) == source_y).sum().item()
        loss = criterion_classifier(pred, source_y)*0.5
        loss.backward()
        optimizerFNN.step()
        optimizerEncoder.step()
        
    source_acc = source_acc / num_datas
    source_acc_.append(source_acc)
    
    
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
        fake_source_embedding = GNet(target_x_embedding)
        pred = CNet(fake_source_embedding)
        target_acc += (pred.argmax(-1) == target_y).sum().item()
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
    print('Epoch: %i, update classifier: source acc: %f; target labled acc: %f; target unlabeled acc: %f'%(epoch+1, source_acc, target_acc, target_unlabel_acc))
    
torch.save(GNet.state_dict(), args.save_path+model_sub_folder+ '/GNet_pre_trained.t7')
torch.save(encoder.state_dict(), args.save_path+model_sub_folder+ '/encoder_pre_trained.t7')
torch.save(CNet.state_dict(), args.save_path+model_sub_folder+ '/CNet_pre_trained.t7')
logger.info('Pre-train: source acc: %f; target labled acc: %f; target unlabeled acc: %f'%(source_acc, target_acc, target_unlabel_acc))
'''
logger.info('Started Training')
for epoch in range(args.epochs):
    # update classifier
    # on source domain
    CNet.train()
    encoder.eval()
    GNet.eval()
    source_acc = 0.0
    num_datas = 0.0
    for batch_id, (source_x, source_y) in tqdm(enumerate(source_dataloader), total=len(source_dataloader)):
        optimizerFNN.zero_grad()
        optimizerEncoder.zero_grad()
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)
        source_x_embedding = encoder_inference(encoder, source_x)
        pred = CNet(source_x_embedding)
        source_acc += (pred.argmax(-1) == source_y).sum().item()
        loss = criterion_classifier(pred, source_y) * 0.5
        loss.backward()
        optimizerFNN.step()
#         optimizerEncoder.step()
        
    source_acc = source_acc / num_datas
    source_acc_.append(source_acc)
    
    # on target domain
    target_acc = 0.0
    num_datas = 0.0
    CNet.train()
    encoder.eval()
    GNet.eval()
    for batch_id, (target_x, target_y) in tqdm(enumerate(target_dataloader), total=len(target_dataloader)):
        optimizerFNN.zero_grad()
        optimizerG.zero_grad()
        optimizerEncoder.zero_grad()
        target_x = target_x.to(device).float()
        target_y = target_y.to(device)
        num_datas += target_x.size(0)
        target_x_embedding = encoder_inference(encoder, target_x)
        fake_source_embedding = GNet(target_x_embedding)
        pred = CNet(fake_source_embedding)
        target_acc += (pred.argmax(-1) == target_y).sum().item()
        loss = criterion_classifier(pred, target_y)
        loss.backward()
        optimizerFNN.step()
#         optimizerG.step()
#         optimizerEncoder.step()
    
    target_acc = target_acc / num_datas
    target_acc_label_.append(target_acc)
    
    logger.info('Epoch: %i, update classifier: source acc: %f; target acc: %f'%(epoch+1, source_acc, target_acc))
    # Assign Pesudo Label
    correct_target = 0.0
    target_pesudo_y = []
    CNet.eval()
    encoder.eval()
    GNet.eval()
    for batch in range(math.ceil(target_unlabel_x.shape[0]/args.batch_size)):
        target_unlabel_x_batch = torch.Tensor(target_unlabel_x[batch*args.batch_size:(batch+1)*args.batch_size]).to(device).float()
        target_unlabel_y_batch = torch.Tensor(target_unlabel_y[batch*args.batch_size:(batch+1)*args.batch_size]).to(device)        
        target_unlabel_x_embedding = encoder_inference(encoder, target_unlabel_x_batch)
        fake_source_embedding = GNet(target_unlabel_x_embedding)
        pred = CNet(fake_source_embedding)
        correct_target += (pred.argmax(-1) == target_unlabel_y_batch).sum().item()
        target_pesudo_y.extend(pred.argmax(-1).cpu().numpy())
        
    target_pesudo_y = np.array(target_pesudo_y)
    pesudo_dict = get_class_data_dict(target_unlabel_x, target_pesudo_y, num_class)

    logger.info('Epoch: %i, assigned pesudo label with accuracy %f'%(epoch+1, correct_target/(target_unlabel_x.shape[0])))
    target_acc_unlabel_.append(correct_target/(target_unlabel_x.shape[0]))
    
    '''
    # Update GAN
    # Update global Discriminator
    CNet.train()
    encoder.train()
    GNet.train()
    DNet_local.train()
    DNet_global.train()
    total_error_D_global = 0
    total_error_G = 0
    for batch_id, ((source_x, source_y), (target_x, target_y)) in tqdm(enumerate(join_dataloader), total=len(join_dataloader)):
        """Update D Net"""
        optimizerD_global.zero_grad()
        source_data = source_x.to(device).float()
        source_embedding = encoder_inference(encoder, source_data)
        target_data = target_x.to(device).float()
        target_embedding = encoder_inference(encoder, target_data)
        fake_source_embedding = GNet(target_embedding).detach()
        
        # adversarial loss
        loss_D_global = DNet_global(fake_source_embedding,1).mean() - DNet_global(source_embedding,1).mean()
        
        total_error_D_global += loss_D_global.item()
        
        loss_D_global.backward()
        optimizerD_global.step()
        
        # Clip weights of discriminator
        for p in DNet_global.parameters():
            p.data.clamp_(-args.clip_value, args.clip_value)
        
        if batch_id % args.n_critic == 0:
            """Update G Network"""
            optimizerG.zero_grad()
            optimizerEncoder.zero_grad()
            fake_source_embedding = GNet(target_embedding)
            
            # adversarial loss
            loss_G = -DNet_global(fake_source_embedding,1).mean()
            
            total_error_G += loss_G.item()
            
            loss_G.backward()
            optimizerG.step()
#             optimizerEncoder.step()
            
    logger.info('Epoch: %i, Global Discrimator Updates: Loss D_global: %f, Loss G: %f'%(epoch+1, total_error_D_global, total_error_G))
    error_D_global.append(total_error_D_global)
    error_G_global.append(total_error_G)
    '''
    
    # Update local Discriminator
    total_error_D_local = 0
    total_error_G = 0
    encoder.eval()
    GNet.train()
    DNet_local.train()
    for batch_id in tqdm(range(math.ceil(target_len/args.batch_size))):
        target_x, target_y, target_weight = get_batch_target_data_on_class(target_dict, pesudo_dict, target_unlabel_x, args.num_per_class, no_pesudo=True)
        source_x, source_y = get_batch_source_data_on_class(source_dict, args.num_per_class)
        
        source_x = torch.Tensor(source_x).to(device).float()
        target_x = torch.Tensor(target_x).to(device).float()
        source_y = torch.LongTensor(target_y).to(device)
        target_y = torch.LongTensor(target_y).to(device)
        target_weight = torch.Tensor(target_weight).to(device)
        source_mask = torch.zeros(source_x.size(0), num_class).to(device).scatter_(1, source_y.unsqueeze(-1), 1)
        target_mask = torch.zeros(target_x.size(0), num_class).to(device).scatter_(1, target_y.unsqueeze(-1), 1)
        target_weight = torch.zeros(target_x.size(0), num_class).to(device).scatter_(1, target_y.unsqueeze(-1), target_weight.unsqueeze(-1))
    
        """Update D Net"""
        optimizerD_local.zero_grad()
        source_embedding = encoder_inference(encoder, source_x)
        target_embedding = encoder_inference(encoder, target_x)
        fake_source_embedding = GNet(target_embedding).detach()
        
        # adversarial loss
        source_DNet_local = DNet_local(source_embedding, source_mask)
        target_DNet_local = DNet_local(fake_source_embedding, target_mask)
        
        source_weight_count = source_mask.sum(dim=0)
        target_weight_count = target_weight.sum(dim=0)
        
        source_DNet_local_mean = source_DNet_local.sum(dim=0) / source_weight_count
        target_DNet_local_mean = (target_DNet_local * target_weight).sum(dim=0) / target_weight_count        
        
        loss_D_local = (target_DNet_local_mean - source_DNet_local_mean).sum()
        loss_D_local = loss_D_local * args.dlocal
        
        total_error_D_local += loss_D_local.item()
        
        loss_D_local.backward()
        optimizerD_local.step()
        
        # Clip weights of discriminator
        for p in DNet_local.parameters():
            p.data.clamp_(-args.clip_value, args.clip_value)
        
        if batch_id % args.n_critic == 0:
            """Update G Network"""
            optimizerG.zero_grad()
            optimizerEncoder.zero_grad()
            fake_source_embedding = GNet(target_embedding)
            
            # adversarial loss
            target_DNet_local = DNet_local(fake_source_embedding, target_mask)
            target_DNet_local_mean = (target_DNet_local * target_weight).sum(dim=0) / target_weight_count        

            loss_G = -target_DNet_local_mean.sum() 
            loss_G = loss_G * args.dlocal
            
            total_error_G += loss_G.item()
            
            loss_G.backward()
            optimizerG.step()
#             optimizerEncoder.step()
            
    logger.info('Epoch: %i, Local Discrimator Updates: Loss D_global: %f, Loss G: %f'%(epoch+1, total_error_D_local, total_error_G))
    error_D_local.append(total_error_D_local)
    error_G_global.append(total_error_G)
    
    np.save(args.save_path+model_sub_folder+'/target_acc_label_.npy',target_acc_label_)
    np.save(args.save_path+model_sub_folder+'/source_acc_.npy',source_acc_)
    np.save(args.save_path+model_sub_folder+'/target_acc_unlabel_.npy',target_acc_unlabel_)
#     np.save(args.save_path+model_sub_folder+'/error_D_global.npy',error_D_global)
#     np.save(args.save_path+model_sub_folder+'/error_G_global.npy',error_G_global)
    np.save(args.save_path+model_sub_folder+'/error_D_local.npy',error_D_local)
    np.save(args.save_path+model_sub_folder+'/error_G_local.npy',error_G_local)

    if epoch % args.model_save_period == 0:
        torch.save(CNet.state_dict(), args.save_path+model_sub_folder+ '/CNet_%i.t7'%(epoch+1))
        torch.save(GNet.state_dict(), args.save_path+model_sub_folder+ '/GNet_%i.t7'%(epoch+1))
#         torch.save(encoder.state_dict(), args.save_path+model_sub_folder+ '/encoder_%i.t7'%(epoch+1))
#         torch.save(DNet_global.state_dict(),  args.save_path+model_sub_folder+ '/DNet_global_%i.t7'%(epoch+1))
        torch.save(DNet_local.state_dict(), args.save_path+model_sub_folder+ '/DNet_local_%i.t7'%(epoch+1))

