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
from FNNLinear import FNNLinear
from FNNSeparated import FNNSeparated
from GAN import Generator, Discriminator
from data_utils import *
from GAN import Generator, Discriminator
from data_utils import *
import argparse
import logging
import logging.handlers
import pickle
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from centerloss import CenterLoss


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


# # Parser

# In[4]:


# Parameters
parser = argparse.ArgumentParser(description='JDA Time series adaptation')
parser.add_argument("--data_path", type=str, default="../data_unzip/", help="dataset path")
parser.add_argument("--task", type=str, help='3A or 3E')
parser.add_argument('--batch_size', type=int, default=400, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr_gan', type=float, default=1e-3, help='learning rate for adversarial')
parser.add_argument('--lr_FNN', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--n_critic', type=float, default=0.16, help='gap: Generator train GAP times, discriminator train once')
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of target labeled data')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of source labeled data')
parser.add_argument('--num_per_class', type=int, default=2, help='number of sample per class when training local discriminator')
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--save_path', type=str, default="../train_related/", help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')
parser.add_argument('--gpweight', type=float, default=10, help='clip_value for WGAN')
parser.add_argument('--sclass', type=float, default=0.7, help='source domain classification weight on loss function')
parser.add_argument('--dlocal', type=float, default=0.01, help='local GAN weight on loss function')
parser.add_argument('--GANweights', type=int, default=-1, help='pretrained GAN weight')
parser.add_argument('--isglobal', type=int, default=0, help='if using global DNet')
parser.add_argument('--lr_centerloss', type=float, default=1e-3, help='center loss weight')
parser.add_argument('--pure_random', type=int, default=1, help='Pure random for n_critic')


args = parser.parse_args()
args.isglobal = True if args.isglobal == 1 else False
args.pure_random = True if args.pure_random == 1 else False


# In[21]:


# # local only
# class local_args:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
        
# args = local_args(**{
#     'data_path': '/home/tianqinl/Code/time-series-domain-adaptation/data_unzip',
#     'task': '3E',
#     'num_class': 50,
#     'batch_size': 100,
#     'num_per_class': 2,
#     'gap': 5,
#     'lbl_percentage':0.2,
#     'lr_gan': 1e-4,
#     'lr_FNN': 1e-4,
#     'lr_encoder': 1e-4,
#     'epochs': 2,
#     'clip_value': 0.01,
#     'n_critic': 0.16,
#     'save_path': '/home/tianqinl/Code/time-series-domain-adaptation/train_related/',
#     'target_lbl_percentage': 0.7,
#     'source_lbl_percentage': 0.7,
#     'lr_centerloss': 1e-3,
#     'sclass': 0.7,
#     'isglobal': True,
#     'gpweight': 10,
#     'dlocal': 1e-2,
#     'model_save_period': 2,
#     'pure_random': True,
# })


# In[12]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

args.task = '3Av2' if args.task == '3A' else '3E'
num_class = 50 if args.task == "3Av2" else 65
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.num_per_class == -1:
    args.num_per_class = math.ceil(args.batch_size / num_class)
    
model_sub_folder = 'Linear_GAN/task_%s_gpweight_%f_dlocal_%f_critic_%f_rand_%i_sclass_%f'%(args.task, args.gpweight, args.dlocal, args.n_critic, args.pure_random, args.sclass)

save_folder = os.path.join(args.save_path, model_sub_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)   


# # Logger

# In[13]:


logger = logging.getLogger()
logger.setLevel(logging.INFO)

if os.path.isfile(os.path.join(save_folder, 'logfile.log')):
    os.remove(os.path.join(save_folder, 'logfile.log'))

file_log_handler = logging.FileHandler(os.path.join(save_folder, 'logfile.log'))
logger.addHandler(file_log_handler)

stdout_log_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_log_handler)

attrs = vars(args)
for item in attrs.items():
    logger.info("%s: %s"%item)


# # Data loading

# In[14]:


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

label_target_len = labeled_target_x.shape[0]

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


join_dataset = JoinDataset(labeled_source_x, labeled_source_y, labeled_target_x, labeled_target_y, random=True)
join_dataloader = DataLoader(join_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

source_labeled_dict = get_class_data_dict(labeled_source_x, labeled_source_y, num_class)
target_labeled_dict = get_class_data_dict(labeled_target_x, labeled_target_y, num_class)


# # weight Initialize

# In[15]:


def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LayerNorm:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# # model creation

# In[16]:


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
encoder_MLP = FNNSeparated(d_in=64 * 2 * 1, d_h1=64*4, d_h2=64*2, dp=0.2).to(device)
CNet = FNNLinear(d_h2=64*2, d_out=num_class).to(device)

DNet_global = Discriminator(feature_dim=64*2, d_out=1).to(device)
DNet_local = Discriminator(feature_dim=64*2, d_out=num_class).to(device)
GNet = Generator(dim=64*2).to(device)


criterion_centerloss = CenterLoss(num_classes=num_class, feat_dim=64*2, use_gpu=torch.cuda.is_available()).to(device)
criterion_classifier = nn.CrossEntropyLoss().to(device)


DNet_global.apply(weights_init)
DNet_local.apply(weights_init)
GNet.apply(weights_init)
encoder.apply(weights_init)
encoder_MLP.apply(weights_init)
CNet.apply(weights_init)
optimizerD_global = torch.optim.Adam(DNet_global.parameters(), lr=args.lr_gan)
optimizerD_local = torch.optim.Adam(DNet_local.parameters(), lr=args.lr_gan)
optimizerG = torch.optim.Adam(GNet.parameters(), lr=args.lr_gan)
optimizerCNet = torch.optim.Adam(CNet.parameters(), lr=args.lr_FNN)
optimizerEncoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)
optimizerEncoderMLP = torch.optim.Adam(encoder_MLP.parameters(), lr=args.lr_encoder)
optimizerCenterLoss = torch.optim.Adam(criterion_centerloss.parameters(), lr=args.lr_centerloss)


# In[17]:


def classifier_inference(encoder, CNet, x):
    CNet.eval()
    encoder.eval()
    with torch.no_grad():
        embedding = encoder_inference(encoder, x)
        pred = CNet(embedding)
    return pred


# In[18]:


def encoder_inference(encoder, encoder_MLP, x):
    real = x[:,:,0].reshape(x.size(0), seq_len, feature_dim).float()
    imag = x[:,:,1].reshape(x.size(0), seq_len, feature_dim).float()
    real, imag = encoder(real, imag)
    cat_embedding = torch.cat((real[:,-1,:], imag[:,-1,:]), -1).reshape(x.shape[0], -1)
    cat_embedding = encoder_MLP(cat_embedding)
    return cat_embedding


# In[19]:


def _gradient_penalty(real_data, generated_data, DNet, mask, num_class, device, args):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = DNet(interpolated, mask)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1, keepdim=True) + 1e-12)

    gradients_norm = gradients_norm * mask
    # Return gradient penalty
    return args.gpweight * ((gradients_norm - 1) ** 2).mean(dim=0)


# # Train

# In[22]:


target_acc_label_ = []
source_acc_ = []
source_acc_unlabel_ = []
target_acc_unlabel_ = []
error_D_global = []
error_G_global = []
error_D_local = []
error_G_local = []


# # pre-trained
# model_PATH = '../train_related/spring_break/stage2/task_3E_SClassWeight_0.700000_old'
# CNet.load_state_dict(torch.load(model_PATH+'/CNet_pre_trained.t7', map_location=device))
# encoder.load_state_dict(torch.load(model_PATH+'/encoder_pre_trained.t7', map_location=device))
# GNet.load_state_dict(torch.load(model_PATH+'/GNet_pre_trained.t7', map_location=device))
# print('Model Loaded!')

# if args.GANweights !='-1':
#     GAN_PATH = '../train_related/spring_break/stage3_local_gp_no_class/task_3E_gpweight_20.000000_dlocal_0.010000_critic_0.160000_sclass_0.700000'
#     GNet.load_state_dict(torch.load(GAN_PATH+'/GNet_%i.t7'%args.GANweights, map_location=device))
#     DNet_local.load_state_dict(torch.load(GAN_PATH+'/DNet_local_%i.t7'%args.GANweights, map_location=device))

print('Started training')
for epoch in range(args.epochs):
    # update classifier
    # on source domain
    CNet.train()
    encoder.train()
    encoder_MLP.train()
    GNet.train()
    source_acc = 0.0
    num_datas = 0.0
    for batch_id, (source_x, source_y) in tqdm(enumerate(labeled_source_dataloader), total=len(labeled_source_dataloader)):
        optimizerCNet.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerEncoderMLP.zero_grad()
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)
        source_x_embedding = encoder_inference(encoder, encoder_MLP, source_x)
        pred = CNet(source_x_embedding)
        source_acc += (pred.argmax(-1) == source_y).sum().item()
        loss = criterion_classifier(pred, source_y) * args.sclass
        loss.backward()
        optimizerCNet.step()
        optimizerEncoder.step()
        optimizerEncoderMLP.step()

        
    source_acc = source_acc / num_datas
    source_acc_.append(source_acc)
    
    
    # on target domain
    target_acc = 0.0
    num_datas = 0.0
    CNet.train()
    encoder.train()
    encoder_MLP.train()
    GNet.train()
    for batch_id, (target_x, target_y) in tqdm(enumerate(labeled_target_dataloader), total=len(labeled_target_dataloader)):
        optimizerCNet.zero_grad()
        optimizerG.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerEncoderMLP.zero_grad()
        target_x = target_x.to(device).float()
        target_y = target_y.to(device)
        num_datas += target_x.size(0)
        target_x_embedding = encoder_inference(encoder, encoder_MLP, target_x)
        fake_source_embedding = GNet(target_x_embedding)
        pred = CNet(fake_source_embedding)
        target_acc += (pred.argmax(-1) == target_y).sum().item()
        loss = criterion_classifier(pred, target_y)
        loss.backward()
        optimizerCNet.step()
        optimizerG.step()
        optimizerEncoder.step()
        optimizerEncoderMLP.step()

    
    target_acc = target_acc / num_datas
    target_acc_label_.append(target_acc)
    
    # eval
    # source_domain
    source_acc_unlabel = 0.0
    num_datas = 0.0
    CNet.eval()
    encoder.eval()
    encoder_MLP.eval()
    for batch_id, (source_x, source_y) in tqdm(enumerate(unlabeled_source_dataloader), total=len(unlabeled_source_dataloader)):
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.shape[0]
        source_x_embedding = encoder_inference(encoder, encoder_MLP, source_x)
        pred = CNet(source_x_embedding)
        source_acc_unlabel += (pred.argmax(-1) == source_y).sum().item()
        
    source_acc_unlabel = source_acc_unlabel/num_datas
    source_acc_unlabel_.append(source_acc_unlabel)
    
    # Assign Pesudo Label
    correct_target = 0.0
    target_pesudo_y = []
    CNet.eval()
    encoder.eval()
    encoder_MLP.eval()
    GNet.eval()
    for batch in range(math.ceil(unlabeled_target_x.shape[0]/args.batch_size)):
        target_unlabel_x_batch = torch.Tensor(unlabeled_target_x[batch*args.batch_size:(batch+1)*args.batch_size]).to(device).float()
        target_unlabel_y_batch = torch.Tensor(unlabeled_target_y[batch*args.batch_size:(batch+1)*args.batch_size]).to(device)        
        # print(target_unlabel_y_batch.shape)
        
        target_unlabel_x_embedding = encoder_inference(encoder, encoder_MLP, target_unlabel_x_batch)
        fake_source_embedding = GNet(target_unlabel_x_embedding)
        pred = CNet(fake_source_embedding)
        correct_target += (pred.argmax(-1) == target_unlabel_y_batch).sum().item()
        target_pesudo_y.extend(pred.argmax(-1).cpu().numpy())
        
    target_pesudo_y = np.array(target_pesudo_y)
    pesudo_dict = get_class_data_dict(unlabeled_target_x, target_pesudo_y, num_class)
    target_acc_unlabel = correct_target/(unlabeled_target_x.shape[0])
    target_acc_unlabel_.append(target_acc_unlabel)
    
    logger.info('Epoch: %i, update classifier: source acc: %f; source unlbl acc: %f; target acc: %f; target unlabel acc: %f'%(epoch+1, source_acc, source_acc_unlabel, target_acc, target_acc_unlabel))

    # Update GAN
    if args.isglobal:
        # Update global Discriminator
        CNet.train()
        encoder.train()
        encoder_MLP.train()
        GNet.train()
        DNet_local.train()
        DNet_global.train()
        total_error_D_global = 0
        total_error_G = 0
        for batch_id, ((source_x, source_y), (target_x, target_y)) in tqdm(enumerate(join_dataloader), total=len(join_dataloader)):
            """Update D Net"""
            optimizerD_global.zero_grad()
            source_data = source_x.to(device).float()
            source_embedding = encoder_inference(encoder, encoder_MLP, source_data)
            target_data = target_x.to(device).float()
            target_embedding = encoder_inference(encoder, encoder_MLP, target_data)
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
                # optimizerEncoder.step()

        
        logger.info('Epoch: %i, Global Discrimator Updates: Loss D_global: %f, Loss G: %f'%(epoch+1, total_error_D_global, total_error_G))

        error_D_global.append(total_error_D_global)
        error_G_global.append(total_error_G)


    # Update local Discriminator
    total_error_D_local = 0
    total_error_G = 0
    encoder.eval()
    encoder_MLP.eval()
    GNet.train()
    DNet_local.train()
    
    update_id = int(1/args.n_critic)
    if args.pure_random:
        update_id = max(1, update_id + np.random.randint(-2,2,1)[0])
#     elif args.adaptive_random:
#         if batch_id > 10:

#             # mean of last 10 trail
#             sd_Gloss = np.std(error_G_local[-10:])
#             sd_Dlocalloss = np.std(error_D_local[-10:])
#             update_id = update_id + np.random.randint(-2,2,1)[0]
    for batch_id in tqdm(range(math.ceil(label_target_len/args.batch_size))):
        target_x, target_y, target_weight = get_batch_target_data_on_class(target_labeled_dict, args.num_per_class, pesudo_dict, no_pesudo=True)
        source_x, source_y = get_batch_source_data_on_class(source_labeled_dict, args.num_per_class)
        
        source_x = torch.Tensor(source_x).to(device).float()
        target_x = torch.Tensor(target_x).to(device).float()
        source_y = torch.LongTensor(target_y).to(device)
        target_y = torch.LongTensor(target_y).to(device)
        target_weight = torch.Tensor(target_weight).to(device)
        source_mask = torch.zeros(source_x.size(0), num_class).to(device).scatter_(1, source_y.unsqueeze(-1), 1)
        target_mask = torch.zeros(target_x.size(0), num_class).to(device).scatter_(1, target_y.unsqueeze(-1), 1)
        target_weight = torch.zeros(target_x.size(0), num_class).to(device).scatter_(1, target_y.unsqueeze(-1), target_weight.unsqueeze(-1))
        

        source_weight_count = source_mask.sum(dim=0)
        target_weight_count = target_weight.sum(dim=0)
    
        if args.n_critic > 1:
            """Update D Net"""
            optimizerD_local.zero_grad()
            source_embedding = encoder_inference(encoder, encoder_MLP, source_x)
            target_embedding = encoder_inference(encoder, encoder_MLP, target_x)
            fake_source_embedding = GNet(target_embedding).detach()

            # adversarial loss
            source_DNet_local = DNet_local(source_embedding, source_mask)
            target_DNet_local = DNet_local(fake_source_embedding, target_mask)

            source_weight_count = source_mask.sum(dim=0)
            target_weight_count = target_weight.sum(dim=0)

            source_DNet_local_mean = source_DNet_local.sum(dim=0) / source_weight_count
            target_DNet_local_mean = (target_DNet_local * target_weight).sum(dim=0) / target_weight_count        

            gp = _gradient_penalty(source_embedding, fake_source_embedding, DNet_local, source_mask, num_class, device, args)

            loss_D_local = (target_DNet_local_mean - source_DNet_local_mean + gp).sum()
            loss_D_local = loss_D_local * args.dlocal

            total_error_D_local += loss_D_local.item()

            loss_D_local.backward()
            optimizerD_local.step()

            if batch_id % args.n_critic == 0:
                """Update G Network"""
                optimizerG.zero_grad()
                optimizerEncoder.zero_grad()
                optimizerEncoderMLP.zero_grad()
                target_embedding = encoder_inference(encoder, encoder_MLP, target_x)
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
        else:
            """Update G Network"""
            optimizerG.zero_grad()
            optimizerEncoder.zero_grad()
            optimizerEncoderMLP.zero_grad()
            target_embedding = encoder_inference(encoder, encoder_MLP, target_x)
            fake_source_embedding = GNet(target_embedding)

            # adversarial loss
            target_DNet_local = DNet_local(fake_source_embedding, target_mask)
            target_DNet_local_mean = (target_DNet_local * target_weight).sum(dim=0) / target_weight_count        

            loss_G = -target_DNet_local_mean.sum() 
            loss_G = loss_G * args.dlocal

            total_error_G += loss_G.item()

            loss_G.backward()
            optimizerG.step()
            
            
            if batch_id % update_id == 0:
                """Update D Net"""
                optimizerD_local.zero_grad()
                source_embedding = encoder_inference(encoder, encoder_MLP, source_x)
                target_embedding = encoder_inference(encoder, encoder_MLP, target_x)
                fake_source_embedding = GNet(target_embedding).detach()

                # adversarial loss
                source_DNet_local = DNet_local(source_embedding, source_mask)
                target_DNet_local = DNet_local(fake_source_embedding, target_mask)

                source_DNet_local_mean = source_DNet_local.sum(dim=0) / source_weight_count
                target_DNet_local_mean = (target_DNet_local * target_weight).sum(dim=0) / target_weight_count        

                gp = _gradient_penalty(source_embedding, fake_source_embedding, DNet_local, source_mask, num_class, device, args)

                loss_D_local = (target_DNet_local_mean - source_DNet_local_mean + gp).sum()
                loss_D_local = loss_D_local * args.dlocal

                total_error_D_local += loss_D_local.item()

                loss_D_local.backward()
                optimizerD_local.step()
            
    logger.info('Epoch: %i, Local Discrimator Updates: Loss D_local: %f, Loss G: %f; update_ratio: %i'%(epoch+1, total_error_D_local, total_error_G, update_id))
    error_D_local.append(total_error_D_local)
    error_G_global.append(total_error_G)
 
    np.save(os.path.join(args.save_path, model_sub_folder, 'target_acc_label_.npy'),target_acc_label_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_acc_.npy'),source_acc_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'target_acc_unlabel_.npy'),target_acc_unlabel_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_acc_unlabel_.npy'),source_acc_unlabel_)
    if args.isglobal:
        np.save(os.path.join(args.save_path, model_sub_folder, 'error_D_global.npy'),error_D_global)
        np.save(os.path.join(args.save_path, model_sub_folder, 'error_G_global.npy'),error_G_global)
    np.save(os.path.join(args.save_path, model_sub_folder, 'error_D_local.npy'),error_D_local)
    np.save(os.path.join(args.save_path, model_sub_folder, 'error_G_local.npy'),error_G_local)

    if epoch % args.model_save_period == 0:
        torch.save(CNet.state_dict(), os.path.join(args.save_path,model_sub_folder, 'CNet_%i.t7'%(epoch+1)))
        torch.save(GNet.state_dict(), os.path.join(args.save_path,model_sub_folder, 'GNet_%i.t7'%(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join(args.save_path,model_sub_folder,'encoder_%i.t7'%(epoch+1)))
        torch.save(encoder_MLP.state_dict(), os.path.join(args.save_path,model_sub_folder,'encoder_MLP_%i.t7'%(epoch+1)))
        if args.isglobal:
            torch.save(DNet_global.state_dict(),  os.path.join(args.save_path,model_sub_folder,'DNet_global_%i.t7'%(epoch+1)))
        torch.save(DNet_local.state_dict(), os.path.join(args.save_path,model_sub_folder, 'DNet_local_%i.t7'%(epoch+1)))


# In[37]:


np.random.randint(-3,3,1)[0]


# In[38]:


error_G_local


# In[43]:


np.std([1,2,3])


# In[ ]:




