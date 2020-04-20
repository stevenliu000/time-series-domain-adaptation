
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
parser.add_argument('--lr_FNN', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--lr_centerloss', type=float, default=0.005, help='learning rate for centerloss')
parser.add_argument('--sclass', type=float, default=0.7, help='target classifier loss weight')
parser.add_argument('--scent', type=float, default=0.0001, help='source domain classification weight on centerloss')
parser.add_argument('--lr_gan', type=float, default=1e-3, help='learning rate for adversarial')
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of which target data has label')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of which source data has label')
parser.add_argument('--num_per_class', type=int, default=-1, help='number of sample per class when training local discriminator')
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--save_path', type=str, help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')
parser.add_argument('--sbinary_loss', type=float, default=1.0, help='weight for binary loss')
parser.add_argument('--epoch_begin_prototype', type=int, default=10, help='starting point to train on prob classifier.')
parser.add_argument('--pseudo_weight', type=float, default=0.5, help='pseudo weight for prob. classifier')
parser.add_argument('--pseudo_same_weight', type=float, default=0.1, help='pseudo weight for same class in prob. classifier')

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
    
model_sub_folder = '/checkpoint2/prob_classifier_pseudo_masking/task_%s_slp_%f_tlp_%f_sclass_%f_scent_%f_sbinary_loss_%f_pseudo_weight_%f_same_weight_%f'%(args.task, args.source_lbl_percentage, args.target_lbl_percentage, args.sclass, args.scent, args.sbinary_loss, args.pseudo_weight, args.pseudo_same_weight)

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


# # Model creation

# In[10]:


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
CNet = FNNLinear(d_h2=64*2, d_out=num_class).to(device)
GNet = Generator(dim=64*2).to(device)

criterion_centerloss = CenterLoss(num_classes=num_class, feat_dim=64*2, use_gpu=torch.cuda.is_available()).to(device)
criterion_classifier = nn.CrossEntropyLoss().to(device)
criterion_probloss = BinaryLossNewImplementation(device).to(device)

encoder.apply(weights_init)
encoder_MLP.apply(weights_init)
CNet.apply(weights_init)
GNet.apply(weights_init)

optimizerCNet = torch.optim.Adam(CNet.parameters(), lr=args.lr_FNN)
optimizerEncoderMLP = torch.optim.Adam(encoder_MLP.parameters(), lr=args.lr_encoder)
optimizerEncoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)
optimizerGNet = torch.optim.Adam(GNet.parameters(), lr=args.lr_gan)
optimizerCenterLoss = torch.optim.Adam(criterion_centerloss.parameters(), lr=args.lr_centerloss)


# In[11]:


def classifier_inference(encoder, CNet, x):
    CNet.eval()
    encoder.eval()
    with torch.no_grad():
        embedding = encoder_inference(encoder, x)
        pred = CNet(embedding)
    return pred


# In[12]:


def encoder_inference(encoder, encoder_MLP, x):
    real = x[:,:,0].reshape(x.size(0), seq_len, feature_dim).float()
    imag = x[:,:,1].reshape(x.size(0), seq_len, feature_dim).float()
    real, imag = encoder(real, imag)
    cat_embedding = torch.cat((real[:,-1,:], imag[:,-1,:]), -1).reshape(x.shape[0], -1)
    cat_embedding = encoder_MLP(cat_embedding)
    return cat_embedding


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
    CNet.train()
    encoder.train()
    encoder_MLP.train()
    source_acc_label = 0.0
    num_datas = 0.0
    for batch_id, (source_x, source_y) in tqdm(enumerate(labeled_source_dataloader), total=len(labeled_source_dataloader)):
        optimizerCNet.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerEncoderMLP.zero_grad()
        optimizerCenterLoss.zero_grad()
        source_x = source_x.to(device).float()
        source_y = source_y.to(device)
        num_datas += source_x.size(0)
        source_x_embedding = encoder_inference(encoder, encoder_MLP, source_x)
        pred = CNet(source_x_embedding)
        source_acc_label += (pred.argmax(-1) == source_y).sum().item()
        loss = (criterion_classifier(pred, source_y) +
                criterion_centerloss(source_x_embedding, source_y) * args.scent) * args.sclass
        loss.backward()
        optimizerCNet.step()
        optimizerCenterLoss.step()
        optimizerEncoderMLP.step()
        optimizerEncoder.step()
        
    source_acc_label = source_acc_label / num_datas
    source_acc_label_.append(source_acc_label)
    
    # on target domain
    CNet.train()
    encoder.train()
    encoder_MLP.train()
    GNet.train()
    target_acc_label = 0.0
    num_datas = 0.0
    for batch_id, (target_x, target_y) in tqdm(enumerate(labeled_target_dataloader), total=len(labeled_target_dataloader)):
        optimizerCNet.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerEncoderMLP.zero_grad()
        optimizerGNet.zero_grad()
        target_x = target_x.to(device).float()
        target_y = target_y.to(device)
        num_datas += target_x.size(0)
        target_x_embedding = encoder_inference(encoder, encoder_MLP, target_x)
        fake_x_embedding = GNet(target_x_embedding)
        pred = CNet(fake_x_embedding)
        target_acc_label += (pred.argmax(-1) == target_y).sum().item()
        loss = criterion_classifier(pred, target_y) 
        loss.backward()
        optimizerCNet.step()
        optimizerGNet.step()
        optimizerEncoderMLP.step()
        optimizerEncoder.step()
        
    target_acc_label = target_acc_label / num_datas
    target_acc_label_.append(target_acc_label)
    
    # prob. classifier
    if epoch >= args.epoch_begin_prototype:
        encoder.train()
        encoder_MLP.train()
        GNet.train()
              
        # TODO: fix range 
        for batch_id in tqdm(range(len(join_dataloader))):
            optimizerGNet.zero_grad()
            optimizerEncoder.zero_grad()
            optimizerEncoderMLP.zero_grad()
            
            target_x, target_y, target_weight = get_batch_target_data_on_class(target_labeled_dict, args.num_per_class, pesudo_dict, no_pesudo=True)
            source_x, source_y = get_batch_source_data_on_class(source_labeled_dict, args.num_per_class)
    
            source_x = torch.Tensor(source_x).to(device).float()
            target_x = torch.Tensor(target_x).to(device).float()
            source_y = torch.LongTensor(target_y).to(device)
            target_y = torch.LongTensor(target_y).to(device)
            
            source_x_embedding = encoder_inference(encoder, encoder_MLP, source_x)
            target_x_embedding = encoder_inference(encoder, encoder_MLP, target_x)
            fake_source_embedding = GNet(target_x_embedding)
            
            loss = args.sbinary_loss * criterion_probloss(fake_source_embedding, source_x_embedding, num_class, args.num_per_class)
            loss.backward()
            optimizerGNet.step()
            optimizerEncoderMLP.step()
            optimizerEncoder.step()
            
        # pseudo label
        for batch_id in tqdm(range(len(join_dataloader))):
            optimizerGNet.zero_grad()
            optimizerEncoder.zero_grad()
            optimizerEncoderMLP.zero_grad()
            
            target_x, target_y, target_weight = get_batch_target_data_on_class(target_labeled_dict, args.num_per_class, pesudo_dict, no_pesudo=False)
            source_x, source_y = get_batch_source_data_on_class(source_labeled_dict, args.num_per_class)
    
            source_x = torch.Tensor(source_x).to(device).float()
            target_x = torch.Tensor(target_x).to(device).float()
            source_y = torch.LongTensor(target_y).to(device)
            target_y = torch.LongTensor(target_y).to(device)
            
            source_x_embedding = encoder_inference(encoder, encoder_MLP, source_x)
            target_x_embedding = encoder_inference(encoder, encoder_MLP, target_x)
            fake_source_embedding = GNet(target_x_embedding)
            
            mask = torch.ones([args.num_per_class * num_class, args.num_per_class * num_class])
            for i in range(num_class):
                mask[i*args.num_per_class:(i+1)*args.num_per_class, i*args.num_per_class:(i+1)*args.num_per_class] = args.pseudo_same_weight
            
            pseudo_weight = min(args.pseudo_weight * target_acc_label / 0.9, args.pseudo_weight)
            loss = pseudo_weight * args.sbinary_loss * criterion_probloss(fake_source_embedding, source_x_embedding, num_class, args.num_per_class, mask=mask)
            loss.backward()
            optimizerGNet.step()
            optimizerEncoderMLP.step()
            optimizerEncoder.step()
            
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
    
    # target_domain
    target_acc_unlabel = 0.0
    num_datas = 0.0
    CNet.eval()
    encoder.eval()
    encoder_MLP.eval()
    
    target_pseudo_x = torch.empty(0).to(device)
    target_pseudo_y = torch.empty(0).long().to(device)
    
    for batch_id, (target_x, target_y) in tqdm(enumerate(unlabeled_target_dataloader), total=len(unlabeled_target_dataloader)):
        target_x = target_x.to(device).float()
        target_y = target_y.to(device)
        num_datas += target_x.shape[0]
        target_pseudo_x = torch.cat([target_pseudo_x, target_x])
        target_x_embedding = encoder_inference(encoder, encoder_MLP, target_x)
        fake_x_embedding = GNet(target_x_embedding)
        pred = CNet(fake_x_embedding)
        target_acc_unlabel += (pred.argmax(-1) == target_y).sum().item()
        target_pseudo_y = torch.cat([target_pseudo_y, pred.argmax(-1)])
    
    target_pseudo_x = target_pseudo_x.cpu().numpy()
    target_pseudo_y = target_pseudo_y.cpu().numpy()
    
    pesudo_dict = get_class_data_dict(target_pseudo_x, target_pseudo_y, num_class)
    
    target_acc_unlabel = target_acc_unlabel/num_datas
    target_acc_unlabel_.append(target_acc_unlabel)
    
    if epoch % args.model_save_period == 0:
        torch.save(encoder.state_dict(), args.save_path+model_sub_folder+ '/encoder_%i.t7'%(epoch+1))
        torch.save(encoder_MLP.state_dict(), args.save_path+model_sub_folder+ '/encoder_MLP%i.t7'%(epoch+1))
        torch.save(CNet.state_dict(), args.save_path+model_sub_folder+ '/CNet_%i.t7'%(epoch+1))
        torch.save(GNet.state_dict(), args.save_path+model_sub_folder+ '/GNet_%i.t7'%(epoch+1))
        torch.save(criterion_centerloss.state_dict(), args.save_path+model_sub_folder+ '/centerloss_%i.t7'%(epoch+1))
    if epoch == args.epoch_begin_prototype:
        logger.info('Epochs %i: Start prob. classifier'%(epoch+1))
    logger.info('Epochs %i: src labeled acc: %f; src unlabeled acc: %f; tgt labeled acc: %f; tgt unlabeled acc: %f'%(epoch+1, source_acc_label, source_acc_unlabel, target_acc_label, target_acc_unlabel))
    np.save(args.save_path+model_sub_folder+'/target_acc_label_.npy', target_acc_label_)
    np.save(args.save_path+model_sub_folder+'/target_acc_unlabel_.npy', target_acc_unlabel_)
    np.save(args.save_path+model_sub_folder+'/source_acc_label_.npy', source_acc_label_)
    np.save(args.save_path+model_sub_folder+'/source_acc_unlabel_.npy', source_acc_unlabel_)
    
    

