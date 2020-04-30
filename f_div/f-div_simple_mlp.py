
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
parser.add_argument("--data_path", type=str, default="../data_unzip", help="dataset path")
parser.add_argument("--task", type=str, help='3A or 3E')
parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_centerloss', type=float, default=0.005, help='learning rate for centerloss')
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of which target data has label')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of which source data has label')
parser.add_argument('--num_per_class', type=int, default=-1, help='number of sample per class when training local discriminator')
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--save_path', type=str, default='../train_related/', help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')
parser.add_argument('--model_path', type=str, help='where the data is stored')
parser.add_argument('--intervals', type=int, default=2, help='freq of compute f-div')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--gfunction_epoch', type=int, default=5000, help='epoch of which gfunction is trained for')
parser.add_argument('--KL', type=bool, default=False, help="if calculate KL divergence")
parser.add_argument('--JS', type=bool, default=False, help="if calculate JS divergence")
parser.add_argument('--classifier', type=bool, default=False, help="if optmizer classifier")
parser.add_argument('--sclass', type=float, default=0.7, help='target classifier loss weight')
parser.add_argument('--scent', type=float, default=0.0001, help='source domain classification weight on centerloss')
parser.add_argument('--centerloss', type=bool, default=False, help='if use centerloss')
parser.add_argument('--classifier_epoch', type=int, default=10000, help='max iteration to train classifier')


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
    
model_sub_folder = '/f-gan/'+args.model_name
if args.KL: model_sub_folder += '_KL'
if args.JS: model_sub_folder += '_JS'   
if args.classifier: model_sub_folder += '_classifier'
model_sub_folder += '/'

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


# In[ ]:


class Gfunction(nn.Sequential):
    def __init__(self):
        super(Gfunction, self).__init__(
            nn.Linear(200,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100,1)
        )


# In[1]:


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


# In[52]:


device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)

encoder = SimpleMLP1().to(device)

if args.KL:
    gfunction_KL_div_labeled = Gfunction().to(device)
    gfunction_KL_div_unlabeled = Gfunction().to(device)

if args.JS:
    gfunction_JS_div_labeled = Gfunction().to(device)
    gfunction_JS_div_unlabeled = Gfunction().to(device)
    
if args.classifier:
    CNet = SimpleMLP2().to(device)
    criterion_classifier = nn.CrossEntropyLoss().to(device)


# # Train

# In[16]:


logger.info('Started loading')
source_acc_label_ = np.load(os.path.join(args.model_path, 'source_acc_label_.npy'))
source_acc_unlabel_ = np.load(os.path.join(args.model_path, 'source_acc_unlabel_.npy'))
target_acc_label_ = np.load(os.path.join(args.model_path, 'target_acc_label_.npy'))
target_acc_unlabel_ = np.load(os.path.join(args.model_path, 'target_acc_unlabel_.npy'))

labeled_KL = []
unlabeled_KL = []
labeled_JS = []
unlabeled_JS = []
acc_source_unlabeled_classifier_ = []
acc_target_unlabeled_classifier_ = []

source_acc_label = []
source_acc_unlabel = []
target_acc_label = []
target_acc_unlabel = []

epochs = []

for epoch in range(3, source_acc_label_.shape[0], args.intervals*args.model_save_period):
    # initialize 
    if args.KL:
        gfunction_KL_div_labeled.apply(weights_init)
        optimizer_gfunction_KL_div_labeled = torch.optim.Adam(gfunction_KL_div_labeled.parameters(), lr=args.lr)
        gfunction_KL_div_unlabeled.apply(weights_init)
        optimizer_gfunction_KL_div_unlabeled = torch.optim.Adam(gfunction_KL_div_unlabeled.parameters(), lr=args.lr)

    if args.JS:
        gfunction_JS_div_labeled.apply(weights_init)
        optimizer_gfunction_JS_div_labeled = torch.optim.Adam(gfunction_JS_div_labeled.parameters(), lr=args.lr)
        gfunction_JS_div_unlabeled.apply(weights_init)
        optimizer_gfunction_JS_div_unlabeled = torch.optim.Adam(gfunction_JS_div_unlabeled.parameters(), lr=args.lr)

    if args.classifier:
        CNet.load_state_dict(torch.load(os.path.join(args.model_path, 'CNet_%i.t7'%epoch)))
        optimizer_CNet = torch.optim.Adam(CNet.parameters(), lr=args.lr)
    
    # load weight
    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder_%i.t7'%epoch)))
    
    # inferencing
    encoder.eval()
    
    # get source/target embedding
    source_x_labeled_embedding = torch.empty(0).to(device)
    source_y_labeled = torch.empty(0).long().to(device)
    source_x_unlabeled_embedding = torch.empty(0).to(device)
    source_y_unlabeled = torch.empty(0).long().to(device)
    target_x_labeled_embedding = torch.empty(0).to(device)
    target_y_labeled = torch.empty(0).long().to(device)
    target_x_unlabeled_embedding = torch.empty(0).to(device)
    target_y_unlabeled = torch.empty(0).long().to(device)
    with torch.no_grad():
        for batch_id, (source_x, source_y) in tqdm(enumerate(labeled_source_dataloader), total=len(labeled_source_dataloader)):
            source_x = source_x.to(device).view(-1,3200).float()
            source_y = source_y.to(device).long()
            source_x_embedding = encoder(source_x).detach()
            source_x_labeled_embedding = torch.cat([source_x_labeled_embedding, source_x_embedding])
            source_y_labeled = torch.cat([source_y_labeled, source_y])
            
        for batch_id, (source_x, source_y) in tqdm(enumerate(unlabeled_source_dataloader), total=len(unlabeled_source_dataloader)):
            source_x = source_x.to(device).view(-1,3200).float()
            source_y = source_y.to(device).long()
            source_x_embedding = encoder(source_x).detach()
            source_x_unlabeled_embedding = torch.cat([source_x_unlabeled_embedding, source_x_embedding])
            source_y_unlabeled = torch.cat([source_y_unlabeled, source_y])
            
        for batch_id, (target_x, target_y) in tqdm(enumerate(labeled_target_dataloader), total=len(labeled_target_dataloader)):
            target_x = target_x.to(device).view(-1,3200).float()
            target_y = target_y.to(device).long()
            fake_x_embedding = encoder(target_x).detach()
            target_x_labeled_embedding = torch.cat([target_x_labeled_embedding, fake_x_embedding])     
            target_y_labeled = torch.cat([target_y_labeled, target_y])

            
        for batch_id, (target_x, target_y) in tqdm(enumerate(unlabeled_target_dataloader), total=len(unlabeled_target_dataloader)):
            target_x = target_x.to(device).view(-1,3200).float()
            target_y = target_y.to(device).long()
            fake_x_embedding = encoder(target_x).detach()
            target_x_unlabeled_embedding = torch.cat([target_x_unlabeled_embedding, fake_x_embedding])    
            target_y_unlabeled = torch.cat([target_y_unlabeled, target_y])
            
    # for loop to train the gfunction 
    for i in tqdm(range(args.gfunction_epoch)):
        if args.KL:
            optimizer_gfunction_KL_div_labeled.zero_grad()
            source_x_labeled_g = gfunction_KL_div_labeled(source_x_labeled_embedding)
            target_x_labeled_g = gfunction_KL_div_labeled(target_x_labeled_embedding)
            loss_KL_labeled = - KLDiv(source_x_labeled_g, target_x_labeled_g, device) # maximize
            loss_KL_labeled.backward()
            optimizer_gfunction_KL_div_labeled.step()
            if i % 500 == 0:
                print("Epoch %i, Iter %i, labeled KL: %f"%(epoch, i, -loss_KL_labeled.item()))
       
        if args.JS:
            optimizer_gfunction_JS_div_labeled.zero_grad()
            source_x_labeled_g = gfunction_JS_div_labeled(source_x_labeled_embedding)
            target_x_labeled_g = gfunction_JS_div_labeled(target_x_labeled_embedding)
            loss_JS_labeled = - JSDiv(source_x_labeled_g, target_x_labeled_g, device) # maximize
            loss_JS_labeled.backward()
            optimizer_gfunction_JS_div_labeled.step()
            if i % 500 == 0:
                print("Epoch %i, Iter %i, labeled JS: %f"%(epoch, i, -loss_JS_labeled.item()))
            
    if args.KL:
        loss_KL_labeled = - loss_KL_labeled.item()
        labeled_KL.append(loss_KL_labeled)
      
    if args.JS:
        loss_JS_labeled = - loss_JS_labeled.item()
        labeled_JS.append(loss_JS_labeled)
    
    for i in tqdm(range(args.gfunction_epoch)):
        if args.KL:
            optimizer_gfunction_KL_div_unlabeled.zero_grad()
            source_x_unlabeled_g = gfunction_KL_div_unlabeled(source_x_unlabeled_embedding)
            target_x_unlabeled_g = gfunction_KL_div_unlabeled(target_x_unlabeled_embedding)
            loss_KL_unlabeled = - KLDiv(source_x_unlabeled_g, target_x_unlabeled_g, device) # maximize
            loss_KL_unlabeled.backward()
            optimizer_gfunction_KL_div_unlabeled.step()
            if i % 500 == 0:
                print("Epoch %i, Iter %i, unlabeled KL: %f"%(epoch, i, -loss_KL_unlabeled.item()))

        if args.JS:
            optimizer_gfunction_JS_div_unlabeled.zero_grad()
            source_x_unlabeled_g = gfunction_JS_div_unlabeled(source_x_unlabeled_embedding)
            target_x_unlabeled_g = gfunction_JS_div_unlabeled(target_x_unlabeled_embedding)
            loss_JS_unlabeled = - JSDiv(source_x_unlabeled_g, target_x_unlabeled_g, device) # maximize
            loss_JS_unlabeled.backward()
            optimizer_gfunction_JS_div_unlabeled.step()
            if i % 500 == 0:
                print("Epoch %i, Iter %i, unlabeled JS: %f"%(epoch, i, -loss_JS_unlabeled.item()))
            
    if args.KL:  
        loss_KL_unlabeled = - loss_KL_unlabeled.item()
        unlabeled_KL.append(loss_KL_unlabeled)
    
    if args.JS:
        loss_JS_unlabeled = - loss_JS_unlabeled.item()
        unlabeled_JS.append(loss_JS_unlabeled)

    acc_source_labeled_classifier = 0
    acc_target_labeled_classifier = 0
    if args.classifier:
#         while i < args.classifier_epoch or (acc_source_labeled_classifier < 0.98 and acc_target_labeled_classifier < 0.98):
#             i += 1
        for i in tqdm(range(args.classifier_epoch)):
            CNet.train()
            optimizer_CNet.zero_grad()
            pred = CNet(source_x_labeled_embedding)
            acc_source_labeled_classifier = (pred.argmax(-1) == source_y_labeled).sum().item() / pred.size(0)
            loss_source_classifier_labeled = criterion_classifier(pred, source_y_labeled) * args.sclass
            if args.centerloss: loss_source_classifier_labeled += criterion_centerloss(source_x_labeled_embedding, source_y_labeled) * args.scent * args.sclass
            loss_source_classifier_labeled.backward()
            if args.centerloss: optimizer_centerloss.step()
            optimizer_CNet.step()
            
            optimizer_CNet.zero_grad()
            pred = CNet(target_x_labeled_embedding)
            acc_target_labeled_classifier = (pred.argmax(-1) == target_y_labeled).sum().item() / pred.size(0)
            loss_target_classifier_labeled = criterion_classifier(pred, target_y_labeled)
            loss_target_classifier_labeled.backward()
            optimizer_CNet.step()
            
#             if i % 500 == 0:
#                 CNet.eval()
#                 pred = CNet(source_x_unlabeled_embedding)
#                 acc_source_unlabeled_classifier = (pred.argmax(-1) == source_y_unlabeled).sum().item() / pred.size(0)
#                 pred = CNet(target_x_unlabeled_embedding)
#                 acc_target_unlabeled_classifier = (pred.argmax(-1) == target_y_unlabeled).sum().item() / pred.size(0)
#                 print("Iter %i: source acc: labeled: %f, unlabeled: %f; target acc: labeled: %f, unlabeled: %f"%(
#                     i, acc_source_labeled_classifier, acc_source_unlabeled_classifier, acc_target_labeled_classifier, acc_target_unlabeled_classifier))
        
        CNet.eval()
        pred = CNet(source_x_unlabeled_embedding)
        acc_source_unlabeled_classifier = (pred.argmax(-1) == source_y_unlabeled).sum().item() / pred.size(0)
        pred = CNet(target_x_unlabeled_embedding)
        acc_target_unlabeled_classifier = (pred.argmax(-1) == target_y_unlabeled).sum().item() / pred.size(0)
        acc_source_unlabeled_classifier_.append(acc_source_unlabeled_classifier)
        acc_target_unlabeled_classifier_.append(acc_target_unlabeled_classifier)
        
    # save corresponding acc
    source_acc_label.append(source_acc_label_[epoch-1])
    source_acc_unlabel.append(source_acc_unlabel_[epoch-1])
    target_acc_label.append(target_acc_label_[epoch-1])
    target_acc_unlabel.append(target_acc_unlabel_[epoch-1])
    epochs.append(epoch)
    
    logger.info("-----------------------------------------")
    log_string = "Epoch %i: "%epoch
    if args.KL: log_string += "labeled KL: %f, unlabeled KL: %f; "%(loss_KL_labeled, loss_KL_unlabeled)
    if args.JS: log_string += "labeled JS: %f, unlabeled JS: %f; "%(loss_JS_labeled, loss_JS_unlabeled)   
    if args.classifier: log_string += "src unlbl acc: %f, tgt unlbl acc: %f; "%(acc_source_unlabeled_classifier, acc_target_unlabeled_classifier)      
    logger.info(log_string)
    logger.info("-----------------------------------------")
    
    np.save(args.save_path+model_sub_folder+'/epochs.npy', epochs)
    np.save(args.save_path+model_sub_folder+'/source_acc_label.npy', source_acc_label)
    np.save(args.save_path+model_sub_folder+'/source_acc_unlabel.npy', source_acc_unlabel)
    np.save(args.save_path+model_sub_folder+'/target_acc_label.npy', target_acc_label)
    np.save(args.save_path+model_sub_folder+'/target_acc_unlabel.npy', target_acc_unlabel)
    
    if args.KL: 
        np.save(args.save_path+model_sub_folder+'/labeled_KL.npy', labeled_KL)
        np.save(args.save_path+model_sub_folder+'/unlabeled_KL.npy', unlabeled_KL)
        
    if args.JS:
        np.save(args.save_path+model_sub_folder+'/labeled_JS.npy', labeled_JS)
        np.save(args.save_path+model_sub_folder+'/unlabeled_JS.npy', unlabeled_JS)
        
    if args.classifier:
        np.save(args.save_path+model_sub_folder+'/acc_source_unlabeled_classifier_.npy', acc_source_unlabeled_classifier_)
        np.save(args.save_path+model_sub_folder+'/acc_target_unlabeled_classifier_.npy', acc_target_unlabeled_classifier_)
    
