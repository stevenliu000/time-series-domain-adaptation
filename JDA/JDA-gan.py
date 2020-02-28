
# coding: utf-8

# In[34]:


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
from dataset import TimeSeriesDataset, TimeSeriesDatasetConcat
from martins.complex_transformer import ComplexTransformer
import argparse
import os

# In[33]:


# Parameters
parser = argparse.ArgumentParser(description='JDA Time series adaptation')
parser.add_argument("--data_path", type=str, default="/projects/rsalakhugroup/complex/domain_adaptation", help="dataset path")
parser.add_argument("--task", type=str, help='3A or 3E')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr_gan', type=float, default=1e-4, help='learning rate for adversarial')
parser.add_argument('--lr_clf', type=float, default=1e-4, help='learning rate for classification')
parser.add_argument('--gap', type=int, default=4, help='gap: Generator train GAP times, discriminator train once')
parser.add_argument('--lbl_percentage', type=float, default=0.2, help='percentage of which target data has label')
parser.add_argument('--num_per_class', type=int, default=-1, help='number of sample per class when training local discriminator')
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--classifier', type=str, help='cnet model file')
parser.add_argument('--save_path', type=str, default='../train_related/JDA_GAN', help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')

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


# #local only
# 
# class fake_args():
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
# 
#         
# args = fake_args(data_path='../data_unzip/', 
#                  task='3A', 
#                  batch_size=100,
#                  epochs=10,
#                  lr_gan=1e-3,
#                  lr_clf=1e-3,
#                  gap=2,
#                  lbl_percentage=0.2,
#                  num_per_class=-1,
#                  seed=0,
#                  save_path='../train_related/JDA_GAN',
#                  model_save_period=1,
#                  classifier='/Users/stevenliu/time-series-adaption/time-series-domain-adaptation/JDA/FNN_trained_model'
#                  )

# In[32]:


class Generator(nn.Module):
    def __init__(self, feature_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            #nn.Tanh()
        ) 

    def forward(self, x):
        # x: [bs, seq, init_size (small)]
        return self.net(x)


# In[31]:


class Discriminator(nn.Module):
    def __init__(self, feature_dim, d_out):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(

            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ) 
        self.fc = nn.Linear(3200, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: [bs, seq, feature_dim]
        x = self.net(x)
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        out = self.sigmoid(self.fc(x))
        return out


# In[30]:


class FNN(nn.Module):
    def __init__(self, d_in, d_h, d_out, dp):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_out)
        self.dp = nn.Dropout(dp)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp(x)
        x = self.fc2(x)

        return x


# In[8]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('LayerNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# In[9]:


def get_source_dict(file_path, num_class, data_len=None):
    '''
    output:
        {class: [data]},
        data_len
    '''
    data_ = np.load(file_path, allow_pickle=True)
    train_data = data_['tr_data']
    train_lbl = data_['tr_lbl']
    if data_len:
        train_data = data_['tr_data'][:data_len]
        train_lbl = data_['tr_lbl'][:data_len]
    data_dict = get_class_data_dict(train_data, train_lbl, num_class)
    
    return data_dict, train_data.shape[0]

def get_target_dict(file_path, num_class, lbl_percentage, seed=0):
    '''
    split target domain data
    
    output:
        with label:
            {class: [data]}
        without label:
            [data], [lbl]
        data_len
    '''
    data_ = np.load(file_path, allow_pickle=True)
    train_data = data_['te_data']
    train_lbl = data_['te_lbl']
    
    np.random.seed(seed)
    index = np.random.permutation(train_data.shape[0])
    train_data = train_data[index]
    train_lbl = np.argmax(train_lbl[index], -1)

    with_label = {i:[] for i in range(num_class)}
    labeled_index = []
    for i in with_label:
        index = np.argwhere(train_lbl==i).flatten()
        np.random.seed(seed)
        index = np.random.choice(index, int(lbl_percentage*train_lbl.shape[0]/num_class))
        labeled_index.append(index)
        with_label[i] = train_data[index]

    
    return with_label, (np.delete(train_data,labeled_index,axis=0), np.delete(train_lbl,labeled_index,axis=0)), train_data.shape[0]

def get_class_data_dict(data, lbl, num_class):
    '''
    construct a dict {label: data}  
    '''
    lbl_not_one_hot = np.argmax(lbl, -1)
    result = {i:[] for i in range(num_class)}
    for i in result:
        index = np.argwhere(lbl_not_one_hot==i).flatten()
        result[i] = data[index]
        
    return result

def get_batch_source_data_on_class(class_dict, num_per_class):
    '''
    get batch from source data given a required number of sample per class
    '''
    batch_x = []
    batch_y = []
    for key, value in class_dict.items():
        index = random.sample(range(len(value)), num_per_class)
        batch_x.extend(value[index])
        batch_y.extend([key] * num_per_class)
        
    return np.array(batch_x), np.array(batch_y)

def get_batch_target_data_on_class(real_dict, pesudo_dict, unlabel_data, num_per_class, compromise=1, real_weight=1, pesudo_weight=0.1):
    '''
    get batch from target data given a required number of sample per class
    '''
    batch_x = []
    batch_y = []
    batch_real_or_pesudo = []
    for key in real_dict:
        real_num = len(real_dict[key])
        pesudo_num = len(pesudo_dict[key])
        num_in_class = real_num + pesudo_num
        
        if num_in_class < num_per_class:
            # if totoal number sample in this class is less than the required number of sample
            # then fetch the remainding data randomly from the unlabeled set with a compromise
            
            num_fetch_unlabeled = (num_per_class - num_in_class) * compromise
            index = random.sample(range(unlabel_data.shape[0]), num_fetch_unlabeled)
            batch_x.extend(unlabel_data[index])
            batch_y.extend([key] * num_fetch_unlabeled)
            batch_real_or_pesudo.extend([pesudo_weight] * num_fetch_unlabeled)
            
            batch_x.extend(real_dict[key])
            batch_real_or_pesudo.extend([real_weight] * real_num)
            batch_x.extend(pesudo_dict[key])
            batch_real_or_pesudo.extend([pesudo_weight] * pesudo_num)
            batch_y.extend([key] * num_in_class)
            
        else:
            index = random.sample(range(num_in_class), num_per_class)
            index_in_real = []
            index_in_pesudo = []
            for i in index:
                if i >= real_num:
                    index_in_pesudo.append(i-real_num)
                else:
                    index_in_real.append(i)
                    
            batch_x.extend(real_dict[key][index_in_real])
            batch_real_or_pesudo.extend([real_weight] * len(index_in_real))
            batch_x.extend(pesudo_dict[key][index_in_pesudo,:])
            batch_real_or_pesudo.extend([pesudo_weight] * len(index_in_pesudo))
            batch_y.extend([key] * num_per_class)
    
    return np.array(batch_x), np.array(batch_y), np.array(batch_real_or_pesudo)


# # Main

# Model architecture
# 
# encoder: feature extractor
# CNet:    Classifier
# DNet_global:    global Discriminator
# DNet_local:    class-wise Discriminator
# GNet:    Generator (Adaptor)

# In[35]:



args.task = '3Av2' if args.task == '3A' else '3E'
d_out = 50 if args.task == "3Av2" else 65
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if args.num_per_class == -1:
    args.num_per_class = math.ceil(args.batch_size / d_out)
    
model_sub_folder = '/task_%s_gap_%s_lblPer_%i_numPerClass_%i'%(args.task, args.gap, args.lbl_percentage, args.num_per_class)
    

# In[22]:


target_dict, (target_unlabel_x, target_unlabel_y), target_len  = get_target_dict(args.data_path+'processed_file_%s.pkl'%args.task, d_out, args.lbl_percentage)
source_dict, source_len = get_source_dict(args.data_path+'/processed_file_%s.pkl'%args.task, d_out, data_len=target_len)


# In[23]:


seq_len = 10 
feature_dim = 160
classifier_model_folder = 'Final_FNN_' + args.task 
CNet_path = args.classifier + '/' + classifier_model_folder + "/CNet_model.ep100"
encoder_path = args.classifier + '/' + classifier_model_folder + "/Encoder_model.ep100"
    
CNet = FNN(d_in=feature_dim * 2 * seq_len, d_h=500, d_out=d_out, dp=0.5)

encoder = ComplexTransformer(layers=1,
                       time_step=seq_len,
                       input_dim=feature_dim,
                       hidden_size=512,
                       output_dim=512,
                       num_heads=8,
                       out_dropout=0.5)

if torch.cuda.is_available():
    CNet.load_state_dict(torch.load(CNet_path))
    encoder.load_state_dict(torch.load(encoder_path))
else:
    CNet.load_state_dict(torch.load(CNet_path, map_location=torch.device('cpu')))
    encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
encoder.to(device)
CNet.to(device)
def classifier_inference(encoder, CNet, x, x_mean_tr, x_std_tr, batch_size):
    CNet.eval()
    encoder.eval()
    with torch.no_grad():
        #normalize data
        x = (x - x_mean_tr) / x_std_tr
        # take the real and imaginary part out
        real = x[:,:,0].reshape(batch_size, seq_len, feature_dim).float()
        imag = x[:,:,1].reshape(batch_size, seq_len, feature_dim).float()
        if torch.cuda.is_available():
            real.to(device)
            imag.to(device)
        real, imag = encoder(real, imag)
        pred = CNet(torch.cat((real, imag), -1).reshape(x.shape[0], -1))
    return pred


# In[29]:


# initialize GAN
real_label = 0.99 # target domain
fake_label = 0.01 # source domain

feature_dim_joint = 2 * feature_dim
DNet_global = Discriminator(feature_dim=feature_dim_joint, d_out=d_out).to(device)
DNet_local = Discriminator(feature_dim=feature_dim_joint, d_out=d_out).to(device)
GNet = Generator(feature_dim=feature_dim_joint).to(device)
DNet_global.apply(weights_init)
DNet_local.apply(weights_init)
GNet.apply(weights_init)
optimizerD_global = torch.optim.Adam(DNet_global.parameters(), lr=args.lr_gan)
optimizerD_local = torch.optim.Adam(DNet_local.parameters(), lr=args.lr_gan)
optimizerG = torch.optim.Adam(GNet.parameters(), lr=args.lr_gan)

# TODO: add global & local loss
criterion_gan_global = nn.BCELoss()
criterion_gan_local = nn.BCELoss(reduction='none')
schedulerD_global = torch.optim.lr_scheduler.StepLR(optimizerD_global, step_size=30, gamma=0.1)
schedulerD_local = torch.optim.lr_scheduler.StepLR(optimizerD_local, step_size=30, gamma=0.1)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.1)

feature_dim_joint = 2 * feature_dim
joint_set = TimeSeriesDatasetConcat(root_dir=args.data_path, file_name='processed_file_%s.pkl'%args.task, seed=args.seed)
joint_loader = DataLoader(joint_set, batch_size=args.batch_size, shuffle=True)
total_error_D = total_error_G = 0

source_mean = joint_set.tr_data_mean
target_mean = joint_set.te_data_mean
source_std = joint_set.tr_data_std
target_std = joint_set.te_data_std

D_global_losses = []
D_local_losses = []
G_losses = []
classifier_acc = []
print(device)


# Create target Directory if don't exist
if not os.path.exists(args.save_path+model_sub_folder):
    os.mkdir(args.save_path+model_sub_folder)
    print("Directory " , args.save_path+model_sub_folder ,  " Created ")
else:    
    print("Directory " , args.save_path+model_sub_folder ,  " already exists")

for epoch in range(args.epochs):
    correct_target = 0.0
    target_pesudo_y = []
    for batch in range(math.ceil(target_unlabel_x.shape[0]/args.batch_size)):
        batch_size = target_unlabel_x[batch*args.batch_size:(batch+1)*args.batch_size].shape[0]
        target_unlabel_x_batch = torch.tensor(target_unlabel_x[batch*args.batch_size:(batch+1)*args.batch_size], device=device).float()
        target_unlabel_y_batch = torch.tensor(target_unlabel_y[batch*args.batch_size:(batch+1)*args.batch_size], device=device)
        pred = classifier_inference(encoder, CNet, target_unlabel_x_batch, target_mean, target_std, batch_size)
        correct_target += (pred.argmax(-1) == target_unlabel_y_batch.argmax(-1)).sum().item()
        target_pesudo_y.extend(pred.argmax(-1).cpu().numpy())
    
    target_pesudo_y = np.array(target_pesudo_y)
    pesudo_dict = get_class_data_dict(target_unlabel_x, target_pesudo_y, d_out)
    print('Epoch: %i, Classifier Acc on Target Domain: %f'%(epoch-1, correct_target/target_unlabel_x.shape[0]))
    classifier_acc.append(correct_target/target_unlabel_x.shape[0])
    
    print('Start Training On global Discriminator')
    total_error_D_global = 0
    total_error_D_local = 0
    total_error_G = 0
    total_error_D = total_error_G_global = total_error_G_local = 0
    for batch_id, (source_x, target_x) in tqdm(enumerate(joint_loader)):
        batch_size = target_x.shape[0]
        target_x = target_x.reshape(batch_size, seq_len, feature_dim_joint)
        source_x = source_x.reshape(batch_size, seq_len, feature_dim_joint)

        # Data Normalization
        target_x = (target_x - target_mean) / target_std
        source_x = (source_x - source_mean) / source_std

        """Update D Net"""
        # train with source domain
        DNet_global.zero_grad()
        source_data = source_x.to(device).float()
        label = torch.full((batch_size,), real_label, device=device)
        output = DNet_global(source_data).view(-1)
        #print(output.mean().item())
        errD_global_source = criterion_gan_global(output, label)
        errD_global_source.backward()

        # train with target domain
        target_data = target_x.to(device).float()
        fake = GNet(target_data)
        #print(fake)
        label.fill_(fake_label)
        output = DNet_global(fake.detach()).view(-1)
        #print(output.mean().item())
        errD_global_target = criterion_gan_global(output, label)
        errD_global_target.backward()
        total_error_D_global += (errD_global_source + errD_global_target).item()
        
        if batch_id % args.gap == 0:
            optimizerD_global.step()

        """Update G Network"""
        GNet.zero_grad()
        label.fill_(real_label) # fake labels are real for generator cost
        output = DNet_global(fake).view(-1)
        #print(output.mean().item())
        #print()
        errG = criterion_gan_global(output, label)
        errG.backward()
        optimizerG.step()
        total_error_G += errG.item()

    print('Start Training On local Discriminator')
    for batch_id in tqdm(range(math.ceil(target_len/args.batch_size))):
        target_x, target_y, target_weight = get_batch_target_data_on_class(target_dict, pesudo_dict, target_unlabel_x, args.num_per_class)
        source_x, source_y = get_batch_source_data_on_class(source_dict, args.num_per_class)
        
        target_x = torch.tensor(target_x, device=device)
        source_x = torch.tensor(source_x, device=device)
        target_weight = torch.tensor(target_weight, device=device)
        batch_size = target_x.shape[0]
        target_x = target_x.reshape(batch_size, seq_len, feature_dim_joint)
        batch_size = source_x.shape[0]
        source_x = source_x.reshape(batch_size, seq_len, feature_dim_joint)
        
        # Data Normalization
        target_x = (target_x - target_mean) / target_std
        source_x = (source_x - source_mean) / source_std
        
        """Update D Net"""
        # train with source domain
        DNet_local.zero_grad()
        source_data = source_x.to(device).float()
        label = torch.full((batch_size,), real_label, device=device)
        output = DNet_local(source_data).view(-1)
        #print(output.mean().item())
        errD_local_source = criterion_gan_local(output, label).mean()
        errD_local_source.backward()

        # train with target domain
        target_data = target_x.to(device).float()
        fake = GNet(target_data)
        #print(fake)
        label.fill_(fake_label)
        output = DNet_local(fake.detach()).view(-1)
        errD_local_target = criterion_gan_local(output, label)
        errD_local_target = (errD_local_target * target_weight).mean()
        errD_local_target.backward()
        total_error_D_local += (errD_local_source + errD_local_target).item()
        
        if batch_id % args.gap == 0:
            optimizerD_local.step()

        """Update G Network"""
        GNet.zero_grad()
        label.fill_(real_label) # fake labels are real for generator cost
        output = DNet_local(fake).view(-1)

        errG = criterion_gan_local(output, label)
        errG = (errG * target_weight).mean()
        errG.backward()
        optimizerG.step()
        total_error_G += errG.item()
        
        
    total_error_G = total_error_G/2
    schedulerD_global.step()
    schedulerD_local.step()
    schedulerG.step()
    print('Epoch: %i, total loss: %f, G loss: %f, D_global loss: %f, D_local loss: %f'%(
        epoch, total_error_D_local+total_error_D_global+total_error_G, total_error_G, total_error_D_global, total_error_D_local))
    D_global_losses.append(total_error_D_global)
    D_local_losses.append(total_error_D_local)
    G_losses.append(total_error_G)
    
    if epoch % args.model_save_period == 0:
        torch.save(DNet_global.state_dict(), args.save_path+model_sub_folder+'/DNet_global_%i'%epoch)
        torch.save(DNet_local.state_dict(), args.save_path+model_sub_folder+'/DNet_local_%i'%epoch)
        torch.save(GNet.state_dict(), args.save_path+model_sub_folder+'/GNet_%i'%epoch)
        
    np.save(args.save_path+model_sub_folder+'/D_global_losses.npy', D_global_losses)
    np.save(args.save_path+model_sub_folder+'/D_local_losses.npy', D_local_losses)
    np.save(args.save_path+model_sub_folder+'/G_loss.npy', G_losses)
    np.save(args.save_path+model_sub_folder+'/classifier_acc.npy', classifier_acc[1:])


correct_target = 0.0
target_pesudo_y = []
for batch in range(math.ceil(target_unlabel_x.shape[0]/args.batch_size)):
    batch_size = target_unlabel_x[batch*args.batch_size:(batch+1)*args.batch_size].shape[0]
    target_unlabel_x_batch = torch.tensor(target_unlabel_x[batch*args.batch_size:(batch+1)*args.batch_size], device=device).float()
    target_unlabel_y_batch = torch.tensor(target_unlabel_y[batch*args.batch_size:(batch+1)*args.batch_size], device=device)
    pred = classifier_inference(encoder, CNet, target_unlabel_x_batch, target_mean, target_std, batch_size)
    correct_target += (pred.argmax(-1) == target_unlabel_y_batch.argmax(-1)).sum().item()
    target_pesudo_y.extend(pred.argmax(-1).cpu().numpy())
    
target_pesudo_y = np.array(target_pesudo_y)
pesudo_dict = get_class_data_dict(target_unlabel_x, target_pesudo_y, d_out)
print('Epoch: %i, Classifier Acc on Target Domain: %f'%(epoch, correct_target/target_unlabel_x.shape[0]))
classifier_acc.append(correct_target/target_unlabel_x.shape[0])
np.save(args.save_path+model_sub_folder+'/classifier_acc.npy', classifier_acc[1:])
