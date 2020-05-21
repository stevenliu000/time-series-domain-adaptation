import os
import sys
import numpy as np
import math
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from models.complex_transformer import ComplexTransformer
from models.FNNLinear import FNNLinear
from models.FNNSeparated import FNNSeparated
from utils import *
import argparse
import logging
from data_utils import SingleDataset, read_data

def train_classification(CNet, encoder, encoder_MLP, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, criterion_classifier, labeled_dataloader, args):
    CNet.train()
    encoder.train()
    encoder_MLP.train()
    acc_label = 0.0
    num_datas = 0.0
    for batch_id, (x, y) in tqdm(enumerate(labeled_dataloader), total=len(labeled_dataloader)):
        optimizerCNet.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerEncoderMLP.zero_grad()
        x = x.to(device).float()
        y = y.to(device)
        num_datas += x.size(0)
        x_embedding = encoder_inference(encoder, encoder_MLP, x)
        pred = CNet(x_embedding)
        acc_label += (pred.argmax(-1) == y).sum().item()
        loss = criterion_classifier(pred, y)
        loss.backward()
        optimizerCNet.step()
        optimizerEncoderMLP.step()
        optimizerEncoder.step()

    acc_label = acc_label / num_datas

    return acc_label

def eval_classification(CNet, encoder, encoder_MLP, unlabeled_dataloader, args):

    acc_unlabel = 0.0
    num_datas = 0.0
    CNet.eval()
    encoder.eval()
    encoder_MLP.eval()
    with torch.no_grad():
        for batch_id, (x, y) in tqdm(enumerate(unlabeled_dataloader), total=len(unlabeled_dataloader)):
            x = x.to(device).float()
            y = y.to(device)
            num_datas += x.shape[0]
            x_embedding = encoder_inference(encoder, encoder_MLP, x)
            pred = CNet(x_embedding)
            acc_unlabel += (pred.argmax(-1) == y).sum().item()
        
    acc_unlabel = acc_unlabel/num_datas

    return acc_unlabel

###############################################################################
#                                 parameters                                  #
###############################################################################
parser = argparse.ArgumentParser(description='Baseline: only train on one domain')
parser.add_argument("--data_path", type=str, required=True, help="dataset path")
parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
parser.add_argument("--domain", type=str, required=True, help='source or domain')
parser.add_argument("--task", type=str, required=True, help='3Av2 or 3E')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr_FNN', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate for encoder')
parser.add_argument('--lbl_percentage', type=float, default=0.7, help='percentage of which source data has label')
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--save_path', type=str, required=True, help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period of which the model is saved')

args = parser.parse_args()

assert args.task in ['3Av2', '3E']
assert args.domain in ['source', 'target']
num_class = 50 if args.task == "3Av2" else 65
device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)
    
model_sub_folder = 'result/baseline/task_%s_lbl_percentage_%f'%(args.task, args.lbl_percentage)
save_folder = os.path.join(args.save_path, model_sub_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


###############################################################################
#                                     Seed                                    #
###############################################################################
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True

###############################################################################
#                                     Logger                                  #
###############################################################################

logger = get_logger(save_folder, args)

###############################################################################
#                                 Data Loading                                #
###############################################################################
labeled_x, labeled_y, unlabeled_x, unlabeled_y = read_data(args, args.domain)
labeled_dataset = SingleDataset(labeled_x, labeled_y)
unlabled_dataset = SingleDataset(unlabeled_x, unlabeled_y)
labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
unlabeled_dataloader = DataLoader(unlabled_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)


###############################################################################
#                               Model Creation                                #
###############################################################################
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
encoder_MLP = FNNSeparated(d_in=64*2, d_h1=64*4, d_h2=64*2, dp=0.2).to(device)
CNet = FNNLinear(d_h2=64*2, d_out=num_class).to(device)
criterion_classifier = nn.CrossEntropyLoss().to(device)

encoder.apply(weights_init)
encoder_MLP.apply(weights_init)
CNet.apply(weights_init)

optimizerCNet = torch.optim.Adam(CNet.parameters(), lr=args.lr_FNN)
optimizerEncoderMLP = torch.optim.Adam(encoder_MLP.parameters(), lr=args.lr_encoder)
optimizerEncoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)

###############################################################################
#                                    Train                                    #
###############################################################################
acc_label_ = []
acc_unlabel_ = []

logger.info('Started Training')
for epoch in range(args.epochs):
    # update classifier
    acc_label = train_classification(CNet, encoder, encoder_MLP, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, criterion_classifier, labeled_dataloader, args)
    acc_label_.append(acc_label)
    
    # eval    
    acc_unlabel = eval_classification(CNet, encoder, encoder_MLP, unlabeled_dataloader, args)
    acc_unlabel_.append(acc_unlabel)
    
    if epoch % args.model_save_period == 0:
        torch.save(encoder.state_dict(), os.path.join(save_folder, 'encoder_%i.t7'%(epoch+1)))
        torch.save(encoder_MLP.state_dict(), os.path.join(save_folder, 'encoder_MLP_%i.t7'%(epoch+1)))
        torch.save(CNet.state_dict(), os.path.join(save_folder, 'CNet_%i.t7'%(epoch+1)))

    logger.info('Epochs %i: source labeled acc: %f; source unlabeled acc: %f'%(epoch+1, acc_label, acc_unlabel))
    np.save(os.path.join(save_folder, 'acc_label_.npy'), acc_label_)
    np.save(os.path.join(save_folder, 'acc_unlabel_.npy'), acc_unlabel_)
    

