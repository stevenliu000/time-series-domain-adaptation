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

def train_classification(ifsource, CNet, encoder, encoder_MLP, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, criterion_classifier, labeled_dataloader, args):
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
        if ifsource: loss *= args.sclass
        loss.backward()
        optimizerCNet.step()
        optimizerEncoderMLP.step()
        optimizerEncoder.step()

    acc_label = acc_label / num_datas
    return acc_label

def eval_classification(CNet, encoder, encoder_MLP, unlabeled_dataloader):
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
parser = argparse.ArgumentParser(description='Naive Adaptation: shared architecture trained on two domains')
parser.add_argument("--data_path", type=str, required=True, help="dataset path")
parser.add_argument("--task", type=str, required=True, help='3Av2 or 3E')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
parser.add_argument('--lr_FNN', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--sclass', type=float, default=0.7, help='source-domain classification loss weight')
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of which target data has label')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of which source data has label')
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--save_path', type=str, help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')

args = parser.parse_args()

assert args.task in ['3Av2', '3E']
num_class = 50 if args.task == "3Av2" else 65

model_sub_folder = 'result/naive_adaption/lbl_percent_%f/task_%s_slp_%f_tlp_%f_sclass_%f'%(args.target_lbl_percentage, args.task, args.source_lbl_percentage, args.target_lbl_percentage, args.sclass)
save_folder = os.path.join(args.save_path, model_sub_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)

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
labeled_target_x, labeled_target_y, unlabeled_target_x, unlabeled_target_y = read_data(args, 'target')
labeled_target_dataset = SingleDataset(labeled_target_x, labeled_target_y)
unlabled_target_dataset = SingleDataset(unlabeled_target_x, unlabeled_target_y)
labeled_target_dataloader = DataLoader(labeled_target_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
unlabeled_target_dataloader = DataLoader(unlabled_target_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

labeled_source_x, labeled_source_y, unlabeled_source_x, unlabeled_source_y = read_data(args, 'source')
labeled_source_dataset = SingleDataset(labeled_source_x, labeled_source_y)
unlabled_source_dataset = SingleDataset(unlabeled_source_x, unlabeled_source_y)
labeled_source_dataloader = DataLoader(labeled_source_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
unlabeled_source_dataloader = DataLoader(unlabled_source_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

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
encoder_MLP = FNNSeparated(d_in=64 * 2 * 1, d_h1=64*4, d_h2=64*2, dp=0.2).to(device)
CNet = FNNLinear(d_h2=64*2, d_out=num_class).to(device)
criterion_classifier = nn.CrossEntropyLoss().to(device)

encoder.apply(weights_init)
encoder_MLP.apply(weights_init)
CNet.apply(weights_init)

optimizerCNet = torch.optim.Adam(CNet.parameters(), lr=args.lr_FNN)
optimizerEncoderMLP = torch.optim.Adam(encoder_MLP.parameters(), lr=args.lr_encoder)
optimizerEncoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)

###############################################################################
#                                    Training                                 #
###############################################################################
source_acc_label_ = []
source_acc_unlabel_ = []
target_acc_label_ = []
target_acc_unlabel_ = []

logger.info('Started Training')
for epoch in range(args.epochs):
    # update classifier
    # on source domain
    source_acc_label = train_classification(True, CNet, encoder, encoder_MLP, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, criterion_classifier, labeled_source_dataloader, args)
    source_acc_label_.append(source_acc_label)

    # on target domain
    target_acc_label = train_classification(False, CNet, encoder, encoder_MLP, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, criterion_classifier, labeled_source_dataloader, args)
    target_acc_label_.append(target_acc_label)

    # eval
    # source_domain
    source_acc_unlabel = eval_classification(CNet, encoder, encoder_MLP, unlabeled_source_dataloader)
    source_acc_unlabel_.append(source_acc_unlabel)

    # target_domain
    target_acc_unlabel = eval_classification(CNet, encoder, encoder_MLP, unlabeled_target_dataloader)
    target_acc_unlabel_.append(target_acc_unlabel)

    if epoch % args.model_save_period == 0:
        torch.save(encoder.state_dict(), args.save_path+model_sub_folder+ 'encoder_%i.t7'%(epoch+1))
        torch.save(encoder_MLP.state_dict(), args.save_path+model_sub_folder+ 'encoder_MLP_%i.t7'%(epoch+1))
        torch.save(CNet.state_dict(), args.save_path+model_sub_folder+ 'CNet_%i.t7'%(epoch+1))

    logger.info('Epochs %i: src labeled acc: %f; src unlabeled acc: %f; tgt labeled acc: %f; tgt unlabeled acc: %f'%(epoch+1, source_acc_label, source_acc_unlabel, target_acc_label, target_acc_unlabel))
    np.save(args.save_path+model_sub_folder+'target_acc_label_.npy', target_acc_label_)
    np.save(args.save_path+model_sub_folder+'target_acc_unlabel_.npy', target_acc_unlabel_)
    np.save(args.save_path+model_sub_folder+'source_acc_label_.npy', source_acc_label_)
    np.save(args.save_path+model_sub_folder+'source_acc_unlabel_.npy', source_acc_unlabel_)


