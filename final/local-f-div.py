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
from models.GAN import Generator
from utils import *
import argparse
import logging
from data_utils import SingleDataset, read_data
from models.FDIV import *

###############################################################################
#                                 parameters                                  #
###############################################################################
parser = argparse.ArgumentParser(description='f-div')
parser.add_argument("--data_path", type=str, required=True, help="dataset path")
parser.add_argument("--data_path", type=str, required=True, help="dataset path")
parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of which target data has label')
parser.add_argument('--what_model', type=str, required=True, help='run f-div on what model : "naive" or "GAN"')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of which source data has label')
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--save_path', type=str, required=True, help='where to store result')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')
parser.add_argument('--model_path', type=str, required=True, help='where the model is stored')
parser.add_argument('--intervals', type=int, default=2, help='freq of compute f-div')
parser.add_argument('--model_name', type=str, required=True, help='name of the model')
parser.add_argument('--gfunction_epoch', type=int, default=5000, help='epoch of which gfunction is trained for')
parser.add_argument('--classifier', type=bool, default=False, help="if optmizer classifier")
parser.add_argument('--sclass', type=float, default=0.7, help='target classifier loss weight')
parser.add_argument('--classifier_epoch', type=int, default=5000, help='max iteration to train classifier')
parser.add_argument('--start_epoch', type=int, default=-1, help='start epoch')
parser.add_argument('--end_epoch', type=int, default=-1, help='end epoch')

args = parser.parse_args()

assert args.task in ['3Av2', '3E']
assert args.what_model in ['naive', "GAN"]
num_class = 50 if args.task == "3Av2" else 65
device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')
print(device)
source_acc_label_ = np.load(os.path.join(args.model_path, 'source_acc_label_.npy'))

start_epoch = args.start_epoch
end_epoch = args.end_epoch

if args.start_epoch == -1:
    start_epoch = 3
if args.end_epoch == -1:
    end_epoch = source_acc_label_.shape[0]

assert start_epoch < end_epoch
print('start at epoch %i, end at epoch %i'%(start_epoch, end_epoch))

model_sub_folder = 'result/conditional-f-div-%s/'%args.what_model+args.model_name
if args.classifier: model_sub_folder += '_classifier'
if args.start_epoch != -1 or args.end_epoch != -1:
    model_sub_folder += '_s{}_e{}'.format(start_epoch, end_epoch)

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
labeled_target_x, labeled_target_y, unlabeled_target_x, unlabeled_target_y = read_data(args.task, 'target', args.target_lbl_percentage, args.data_path)
labeled_target_dataset = SingleDataset(labeled_target_x, labeled_target_y)
unlabeled_target_dataset = SingleDataset(unlabeled_target_x, unlabeled_target_y)
labeled_target_dataloader = DataLoader(labeled_target_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
unlabeled_target_dataloader = DataLoader(unlabeled_target_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

labeled_source_x, labeled_source_y, unlabeled_source_x, unlabeled_source_y = read_data(args.task, 'source', args.source_lbl_percentage, args.data_path)
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

gfunction_JS_div_labeled = Gfunction(num_class).to(device)
gfunction_JS_div_unlabeled = Gfunction(num_class).to(device)
gfunction_JS_div_unlabeled.apply(weights_init)
gfunction_JS_div_labeled.apply(weights_init)

if args.classifier:
    CNet = FNNLinear(d_h2=64*2, d_out=num_class).to(device)
    criterion_classifier = nn.CrossEntropyLoss().to(device)
    CNet.apply(weights_init)

if args.what_model == 'GAN':
    GNet = Generator(dim=64*2).to(device)
    GNet.apply(weights_init)
else:
    GNet = None

encoder.apply(weights_init)
encoder_MLP.apply(weights_init)


def get_embedding(encoder, encoder_MLP, dataloader, GNet=None):
    encoder.eval()
    encoder_MLP.eval()
    x_embedding_ = torch.empty(0).to(device)
    y_ = torch.empty(0).long().to(device)
    with torch.no_grad():
        for batch_id, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = x.to(device).float()
            y = y.to(device).long()
            x_embedding = encoder_inference(encoder, encoder_MLP, x)
            if GNet != None:
                x_embedding = GNet(x_embedding)
            x_embedding = x_embedding.detach()
            x_embedding_ = torch.cat([x_embedding_, x_embedding])
            y_ = torch.cat([y_, y])

    return x_embedding_, y_

def get_KL(source_x_embedding, target_x_embedding, mask_source, mask_target, model, optimizer, args, device):
    model.train()
    for i in tqdm(range(args.gfunction_epoch)):
        optimizer.zero_grad()
        source_x_g = model(source_x_embedding)
        target_x_g = model(target_x_embedding)
        loss_JS_labeled = - JSDiv(source_x_g, target_x_g, mask_source, mask_target, device) # maximize
        loss_JS_labeled.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        source_x_g = model(source_x_g)
        KL = source_x_g.mean().item()

    return KL

###############################################################################
#                                    Training                                 #
###############################################################################
logger.info('Started loading')
source_acc_label_ = np.load(os.path.join(args.model_path, 'source_acc_label_.npy'))
source_acc_unlabel_ = np.load(os.path.join(args.model_path, 'source_acc_unlabel_.npy'))
target_acc_label_ = np.load(os.path.join(args.model_path, 'target_acc_label_.npy'))
target_acc_unlabel_ = np.load(os.path.join(args.model_path, 'target_acc_unlabel_.npy'))

acc_source_unlabeled_classifier_ = []
acc_target_unlabeled_classifier_ = []

source_acc_label = []
source_acc_unlabel = []
target_acc_label = []
target_acc_unlabel = []

epochs = []
labeled_KL_ = []
unlabeled_KL_ = []

for epoch in range(start_epoch, end_epoch, args.intervals*args.model_save_period):
    # initialize
    gfunction_JS_div_labeled.apply(weights_init)
    optimizer_gfunction_JS_div_labeled = torch.optim.Adam(gfunction_JS_div_labeled.parameters(), lr=args.lr)
    gfunction_JS_div_unlabeled.apply(weights_init)
    optimizer_gfunction_JS_div_unlabeled = torch.optim.Adam(gfunction_JS_div_unlabeled.parameters(), lr=args.lr)

    if args.classifier:
        CNet.load_state_dict(torch.load(os.path.join(args.model_path, 'CNet_%i.t7'%epoch)))
        optimizer_CNet = torch.optim.Adam(CNet.parameters(), lr=args.lr)

    # load weight
    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder_%i.t7'%epoch)))
    encoder_MLP.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder_MLP%i.t7'%epoch)))
    if args.what_model == "GAN": GNet.load_state_dict(torch.load(os.path.join(args.model_path, 'GNet_%i.t7'%epoch)))

    # get source/target embedding
    source_x_unlabeled_embedding, source_y_unlabeled = get_embedding(encoder, encoder_MLP, unlabeled_source_dataloader, GNet=GNet)
    source_x_labeled_embedding, source_y_labeled = get_embedding(encoder, encoder_MLP, labeled_source_dataloader, GNet=GNet)    
    target_x_unlabeled_embedding, target_y_unlabeled = get_embedding(encoder, encoder_MLP, unlabeled_target_dataloader, GNet=GNet)
    target_x_labeled_embedding, target_y_labeled = get_embedding(encoder, encoder_MLP, labeled_target_dataloader, GNet=GNet)

    # one-hot encoding of ys
    mask_source_labeled = torch.zeros((source_y_labeled.size(0), num_class), device=device).scatter_(1, source_y_labeled.unsqueeze(1), 1)
    mask_target_labeled = torch.zeros((target_y_labeled.size(0), num_class), device=device).scatter_(1, target_y_labeled.unsqueeze(1), 1)
    mask_source_unlabeled = torch.zeros((source_y_unlabeled.size(0), num_class), device=device).scatter_(1, source_y_unlabeled.unsqueeze(1), 1)
    mask_target_unlabeled = torch.zeros((target_y_unlabeled.size(0), num_class), device=device).scatter_(1, target_y_unlabeled.unsqueeze(1), 1)

    # concatenate x_embedding and one-hot encoding of ys
    source_x_labeled_embedding_cat = torch.cat([source_x_labeled_embedding, mask_source_labeled], dim=1)
    source_x_unlabeled_embedding_cat = torch.cat([source_x_unlabeled_embedding, mask_source_unlabeled], dim=1)
    target_x_labeled_embedding_cat = torch.cat([target_x_labeled_embedding, mask_target_labeled], dim=1)
    target_x_unlabeled_embedding_cat = torch.cat([target_x_unlabeled_embedding, mask_target_unlabeled], dim=1)

    # get KL
    # labeled data
    KL_labeled_eval = get_KL(source_x_labeled_embedding_cat, target_x_labeled_embedding_cat, mask_source_labeled, mask_target_labeled, gfunction_JS_div_labeled, optimizer_gfunction_JS_div_labeled, args, device)
    labeled_KL_.append(KL_labeled_eval)

    # unlabeled data
    KL_unlabeled_eval = get_KL(source_x_unlabeled_embedding_cat, target_x_unlabeled_embedding_cat, mask_source_unlabeled, mask_target_unlabeled, gfunction_JS_div_unlabeled, optimizer_gfunction_JS_div_unlabeled, args, device)
    unlabeled_KL_.append(KL_unlabeled_eval)

    # train classifier
    acc_source_labeled_classifier = 0
    acc_target_labeled_classifier = 0
    if args.classifier:
        for i in tqdm(range(args.classifier_epoch)):
            CNet.train()
            # source domain
            optimizer_CNet.zero_grad()
            pred = CNet(source_x_labeled_embedding)
            acc_source_labeled_classifier = (pred.argmax(-1) == source_y_labeled).sum().item() / pred.size(0)
            loss_source_classifier_labeled = criterion_classifier(pred, source_y_labeled) * args.sclass
            loss_source_classifier_labeled.backward()
            optimizer_CNet.step()

            # target domain
            optimizer_CNet.zero_grad()
            pred = CNet(target_x_labeled_embedding)
            acc_target_labeled_classifier = (pred.argmax(-1) == target_y_labeled).sum().item() / pred.size(0)
            loss_target_classifier_labeled = criterion_classifier(pred, target_y_labeled)
            loss_target_classifier_labeled.backward()
            optimizer_CNet.step()

        # eval
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
    if args.KL: log_string += "labeled KL: %f, unlabeled KL: %f; "%(KL_labeled_eval, KL_unlabeled_eval)
    if args.classifier: log_string += "src unlbl acc: %f, tgt unlbl acc: %f; "%(acc_source_unlabeled_classifier, acc_target_unlabeled_classifier)
    logger.info(log_string)
    logger.info("-----------------------------------------")

    np.save(os.path.join(save_folder, '/epochs.npy'), epochs)
    np.save(os.path.join(save_folder, '/source_acc_label.npy'), source_acc_label)
    np.save(os.path.join(save_folder, '/source_acc_unlabel.npy'), source_acc_unlabel)
    np.save(os.path.join(save_folder, '/target_acc_label.npy'), target_acc_label)
    np.save(os.path.join(save_folder, '/target_acc_unlabel.npy'), target_acc_unlabel)
    np.save(os.path.join(save_folder, '/labeled_KL_.npy'), labeled_KL_)
    np.save(os.path.join(save_folder, '/unlabeled_KL_.npy'), unlabeled_KL_)

    if args.classifier:
        np.save(os.path.join(save_folder, '/acc_source_unlabeled_classifier_.npy'), acc_source_unlabeled_classifier_)
        np.save(os.path.join(save_folder, '/acc_target_unlabeled_classifier_.npy'), acc_target_unlabeled_classifier_)


