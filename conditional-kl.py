"""
Estimating conditional KL via conjugate approximation.

Example:

- For GAN:
$ python conditional-kl.py\
$        --data_path ../data_unzip
$        --save_path ../train_related
$        --model_name global_gan_conditional_KL_lbl0.7
$        --model_path [root folder you save your model]

- For Naive adaptation:
$ python conditional-kl.py\
$        --data_path ../data_unzip
$        --save_path ../train_related
$        --model_name global_gan_conditional_KL_lbl0.7
$        --model_path [root folder you save your model]
$        --naive_adaptation True
"""




import sys, os, inspect
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
from models.complex_transformer import ComplexTransformer
from models.FNNLinear import FNNLinear
from models.FNNSeparated import FNNSeparated
from models.GAN import Generator
from models.FDIV import *
from utils import *
from data_utils import SingleDataset, read_data
import argparse
import logging
from torch.autograd import Variable

###############################################################################
#                                 Parameters                                  #
###############################################################################
parser = argparse.ArgumentParser(description='Conditional KL')
parser.add_argument("--data_path", type=str, default="../data_unzip/", help="dataset path")
parser.add_argument("--task", type=str, default='3E', help='3A or 3E')
parser.add_argument('--gpu_num', type=int, default=0, help='gpu number')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for gFunction and CNet')
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of which target data has label')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of which source data has label')
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--save_path', type=str, default="../train_related/", help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')
parser.add_argument('--model_path', type=str,required=True, help='where the data is stored')
parser.add_argument('--intervals', type=int, default=4, help='freq of compute f-div')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--gfunction_epoch', type=int, default=2000, help='epoch of which gfunction is trained for')
parser.add_argument('--classifier', type=bool, default=True, help="if optmizer classifier")
parser.add_argument('--sclass', type=float, default=0.7, help='target classifier loss weight')
parser.add_argument('--classifier_epoch', type=int, default=5000, help='max iteration to train classifier')
parser.add_argument('--start_epoch', type=int, default=-1, help='start epoch')
parser.add_argument('--end_epoch', type=int, default=-1, help='end epoch')
parser.add_argument('--naive_adaptation', type=bool, default=False, help='Whether to calculate the naive adaptation KL; set False if want to estimate KL for GAN')

args = parser.parse_args()

###############################################################################
#                              Function Definition                            #
###############################################################################

def eval_KL(gfunction_JS_div, source_x_embedding_cat, KL_report_js):
    with torch.no_grad():
        gfunction_JS_div.eval()
        source_x_g = gfunction_JS_div(source_x_embedding_cat)
        KL_eval = source_x_g.mean()
        KL_report_js.append(KL_eval.item())
        return KL_eval

def train_gfunction(optimizer_gfunction_JS_div, gfunction_JS_div, source_x_embedding_cat, target_x_embedding_cat, mask_source, mask_target, device, JSs, args):
    for i in tqdm(range(args.gfunction_epoch)):
        gfunction_JS_div.train()
        optimizer_gfunction_JS_div.zero_grad()
        source_x_g = gfunction_JS_div(source_x_embedding_cat)
        target_x_g = gfunction_JS_div(target_x_embedding_cat)
        loss_JS = - JSDiv(source_x_g, target_x_g, mask_source, mask_target, device) # maximize
        loss_JS.backward()
        optimizer_gfunction_JS_div.step()
        loss_JS = - loss_JS.item()
        JSs.append(loss_JS)
    return loss_JS

def get_target_embedding(labeled_target_dataloader, encoder, encoder_MLP, GNet, target_x_labeled_embedding, target_y_labeled, args):
    with torch.no_grad():
        for batch_id, (target_x, target_y) in tqdm(enumerate(labeled_target_dataloader), total=len(labeled_target_dataloader)):
            target_x = target_x.to(device).float()
            target_y = target_y.to(device).long()
            target_x_embedding = encoder_inference(encoder, encoder_MLP, target_x)
            if not args.naive_adaptation:
                fake_x_embedding = GNet(target_x_embedding).detach()
                target_x_labeled_embedding = torch.cat([target_x_labeled_embedding, fake_x_embedding])
            else:
                target_x_labeled_embedding = torch.cat([target_x_labeled_embedding, target_x_embedding])
            target_y_labeled = torch.cat([target_y_labeled, target_y])
        return target_x_labeled_embedding, target_y_labeled

def get_source_embedding(labeled_source_dataloader, encoder, encoder_MLP, source_x_labeled_embedding, source_y_labeled, args):
    with torch.no_grad():
        for batch_id, (source_x, source_y) in tqdm(enumerate(labeled_source_dataloader), total=len(labeled_source_dataloader)):
            source_x = source_x.to(device).float()
            source_y = source_y.to(device).long()
            source_x_embedding = encoder_inference(encoder, encoder_MLP, source_x).detach()
            source_x_labeled_embedding = torch.cat([source_x_labeled_embedding, source_x_embedding])
            source_y_labeled = torch.cat([source_y_labeled, source_y])
        return source_x_labeled_embedding, source_y_labeled


###############################################################################
#                             Parameter processing                            #
###############################################################################
device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')

# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True

# parameter processing
args.task = '3Av2' if args.task == '3A' else '3E'
num_class = 50 if args.task == "3Av2" else 65
device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')


source_acc_label_ = np.load(os.path.join(args.model_path, 'source_acc_label_.npy'))

start_epoch = args.start_epoch
end_epoch = args.end_epoch

if args.start_epoch == -1:
    start_epoch = 3
if args.end_epoch == -1:
    end_epoch = source_acc_label_.shape[0]

assert start_epoch < end_epoch

###############################################################################
#                              Save Path and Logger                           #
###############################################################################
# save folder
model_sub_folder = 'results/conditional_KL/'+args.model_name
model_sub_folder += '_JS'
if args.classifier: model_sub_folder += '_classifier'
if args.start_epoch != -1 or args.end_epoch != -1:
    model_sub_folder += '_s{}_e{}'.format(start_epoch, end_epoch)

model_sub_folder += '/'
save_folder = os.path.abspath(os.path.join(args.save_path, model_sub_folder))

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# logging
logger = get_logger(save_folder, device, args)

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
#                                 Define models                               #
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

if not args.naive_adaptation:
    GNet = Generator(dim=64*2).to(device)
else:
    GNet = None


gfunction_JS_div_labeled = Gfunction(num_class).to(device)
gfunction_JS_div_unlabeled = Gfunction(num_class).to(device)

if args.classifier:
    CNet = FNNLinear(d_h2=64*2, d_out=num_class).to(device)
    criterion_classifier = nn.CrossEntropyLoss().to(device)

###############################################################################
#                      Initialize Nueral network weights                      #
###############################################################################
encoder.apply(weights_init)
encoder_MLP.apply(weights_init)
if not args.naive_adaptation:
    GNet.apply(weights_init)


###############################################################################
#                   Training gFunction for each epochs                        #
###############################################################################
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
KL_report_js_unlabeled = []
KL_report_js_labeled = []


assert start_epoch < end_epoch
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
    encoder_MLP.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder_MLP_%i.t7'%epoch)))
    if not args.naive_adaptation:
        GNet.load_state_dict(torch.load(os.path.join(args.model_path, 'GNet_%i.t7'%epoch)))

    # inferencing
    if not args.naive_adaptation:
        GNet.eval()
    encoder.eval()
    encoder_MLP.eval()

    # get source/target embedding
    source_x_labeled_embedding = torch.empty(0).to(device)
    source_y_labeled = torch.empty(0).long().to(device)
    source_x_unlabeled_embedding = torch.empty(0).to(device)
    source_y_unlabeled = torch.empty(0).long().to(device)
    target_x_labeled_embedding = torch.empty(0).to(device)
    target_y_labeled = torch.empty(0).long().to(device)
    target_x_unlabeled_embedding = torch.empty(0).to(device)
    target_y_unlabeled = torch.empty(0).long().to(device)

    ## get source embedding
    source_x_labeled_embedding, source_y_labeled = get_source_embedding(labeled_source_dataloader, encoder, encoder_MLP, source_x_labeled_embedding, source_y_labeled, args)
    source_x_unlabeled_embedding, source_y_unlabeled = get_source_embedding(unlabeled_source_dataloader, encoder, encoder_MLP, source_x_unlabeled_embedding, source_y_unlabeled, args)

    ## get target embedding
    target_x_labeled_embedding, target_y_labeled = get_target_embedding(labeled_target_dataloader, encoder, encoder_MLP, GNet, target_x_labeled_embedding, target_y_labeled, args)
    target_x_unlabeled_embedding, target_y_unlabeled = get_target_embedding(unlabeled_target_dataloader, encoder, encoder_MLP, GNet, target_x_unlabeled_embedding, target_y_unlabeled, args)



    # build mask for conditional KL
    mask_source_labeled = torch.zeros((source_y_labeled.size(0), num_class), device=device).scatter_(1, source_y_labeled.unsqueeze(1), 1)
    mask_target_labeled = torch.zeros((target_y_labeled.size(0), num_class), device=device).scatter_(1, target_y_labeled.unsqueeze(1), 1)
    mask_source_unlabeled = torch.zeros((source_y_unlabeled.size(0), num_class), device=device).scatter_(1, source_y_unlabeled.unsqueeze(1), 1)
    mask_target_unlabeled = torch.zeros((target_y_unlabeled.size(0), num_class), device=device).scatter_(1, target_y_unlabeled.unsqueeze(1), 1)

    source_x_labeled_embedding_cat = torch.cat([source_x_labeled_embedding, mask_source_labeled], dim=1)
    source_x_unlabeled_embedding_cat = torch.cat([source_x_unlabeled_embedding, mask_source_unlabeled], dim=1)
    target_x_labeled_embedding_cat = torch.cat([target_x_labeled_embedding, mask_target_labeled], dim=1)
    target_x_unlabeled_embedding_cat = torch.cat([target_x_unlabeled_embedding, mask_target_unlabeled], dim=1)

    # train gfunction for labeled
    loss_JS_labeled = train_gfunction(optimizer_gfunction_JS_div_labeled, gfunction_JS_div_labeled, source_x_labeled_embedding_cat, target_x_labeled_embedding_cat, mask_source_labeled, mask_target_labeled, device, labeled_JS, args)
    # eval labeled KL for this epoch
    KL_labeled_eval = eval_KL(gfunction_JS_div_labeled, source_x_labeled_embedding_cat, KL_report_js_labeled)


    # train gfunction for unlabeled
    loss_JS_unlabeled = train_gfunction(optimizer_gfunction_JS_div_unlabeled, gfunction_JS_div_unlabeled, source_x_unlabeled_embedding_cat, target_x_unlabeled_embedding_cat, mask_source_unlabeled, mask_target_unlabeled, device, unlabeled_JS, args)
    # eval unlabeled KL for this epoch
    KL_unlabeled_eval = eval_KL(gfunction_JS_div_unlabeled, source_x_unlabeled_embedding_cat, KL_report_js_unlabeled)


    # classifier retrained
    acc_source_labeled_classifier = 0
    acc_target_labeled_classifier = 0
    if args.classifier:
        for i in tqdm(range(args.classifier_epoch)):
            CNet.train()
            optimizer_CNet.zero_grad()
            # optimizer_centerloss.zero_grad()
            pred = CNet(source_x_labeled_embedding)
            acc_source_labeled_classifier = (pred.argmax(-1) == source_y_labeled).sum().item() / pred.size(0)
            loss_source_classifier_labeled = criterion_classifier(pred, source_y_labeled) * args.sclass
            loss_source_classifier_labeled.backward()
            # optimizer_centerloss.step()
            optimizer_CNet.step()

            optimizer_CNet.zero_grad()
            pred = CNet(target_x_labeled_embedding)
            acc_target_labeled_classifier = (pred.argmax(-1) == target_y_labeled).sum().item() / pred.size(0)
            loss_target_classifier_labeled = criterion_classifier(pred, target_y_labeled)
            loss_target_classifier_labeled.backward()
            optimizer_CNet.step()

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
    log_string += "labeled JS: %f, unlabeled JS: %f; labeled KL: %f; unlabeled KL: %f; "%(loss_JS_labeled, loss_JS_unlabeled, KL_labeled_eval, KL_unlabeled_eval)
    if args.classifier: log_string += "src unlbl acc: %f; tgt unlbl acc: %f; "%(acc_source_unlabeled_classifier, acc_target_unlabeled_classifier)
    logger.info(log_string)
    logger.info("-----------------------------------------")

    np.save(args.save_path+model_sub_folder+'/epochs.npy', epochs)
    np.save(args.save_path+model_sub_folder+'/source_acc_label.npy', source_acc_label)
    np.save(args.save_path+model_sub_folder+'/source_acc_unlabel.npy', source_acc_unlabel)
    np.save(args.save_path+model_sub_folder+'/target_acc_label.npy', target_acc_label)
    np.save(args.save_path+model_sub_folder+'/target_acc_unlabel.npy', target_acc_unlabel)
    np.save(args.save_path+model_sub_folder+'/KL_report_js_labeled.npy', KL_report_js_labeled)
    np.save(args.save_path+model_sub_folder+'/KL_report_js_unlabeled.npy', KL_report_js_unlabeled)


    np.save(args.save_path+model_sub_folder+'/labeled_JS.npy', labeled_JS)
    np.save(args.save_path+model_sub_folder+'/unlabeled_JS.npy', unlabeled_JS)

    if args.classifier:
        np.save(args.save_path+model_sub_folder+'/acc_source_unlabeled_classifier_.npy', acc_source_unlabeled_classifier_)
        np.save(args.save_path+model_sub_folder+'/acc_target_unlabeled_classifier_.npy', acc_target_unlabeled_classifier_)


