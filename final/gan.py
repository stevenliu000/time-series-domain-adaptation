import sys, os, inspect
import numpy as np
import random
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
from models.GAN import Generator, Discriminator
from utils import *
from data_utils import *
import argparse
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

###############################################################################
#                            Function Definition                              #
###############################################################################
def _gradient_penalty(real_data, generated_data, DNet, mask, num_class, device, args, local=True):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = DNet(interpolated, mask) # mask = 1 if local=False

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon

    if local:
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1, keepdim=True) + 1e-12)
        gradients_norm = gradients_norm * mask
        return args.gpweight * ((gradients_norm - 1) ** 2).mean(dim=0)
    else:
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return args.gpweight * ((gradients_norm - 1) ** 2).mean()

def train_classifier(labeled_dataloader, CNet, encoder, encoder_MLP, GNet, optimizerCNet, optimizerG, optimizerEncoder, optimizerEncoderMLP, args, target):
    acc = 0.0
    num_datas = 0.0
    CNet.train()
    encoder.train()
    encoder_MLP.train()
    if target:
        GNet.train()
    # Loop over mini batch for traininig
    for batch_id, (x, y) in tqdm(enumerate(labeled_dataloader), total=len(labeled_dataloader)):
        # clear optimizer
        optimizerCNet.zero_grad()
        optimizerG.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerEncoderMLP.zero_grad()

        # load data
        x = x.to(device).float()
        y = y.to(device)
        num_datas += x.size(0)
        # encoder inference
        x_embedding = encoder_inference(encoder, encoder_MLP, x)
        if target:
            fake_source_embedding = GNet(x_embedding)
            pred = CNet(fake_source_embedding)
        else:
            pred = CNet(x_embedding)

        # acc and loss
        acc += (pred.argmax(-1) == y).sum().item()
        loss = criterion_classifier(pred, y)
        loss.backward()

        # update CNet, GNet, Encoder, EncoderMLP
        optimizerCNet.step()
        optimizerG.step()
        optimizerEncoder.step()
        optimizerEncoderMLP.step()
    acc = acc / num_datas
    return acc, loss.item()

def eval_classifier(unlabeled_dataloader, CNet, encoder, encoder_MLP, GNet, target=False):
    acc_unlabel = 0.0
    num_datas = 0.0
    CNet.eval()
    encoder.eval()
    encoder_MLP.eval()
    if target:
        GNet.eval()
    for batch_id, (x, y) in tqdm(enumerate(unlabeled_dataloader), total=len(unlabeled_dataloader)):
        x = x.to(device).float()
        y = y.to(device)
        num_datas += x.shape[0]
        x_embedding = encoder_inference(encoder, encoder_MLP, x)
        if target:
            fake_source_embedding = GNet(x_embedding)
            pred = CNet(fake_source_embedding)
        else:
            pred = CNet(x_embedding)
        acc_unlabel += (pred.argmax(-1) == y).sum().item()

    acc_unlabel = acc_unlabel/num_datas
    return acc_unlabel


def update_D(source_embedding, fake_source_embedding, DNet, optimizerD, source_mask, target_mask, source_weight_count, target_weight_count, target_weight, num_class, total_error_D, device, args, local=True):
    optimizerD.zero_grad()
    # adversarial loss
    if local:
        source_DNet_local = DNet(source_embedding, source_mask)
        target_DNet_local = DNet(fake_source_embedding, target_mask)
        source_DNet_local_mean = source_DNet_local.sum(dim=0) / source_weight_count
        target_DNet_local_mean = (target_DNet_local * target_weight).sum(dim=0) / target_weight_count
        # gradient penalty
        gp = _gradient_penalty(source_embedding, fake_source_embedding, DNet, source_mask, num_class, device, args)
        # loss
        loss_D_local = (target_DNet_local_mean - source_DNet_local_mean + gp).sum()
        loss_D = loss_D_local * args.dlocal

    else:
        # loss and gradient penalty
        loss_D_global = DNet_global(fake_source_embedding,1).mean() - DNet_global(source_embedding,1).mean()
        gradient_penalty = _gradient_penalty(source_embedding, fake_source_embedding, DNet_global, 1, num_class, device, args)
        loss_D_global = loss_D_global + gradient_penalty
        loss_D = loss_D_global * args.dglobal

    # accumulate loss for report
    total_error_D += loss_D.item()

    # backward and update D
    loss_D.backward()
    optimizerD.step()

    return total_error_D


def update_G(fake_source_embedding, DNet, optimizerG, total_error_G, target_mask, target_weight, target_weight_count, args, local=True):
    # clear optimizer
    optimizerG.zero_grad()
    # calculte loss for Generator
    if local:
        target_DNet_local = DNet(fake_source_embedding, target_mask)
        target_DNet_local_mean = (target_DNet_local * target_weight).sum(dim=0) / target_weight_count
        loss_G = -target_DNet_local_mean.sum()
        loss_G = loss_G * args.dlocal
    else:
        loss_G = -DNet(fake_source_embedding,1).mean() * args.dglobal

    # accumulate loss for report
    total_error_G += loss_G.item()

    # backward and update G
    loss_G.backward()
    optimizerG.step()

    return total_error_G


###############################################################################
#                                 parameters                                  #
###############################################################################
"""
Default contains the optimal hyperparameters for conditional (local) GAN (target and source lbl 70%). Use --isglobal and --islocal to choose global or conditional GAN for training. Except for the parameter mentioned in example below, the other parameters between local and conditional gan are shared.

Example of command to run:
$ python gan.py --islocal 1 --n_critic 6 --gpweight 10

$ python gan.py --isglobal 1 --n_critic 12 --gpweight 5
"""
parser = argparse.ArgumentParser(description='WGAN for time series domain adaptation')
parser.add_argument("--data_path", type=str, default="../data_unzip/", help="dataset path")
parser.add_argument("--task", type=str, default='3E', help='3A or 3E')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')
parser.add_argument('--lr_gan', type=float, default=1e-3, help='learning rate for adversarial')
parser.add_argument('--lr_CNet', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--n_critic', type=int, default=6, help='gap: Generator train GAP times, discriminator train once')
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of target labeled data')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of source labeled data')
parser.add_argument('--num_per_class', type=int, default=3, help='Number of sample per class inside each batch when training local discriminator')
parser.add_argument('--save_path', type=str, default="../train_related", help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')
parser.add_argument('--gpweight', type=float, default=10, help='clip_value for WGAN')
parser.add_argument('--sclass', type=float, default=0.7, help='source domain classification weight on loss function')
parser.add_argument('--dlocal', type=float, default=0.01, help='local GAN weight on loss function')
parser.add_argument('--dglobal', type=float, default=0.01, help='global GAN weight on loss function')
parser.add_argument('--isglobal', type=int, default=0, help='if using global DNet')
parser.add_argument('--islocal', type=int, default=0, help='if using local (conditional) DNet')

args = parser.parse_args()

args.isglobal = True if args.isglobal == 1 else False
args.islocal = True if args.islocal == 1 else False
# Check at least one mode for GAN is true
assert args.islocal or args.isglobal, "Please at least use one kind of GAN for training, otherwise please refer to 'naive_apdation.py'."

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.task = '3Av2' if args.task == '3A' else '3E'
num_class = 50 if args.task == "3Av2" else 65

###############################################################################
#                             Save Path                                       #
###############################################################################
model_sub_folder = 'results/GAN_adaptation/lbl_percent_%f/task_%s_gpweight_%f_ncritic_%i_sclass_%f_global_%i_dglobal_%f_local_%i_dlocal_%f'%(args.target_lbl_percentage, args.task, args.gpweight, args.n_critic, args.sclass, args.isglobal, args.dglobal, args.islocal, args.dlocal)

save_folder = os.path.abspath(os.path.join(args.save_path, model_sub_folder))
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

###############################################################################
#                               Logger                                        #
###############################################################################
logger = get_logger(save_folder, device, args)


###############################################################################
#                                 Data Loading                                #
###############################################################################
# get target dataset and dataloader
labeled_target_x, labeled_target_y, unlabeled_target_x, unlabeled_target_y = read_data(args.task, 'target', args.target_lbl_percentage, args.data_path)
labeled_target_dataset = SingleDataset(labeled_target_x, labeled_target_y)
unlabeled_target_dataset = SingleDataset(unlabeled_target_x, unlabeled_target_y)
labeled_target_dataloader = DataLoader(labeled_target_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
unlabeled_target_dataloader = DataLoader(unlabeled_target_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

# get source dataset and dataloader
labeled_source_x, labeled_source_y, unlabeled_source_x, unlabeled_source_y = read_data(args.task, 'source', args.source_lbl_percentage, args.data_path)
labeled_source_dataset = SingleDataset(labeled_source_x, labeled_source_y)
unlabled_source_dataset = SingleDataset(unlabeled_source_x, unlabeled_source_y)
labeled_source_dataloader = DataLoader(labeled_source_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
unlabeled_source_dataloader = DataLoader(unlabled_source_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

# create join dataset and dataloader
join_dataset = JoinDataset(labeled_source_x, labeled_source_y, labeled_target_x, labeled_target_y, random=True)
join_dataloader = DataLoader(join_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

# create class data dict for local Discrimitor training
source_labeled_dict = get_class_data_dict(labeled_source_x, labeled_source_y, num_class)
target_labeled_dict = get_class_data_dict(labeled_target_x, labeled_target_y, num_class)


###############################################################################
#                               Model Creation                                #
###############################################################################
seq_len = 10
feature_dim = 160

# model
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
if args.isglobal:
    DNet_global = Discriminator(feature_dim=64*2, d_out=1).to(device)
if args.islocal:
    DNet_local = Discriminator(feature_dim=64*2, d_out=num_class).to(device)

GNet = Generator(dim=64*2).to(device)

# loss
criterion_classifier = nn.CrossEntropyLoss().to(device)

# optimizer
if args.isglobal:
    optimizerD_global = torch.optim.Adam(DNet_global.parameters(), lr=args.lr_gan)
if args.islocal:
    optimizerD_local = torch.optim.Adam(DNet_local.parameters(), lr=args.lr_gan)
optimizerG = torch.optim.Adam(GNet.parameters(), lr=args.lr_gan)
optimizerCNet = torch.optim.Adam(CNet.parameters(), lr=args.lr_CNet)
optimizerEncoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder)
optimizerEncoderMLP = torch.optim.Adam(encoder_MLP.parameters(), lr=args.lr_encoder)


###############################################################################
#                               Model Initilize                               #
###############################################################################
if args.isglobal:
    DNet_global.apply(weights_init)
if args.islocal:
    DNet_local.apply(weights_init)
GNet.apply(weights_init)
encoder.apply(weights_init)
encoder_MLP.apply(weights_init)
CNet.apply(weights_init)


###############################################################################
#                                   Train                                     #
###############################################################################
target_acc_label_ = []
source_acc_label_ = []
source_acc_unlabel_ = []
target_acc_unlabel_ = []
error_D_global = []
error_G_global = []
error_D_local = []
error_G_local = []

print('Started training')
for epoch in range(args.epochs):
    # Train classifier
    # train on source data
    source_acc, _ = train_classifier(labeled_source_dataloader, CNet, encoder, encoder_MLP, None, optimizerCNet, optimizerG, optimizerEncoder, optimizerEncoderMLP, args, target=False)
    source_acc_label_.append(source_acc)

    # train on target data
    target_acc, _ = train_classifier(labeled_target_dataloader, CNet, encoder, encoder_MLP, GNet, optimizerCNet, optimizerG, optimizerEncoder, optimizerEncoderMLP, args, target=True)
    target_acc_label_.append(target_acc)

    # Eval classifier
    # source domain
    source_acc_unlabel = eval_classifier(unlabeled_source_dataloader, CNet, encoder, encoder_MLP, GNet, target=False)
    source_acc_unlabel_.append(source_acc_unlabel)

    # target domain
    target_acc_unlabel = eval_classifier(unlabeled_target_dataloader, CNet, encoder, encoder_MLP, GNet, target=True)
    target_acc_unlabel_.append(target_acc_unlabel)

    logger.info('Epoch: %i, update classifier: source acc: %f; source unlbl acc: %f; target acc: %f; target unlabel acc: %f'%(epoch+1, source_acc, source_acc_unlabel, target_acc, target_acc_unlabel))

    # Train GAN
    if args.isglobal:
        # Train Global GAN
        CNet.eval()
        encoder.eval()
        encoder_MLP.eval()
        GNet.train()
        DNet_global.train()
        total_error_D_global = 0
        total_error_G = 0

        for batch_id, ((source_x, source_y), (target_x, target_y)) in tqdm(enumerate(join_dataloader), total=len(join_dataloader)):
            # get embeddings
            source_data = source_x.to(device).float()
            source_embedding = encoder_inference(encoder, encoder_MLP, source_data)
            target_data = target_x.to(device).float()
            target_embedding = encoder_inference(encoder, encoder_MLP, target_data)
            fake_source_embedding = GNet(target_embedding).detach()

            # Update G Network
            total_error_G = update_G(fake_source_embedding, DNet_global, optimizerG, total_error_G, None, None, None, args, local=False)

            # Update D_global Network
            if batch_id % args.n_critic == 0:
                total_error_D_global = update_D(source_embedding, fake_source_embedding, DNet_global, optimizerD_global, None, None, None, None, None, num_class, total_error_D_global, device, args, local=False)

        logger.info('Epoch: %i, Global Discrimator Updates: Loss D_global: %f, Loss G: %f; Update_ratio (G/D): %i'%(epoch+1, total_error_D_global, total_error_G, args.n_critic))

        # record loss
        error_D_global.append(total_error_D_global)
        error_G_global.append(total_error_G)


    if args.islocal:
        # Update local Discriminator
        total_error_D_local = 0
        total_error_G = 0
        CNet.eval()
        encoder.eval()
        encoder_MLP.eval()
        GNet.train()
        DNet_local.train()

        for batch_id in tqdm(range(math.ceil(labeled_target_x.shape[0]/args.batch_size))):
            # load batch data
            target_x, target_y, target_weight = get_batch_target_data_on_class(target_labeled_dict, args.num_per_class, None, no_pesudo=True)
            source_x, source_y = get_batch_source_data_on_class(source_labeled_dict, args.num_per_class)
            source_x = torch.Tensor(source_x).to(device).float()
            target_x = torch.Tensor(target_x).to(device).float()
            source_y = torch.LongTensor(target_y).to(device)
            target_y = torch.LongTensor(target_y).to(device)
            target_weight = torch.Tensor(target_weight).to(device)

            # create mask
            source_mask = torch.zeros(source_x.size(0), num_class).to(device).scatter_(1, source_y.unsqueeze(-1), 1)
            target_mask = torch.zeros(target_x.size(0), num_class).to(device).scatter_(1, target_y.unsqueeze(-1), 1)
            target_weight = torch.zeros(target_x.size(0), num_class).to(device).scatter_(1, target_y.unsqueeze(-1), target_weight.unsqueeze(-1))
            source_weight_count = source_mask.sum(dim=0)
            target_weight_count = target_weight.sum(dim=0)

            # get embeddings from encoder
            source_embedding = encoder_inference(encoder, encoder_MLP, source_x)
            target_embedding = encoder_inference(encoder, encoder_MLP, target_x)
            fake_source_embedding = GNet(target_embedding).detach()

            # Update G Network
            total_error_G = update_G(fake_source_embedding, DNet_local, optimizerG, total_error_G, target_mask, target_weight, target_weight_count, args, local=True)

            # Update D local Network
            if batch_id % args.n_critic == 0:
                total_error_D_local = update_D(source_embedding, fake_source_embedding, DNet_local, optimizerD_local, source_mask, target_mask, source_weight_count, target_weight_count, target_weight, num_class, total_error_D_local, device, args, local=True)

        logger.info('Epoch: %i, Local Discrimator Updates: Loss D_local: %f, Loss G: %f; Update_ratio (G:D): %i'%(epoch+1, total_error_D_local, total_error_G, args.n_critic))

        # record loss
        error_D_local.append(total_error_D_local)
        error_G_global.append(total_error_G)


    # Save models and results
    np.save(os.path.join(args.save_path, model_sub_folder, 'target_acc_label_.npy'),target_acc_label_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_acc_label_.npy'),source_acc_label_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'target_acc_unlabel_.npy'),target_acc_unlabel_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_acc_unlabel_.npy'),source_acc_unlabel_)
    if args.isglobal:
        np.save(os.path.join(args.save_path, model_sub_folder, 'error_D_global.npy'),error_D_global)
        np.save(os.path.join(args.save_path, model_sub_folder, 'error_G_global.npy'),error_G_global)
    if args.islocal:
        np.save(os.path.join(args.save_path, model_sub_folder, 'error_D_local.npy'),error_D_local)
        np.save(os.path.join(args.save_path, model_sub_folder, 'error_G_local.npy'),error_G_local)

    if epoch % args.model_save_period == 0:
        torch.save(CNet.state_dict(), os.path.join(args.save_path,model_sub_folder, 'CNet_%i.t7'%(epoch+1)))
        torch.save(GNet.state_dict(), os.path.join(args.save_path,model_sub_folder, 'GNet_%i.t7'%(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join(args.save_path,model_sub_folder,'encoder_%i.t7'%(epoch+1)))
        torch.save(encoder_MLP.state_dict(), os.path.join(args.save_path,model_sub_folder,'encoder_MLP_%i.t7'%(epoch+1)))
        if args.isglobal:
            torch.save(DNet_global.state_dict(),  os.path.join(args.save_path, model_sub_folder, 'DNet_global_%i.t7'%(epoch+1)))
        if args.islocal:
            torch.save(DNet_local.state_dict(), os.path.join(args.save_path, model_sub_folder, 'DNet_local_%i.t7'%(epoch+1)))


