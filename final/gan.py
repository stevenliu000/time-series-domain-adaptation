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
from models.GAN import Generator, Discriminator
from utils import *
import argparse
import logging
from data_utils import *
import random
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


def _gradient_penalty(real_data, generated_data, DNet, mask, num_class, device, args):
    '''
    ref: https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
    '''
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
    if args.isglobal:
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    else:
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1, keepdim=True) + 1e-12)

        gradients_norm = gradients_norm * mask
    
    # Return gradient penalty
    if args.isglobal:
        return args.gpweight * ((gradients_norm - 1) ** 2).mean()
    else:
        return args.gpweight * ((gradients_norm - 1) ** 2).mean(dim=0)


def train_classification(if_target, CNet, encoder, encoder_MLP, GNet, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, optimizerG, labeled_dataloader, args):
    CNet.train()
    encoder.train()
    encoder_MLP.train()
    GNet.train()
    acc = 0.0
    num_datas = 0.0
    for batch_id, (x, y) in tqdm(enumerate(labeled_dataloader), total=len(labeled_dataloader)):
        optimizerCNet.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerEncoderMLP.zero_grad()
        if if_target: optimizerG.zero_grad()
        x = x.to(device).float()
        y = y.to(device)
        num_datas += x.size(0)
        x_embedding = encoder_inference(encoder, encoder_MLP, x)
        if if_target: x_embedding = GNet(x_embedding)
        pred = CNet(x_embedding)
        acc += (pred.argmax(-1) == y).sum().item()
        loss = criterion_classifier(pred, y) 
        if not if_target: loss = loss * args.sclass
        loss.backward()
        optimizerCNet.step()
        if if_target: optimizerG.step()
        optimizerEncoder.step()
        optimizerEncoderMLP.step()

    acc = acc / num_datas
    return acc

def eval_classification(if_target, CNet, encoder, encoder_MLP, GNet, unlabeled_dataloader, args):
    correct_target = 0.0
    pesudo_y = []
    CNet.eval()
    encoder.eval()
    encoder_MLP.eval()
    GNet.eval()
    with torch.no_grad():
        for batch_id, (x, y) in tqdm(enumerate(unlabeled_dataloader), total=len(unlabeled_dataloader)):
            x = x.to(device).float()
            y = y.to(device)
            num_datas += x.shape[0]
            x_embedding = encoder_inference(encoder, encoder_MLP, x)
            if if_target: x_embedding = GNet(x_embedding)
            pred = CNet(x_embedding)
            acc_unlabel += (pred.argmax(-1) == y).sum().item()
            pesudo_y.extend(pred.argmax(-1).cpu().numpy())

    pesudo_y = np.array(pesudo_y)
    acc_unlabel = acc_unlabel/num_datas

    return acc_unlabel, pesudo_y

def train_global_GAN(CNet, encoder, encoder_MLP, GNet, DNet_global, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, optimizerG, optimizerD_global, join_dataloader, args):
    CNet.train()
    encoder.train()
    encoder_MLP.train()
    GNet.train()
    DNet_global.train()
    total_error_D_global = 0
    total_error_G = 0
    for batch_id, ((source_x, source_y), (target_x, target_y)) in tqdm(enumerate(join_dataloader), total=len(join_dataloader)):
        optimizerD_global.zero_grad()
        optimizerG.zero_grad()
        source_data = source_x.to(device).float()
        source_embedding = encoder_inference(encoder, encoder_MLP, source_data)
        target_data = target_x.to(device).float()
        target_embedding = encoder_inference(encoder, encoder_MLP, target_data)
        fake_source_embedding = GNet(target_embedding)
        """Update G Network"""

        # adversarial loss
        loss_G = -DNet_global(fake_source_embedding,1).mean()
        total_error_G += loss_G.item() * args.dgan

        loss_G.backward()
        optimizerG.step()

        if batch_id % args.n_critic == 0:
            """Update D Net"""
            optimizerD_global.zero_grad()
            source_data = source_x.to(device).float()
            source_embedding = encoder_inference(encoder, encoder_MLP, source_data)
            target_data = target_x.to(device).float()
            target_embedding = encoder_inference(encoder, encoder_MLP, target_data)
            fake_source_embedding = GNet(target_embedding).detach()
            # adversarial loss
            loss_D_global = DNet_global(fake_source_embedding,1).mean() - DNet_global(source_embedding,1).mean()
            gradient_penalty = _gradient_penalty(source_embedding, fake_source_embedding, DNet_global, 1, num_class, device, args)

            loss_D_global = loss_D_global + gradient_penalty
            loss_D_global = loss_D_global * args.dgan
            total_error_D_global += loss_D_global.item()

            loss_D_global.backward()
            optimizerD_global.step()

    return total_error_D_global, total_error_G

def train_local_GAN(CNet, encoder, encoder_MLP, GNet, DNet_local, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, optimizerG, optimizerD_local, join_dataloader, args):
    # Update local Discriminator
    total_error_D_local = 0
    total_error_G = 0
    encoder.eval()
    encoder_MLP.eval()
    GNet.train()
    DNet_local.train()
    for batch_id, ((source_x, source_y), (target_x, target_y, target_weight)) in tqdm(enumerate(join_dataloader), total=len(join_dataloader)):
        source_x = source_x.to(device)
        target_x = target_x.to(device)
        source_y = source_y.to(device)
        target_y = target_y.to(device)
        target_weight = target_weight.to(device)

        source_mask = torch.zeros(source_x.size(0), num_class).to(device).scatter_(1, source_y.unsqueeze(-1), 1)
        target_mask = torch.zeros(target_x.size(0), num_class).to(device).scatter_(1, target_y.unsqueeze(-1), 1)
        target_weight = torch.zeros(target_x.size(0), num_class).to(device).scatter_(1, target_y.unsqueeze(-1), target_weight.unsqueeze(-1))

        source_weight_count = source_mask.sum(dim=0)
        target_weight_count = target_weight.sum(dim=0)

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
        loss_G = loss_G * args.dgan
        total_error_G += loss_G.item()
        loss_G.backward()
        optimizerG.step()

        if batch_id % args.n_critic == 0:
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
            loss_D_local = loss_D_local * args.dgan

            total_error_D_local += loss_D_local.item()

            loss_D_local.backward()
            optimizerD_local.step()

    return total_error_D_local, total_error_G

###############################################################################
#                                 parameters                                  #
###############################################################################
parser = argparse.ArgumentParser(description='GAN')
parser.add_argument("--data_path", type=str, required=True, help='dataset path')
parser.add_argument("--task", type=str, required=True, help='3Av2 or 3E')
parser.add_argument('--batch_size', type=int, default=2000, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr_gan', type=float, default=1e-3, help='learning rate for adversarial')
parser.add_argument('--lr_FNN', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate for classification')
parser.add_argument('--n_critic', type=float, default=0.16, help='gap: Generator train GAP times, discriminator train once')
parser.add_argument('--target_lbl_percentage', type=float, default=0.7, help='percentage of target labeled data')
parser.add_argument('--source_lbl_percentage', type=float, default=0.7, help='percentage of source labeled data')
parser.add_argument('--num_per_class', type=int, default=2, help='number of sample per class when training local discriminator')
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--save_path', type=str, required=True, help='where to store data')
parser.add_argument('--model_save_period', type=int, default=2, help='period in which the model is saved')
parser.add_argument('--gpweight', type=float, default=10, help='gradient penalty for WGAN-gp')
parser.add_argument('--sclass', type=float, default=0.7, help='source domain classification weight on loss function')
parser.add_argument('--dgan', type=float, default=0.01, help='GAN weight on loss function')
parser.add_argument('--isglobal', type=int, default=0, help='if using global GAN')
parser.add_argument('--pure_random', type=int, default=1, help='Pure random for n_critic')

args = parser.parse_args()
args.isglobal = True if args.isglobal == 1 else False
args.pure_random = True if args.pure_random == 1 else False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

assert args.task in ['3Av2', '3E']
num_class = 50 if args.task == "3Av2" else 65

if args.num_per_class == -1:
    args.num_per_class = math.ceil(args.batch_size / num_class)

model_sub_folder = 'result/global_GAN' if args.isglobal else 'result/conditional_GAN'
model_sub_folder += '/lbl_percent_%f/task_%s_gpweight_%f_dgan_%f_ncritic_%f_sclass_%f'%(args.target_lbl_percentage, args.task, args.gpweight, args.dgan, args.n_critic, args.sclass)

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

join_dataset = JoinDataset(labeled_source_x, labeled_source_y, labeled_target_x, labeled_target_y, random=True)
join_dataloader = DataLoader(join_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

source_labeled_dict = get_class_data_dict(labeled_source_x, labeled_source_y, num_class)
target_labeled_dict = get_class_data_dict(labeled_target_x, labeled_target_y, num_class)

label_target_len = labeled_target_x.shape[0]

classwise_dataloader = ClassWiseDataLoader(label_target_len, args.batch_size, args.num_per_class, source_labeled_dict, target_labeled_dict)

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
                               leaky_slope=0.2)
encoder.to(device)
encoder_MLP = FNNSeparated(d_in=64*2, d_h1=64*4, d_h2=64*2, dp=0.2).to(device)
CNet = FNNLinear(d_h2=64*2, d_out=num_class).to(device)

DNet_global = Discriminator(feature_dim=64*2, d_out=1).to(device)
DNet_local = Discriminator(feature_dim=64*2, d_out=num_class).to(device)
GNet = Generator(dim=64*2).to(device)

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


###############################################################################
#                                    Training                                 #
###############################################################################



target_acc_label_ = []
source_acc_ = []
source_acc_unlabel_ = []
target_acc_unlabel_ = []
if args.isglobal:
    error_D_global = []
    error_G_global = []
else:
    error_D_local = []
    error_G_local = []

print('Started training')
for epoch in range(args.epochs):
    # update classifier
    # on source domain
    source_acc = train_classification(False, CNet, encoder, encoder_MLP, GNet, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, optimizerG, labeled_source_dataloader, args)
    source_acc_.append(source_acc)

    # update classifier
    # on target domain
    target_acc = train_classification(True, CNet, encoder, encoder_MLP, GNet, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, optimizerG, labeled_source_dataloader, args)
    target_acc_label_.append(target_acc)

    # eval
    # source_domain
    source_acc_unlabel, _ = eval_classification(False, CNet, encoder, encoder_MLP, GNet, unlabeled_source_dataloader, args)
    source_acc_unlabel_.append(source_acc_unlabel)

    # eval
    # target_domain
    target_acc_unlabel, target_pesudo_y = eval_classification(True, CNet, encoder, encoder_MLP, GNet, unlabeled_target_dataloader, args)
    pesudo_dict = get_class_data_dict(unlabeled_target_x, target_pesudo_y, num_class)
    target_acc_unlabel_.append(target_acc_unlabel)

    logger.info('Epoch: %i, update classifier: source acc: %f; source unlbl acc: %f; target acc: %f; target unlabel acc: %f'%(epoch+1, source_acc, source_acc_unlabel, target_acc, target_acc_unlabel))

    # Update GAN
    if args.isglobal:
        total_error_D_global, total_error_G = train_global_GAN(CNet, encoder, encoder_MLP, GNet, DNet_global, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, optimizerG, optimizerD_global, join_dataloader, args)

        error_D_global.append(total_error_D_global)
        error_G_global.append(total_error_G)

        logger.info('Epoch: %i, Global Discrimator Updates: Loss D_global: %f, Loss G: %f; update_ratio: %i'%(epoch+1, total_error_D_global, total_error_G, args.n_critic))

    else:
        # Update local Discriminator
        total_error_D_local, total_error_G = train_local_GAN(CNet, encoder, encoder_MLP, GNet, DNet_local, optimizerCNet, optimizerEncoder, optimizerEncoderMLP, optimizerG, optimizerD_local, classwise_dataloader, args)

        error_D_local.append(total_error_D_local)
        error_G_global.append(total_error_G)

        logger.info('Epoch: %i, Local Discrimator Updates: Loss D_local: %f, Loss G: %f; update_ratio: %i'%(epoch+1, total_error_D_local, total_error_G, args.n_critic))

    # save
    np.save(os.path.join(args.save_path, model_sub_folder, 'target_acc_label_.npy'),target_acc_label_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_acc_.npy'),source_acc_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'target_acc_unlabel_.npy'),target_acc_unlabel_)
    np.save(os.path.join(args.save_path, model_sub_folder, 'source_acc_unlabel_.npy'),source_acc_unlabel_)
    if args.isglobal:
        np.save(os.path.join(args.save_path, model_sub_folder, 'error_D_global.npy'),error_D_global)
        np.save(os.path.join(args.save_path, model_sub_folder, 'error_G_global.npy'),error_G_global)
    else:
        np.save(os.path.join(args.save_path, model_sub_folder, 'error_D_local.npy'),error_D_local)
        np.save(os.path.join(args.save_path, model_sub_folder, 'error_G_local.npy'),error_G_local)

    if epoch % args.model_save_period == 0:
        torch.save(CNet.state_dict(), os.path.join(args.save_path,model_sub_folder, 'CNet_%i.t7'%(epoch+1)))
        torch.save(GNet.state_dict(), os.path.join(args.save_path,model_sub_folder, 'GNet_%i.t7'%(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join(args.save_path,model_sub_folder,'encoder_%i.t7'%(epoch+1)))
        torch.save(encoder_MLP.state_dict(), os.path.join(args.save_path,model_sub_folder,'encoder_MLP_%i.t7'%(epoch+1)))
        if args.isglobal:
            torch.save(DNet_global.state_dict(),  os.path.join(args.save_path,model_sub_folder,'DNet_global_%i.t7'%(epoch+1)))
        else:
            torch.save(DNet_local.state_dict(), os.path.join(args.save_path,model_sub_folder, 'DNet_local_%i.t7'%(epoch+1)))




