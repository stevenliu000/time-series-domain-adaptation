import numpy as np
import os
import argparse
import random
from tqdm import tqdm
import datetime
import time


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from martins.complex_transformer import ComplexTransformer
from martins.dataset import TimeSeriesDataset
from FNN import FNN
import argparse

parser = argparse.ArgumentParser(description='Time series adaptation')
parser.add_argument("--data_path", type=str, default="/Users/tianqinli/Code/Working-on/Russ/time-series-domain-adaptation/data_unzip/", help="dataset folder path")
parser.add_argument('--train_file', type=str, default="train_{}.pkl", help='which training file to perform')
parser.add_argument('--vali_file', type=str, default="validation_{}.pkl", help='which validation file to perform')
parser.add_argument("--task", type=str, default="3Av2", help='3Av2 or 3E')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr_clf', type=float, default=1e-4, help='learning rate for classification')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--PATH', type=str, default= "/Users/tianqinli/Code/Working-on/Russ/time-series-domain-adaptation/data_results/", help='Model save location')
parser.add_argument('--log', type=str, default="FNN_log.out", help="Output log file for training and validation loss")
parser.add_argument('--job_type', type=str, default="source", help="choose form source or target")
parser.add_argument('--model_save_steps', type=int, default=5, help="steps between which saves the model")


args = parser.parse_args()

# other parameters
seq_len = 10
feature_dim = 160
d_out = 50 if args.task == "3Av2" else 65
device = torch.device("cuda:0")



training_set = TimeSeriesDataset(root_dir=args.data_path, file_name=args.train_file.format(args.task), train=True)
vali_set = TimeSeriesDataset(root_dir=args.data_path, file_name=args.vali_file.format(args.task), train=True)
# test_set = TimeSeriesDataset(root_dir=args.data_path, file_name=args.file.format(args.task), train=False)






train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
vali_loader = DataLoader(vali_set, batch_size=args.batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)



encoder = ComplexTransformer(layers=3,
                               time_step=seq_len,
                               input_dim=feature_dim,
                               hidden_size=64,
                               output_dim=64,
                               num_heads=8,
                               out_dropout=0.2,
                               leaky_slope=0.2)

if torch.cuda.is_available(): encoder.to(device)


CNet = FNN(d_in=64 * 2 * seq_len, d_h1=500, d_h2=500, d_out=d_out, dp=0.2)
if torch.cuda.is_available():
    CNet = CNet.to(device)

params = list(encoder.parameters()) + list(CNet.parameters())
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    criterion = criterion.cuda()




optimizer = torch.optim.Adam(params, lr=args.lr_clf)
scheduler = ReduceLROnPlateau(optimizer, 'min')


# Encoding by complex transformer
x_mean_tr = training_set.data_mean
x_std_tr = training_set.data_std

#x_mean_te = test_set.data_mean
#x_std_te = test_set.data_std

best_acc_train = best_acc_test = 0

unique_id = args.job_type + "-b" + str(args.batch_size) + ".e" + str(args.epochs) + ".lr" + str(args.lr_clf) + ".task" + args.task


folder_path = args.PATH + unique_id + ".model" + "/"
os.makedirs(folder_path, exist_ok=True)


logfile_full_path = folder_path + args.log + unique_id
with open(logfile_full_path, 'w') as f:
    f.write("Epoch\tTime\ttrain_acc\ttrain_loss\tvalidation_acc")
    f.write("\n")

#### train
for epoch in range(args.epochs):
    """Training"""
    correct_train = 0
    total_bs_train = 0 # total batch size
    train_loss = 0
    start_train_time = time.time()
    for batch_id, (x, y) in enumerate(tqdm(train_loader)):
        if torch.cuda.is_available():
            x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]
        #normalize data
        x = (x - x_mean_tr) / x_std_tr
        # take the real and imaginary part out
        real = x[:,:,0].reshape(batch_size, seq_len, feature_dim).float()
        imag = x[:,:,1].reshape(batch_size, seq_len, feature_dim).float()
        if torch.cuda.is_available():
            real.to(device)
            imag.to(device)
        real, imag = encoder(real, imag)
        # print(real.shape)
        pred = CNet(torch.cat((real, imag), -1).reshape(x.shape[0], -1))
        loss = criterion(pred, y.argmax(-1))
        #print(pred.argmax(-1), y.argmax(-1))
        correct_train += (pred.argmax(-1) == y.argmax(-1)).sum().item()
        total_bs_train += y.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.shape[0]
    train_acc = float(correct_train) / total_bs_train
    train_loss = train_loss / total_bs_train
    best_acc_train = max(best_acc_train, train_acc)

    CNet_PATH = folder_path + "CNet_" + "model.ep" + str(epoch) + ".model"
    encoder_PATH = folder_path + "Encoder_" + "model.ep" + str(epoch) + ".model"
    
    if epoch % args.model_save_steps == 0:
        torch.save(CNet.state_dict(), CNet_PATH)
        torch.save(encoder.state_dict(), encoder_PATH)

    time_now = datetime.datetime.now()
    end_train_time = time.time()
    train_epoch_time = round(end_train_time - start_train_time, 2)
    logstr = str(time_now) + "\nepoch " + str(epoch) + ": train_acc: "+ str(train_acc) + "; train_loss: " + str(train_loss)


    with open(logfile_full_path, 'a') as f:
        f.write(str(epoch) + "\t" + str(train_epoch_time) + "\t" + str(train_acc) + "\t" + str(train_loss) + "\t")


    """Testing"""
    correct_vali = 0
    total_bs_vali = 0
    vali_loss = 0
    for batch_id, (x, y) in enumerate(vali_loader):
        if torch.cuda.is_available():
            x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]
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
            loss = criterion(pred, y.argmax(-1))
            #print(pred.argmax(-1), y.argmax(-1))
            correct_vali += (pred.argmax(-1) == y.argmax(-1)).sum().item()
            total_bs_vali += y.shape[0]
    vali_acc = float(correct_vali) / total_bs_vali
    vali_log_str = " validation_acc: "+ str(vali_acc)

    with open(logfile_full_path, 'a') as f:
        f.write(str(vali_acc))
        f.write("\n")
    print(logstr + vali_log_str)
