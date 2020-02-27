import numpy as np
import argparse
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset import TimeSeriesDataset, TimeSeriesDatasetConcat
from complex_transformer import ComplexTransformer

parser = argparse.ArgumentParser(description='Time series adaptation')
parser.add_argument("--data-path", type=str, default="/projects/rsalakhugroup/complex/domain_adaptation", help="dataset path")
parser.add_argument("--task", type=str, help='3A or 3E')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs_gan', type=int, default=50, help='number of epochs for training GAN')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr_gan', type=float, default=1e-4, help='learning rate for adversarial')
parser.add_argument('--lr_clf', type=float, default=1e-4, help='learning rate for classification')
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--gap', type=int, help='gap: Generator train GAP times, discriminator train once')
parser.add_argument('--model', type=str, default="fnn", help='manual seed')
parser.add_argument('--file', type=str, default="processed_file_{}_train_test_20.pkl", help='which file(data split) to perform')

args = parser.parse_args()
device = torch.device("cuda:0")

# seed
if args.seed is None:
    args.seed = random.randint(1, 10000)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True

print(args.model)


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
'''Add normalization (layer norm or batch norm) later'''
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
        self.tanh = nn.Tanh()
    def forward(self, x):
        # x: [bs, seq, feature_dim]
        x = self.net(x)
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        out = self.tanh(self.fc(x))
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('LayerNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



def main():
    training_set = TimeSeriesDataset(root_dir=args.data_path, file_name=args.file.format(args.task), train=True)
    test_set = TimeSeriesDataset(root_dir=args.data_path, file_name=args.file.format(args.task), train=False)
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    
    seq_len = 10 
    feature_dim = 160
    d_out = 50 if args.task == "3A" else 65

    # Model architecture

    # encoder: feature extractor
    # CNet:    Classifier
    # DNet:    Discriminator
    # GNet:    Generator (Adaptor)
    encoder = ComplexTransformer(layers=1, 
                               time_step=seq_len, 
                               input_dim=feature_dim, 
                               hidden_size=512, 
                               output_dim=512, 
                               num_heads=8,
                               out_dropout=0.5)
    #encoder.load_state_dict(torch.load("encoder_{0}{1}.pth".format(args.task, args.file)))
    encoder.to(device)
    if args.model == "fnn":
        pass
    elif args.model == "tvd":
        # Generator: IS ADAPTOR
        feature_dim_joint = 2 * feature_dim
        DNet = Discriminator(feature_dim=feature_dim_joint, d_out=d_out).to(device)
        GNet = Generator(feature_dim=feature_dim_joint).to(device)
        DNet.apply(weights_init)
        GNet.apply(weights_init)
        CNet = FNN(d_in=feature_dim * 2 * seq_len, d_h=500, d_out=d_out, dp=0.5).to(device)
        #CNet.load_state_dict(torch.load("CNet_{0}{1}.pth".format(args.task, args.file)))
        #DNet.load_state_dict(torch.load("DNet_{0}{1}.pth".format(args.task, args.file)))
        #GNet.load_state_dict(torch.load("GNet_{0}{1}.pth".format(args.task, args.file)))

        optimizerD = torch.optim.Adam(DNet.parameters(), lr=args.lr_gan)
        optimizerG = torch.optim.Adam(GNet.parameters(), lr=args.lr_gan)
        schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.1)
        schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.1)

        real_label = 0.99 # target domain
        fake_label = 0.01 # source domain

        # Generator: target to source

        # a joint dataset for training GAN
        joint_set = TimeSeriesDatasetConcat(root_dir=args.data_path, file_name=args.file.format(args.task), seed=args.seed)
        joint_loader = DataLoader(joint_set, batch_size=args.batch_size, shuffle=True)
        total_error_D = total_error_G = 0

        real_mean = joint_set.tr_data_mean
        fake_mean = joint_set.te_data_mean
        real_std = joint_set.tr_data_std
        fake_std = joint_set.te_data_std
        for epoch in range(args.epochs_gan):
            # fake: target domain
            # real: source domain
            total_error_D = total_error_G = 0
            for batch_id, (x_real, x_fake) in enumerate(joint_loader):
                batch_size = x_fake.shape[0]
                x_fake = x_fake.reshape(batch_size, seq_len, feature_dim_joint)
                x_real = x_real.reshape(batch_size, seq_len, feature_dim_joint)

                # Data Normalization
                x_fake = (x_fake - fake_mean) / fake_std
                x_real = (x_real - real_mean) / real_std

                """Update D Net"""
                # train with real(source domain)
                DNet.zero_grad()
                real_data = x_real.to(device).float()
                label = torch.full((batch_size,), real_label, device=device)
                output = DNet(real_data).view(-1)
                #print("1", output.mean().item())
                #print(output.mean().item())
                errD_real = -(output).mean()
                errD_real.backward()

                # train with fake(target domain)
                fake_data = x_fake.to(device).float()
                fake = GNet(fake_data)
                #print(fake)
                label.fill_(fake_label)
                output = DNet(fake.detach()).view(-1)
                #print("-1", output.mean().item())
                errD_fake = (output).mean()
                errD_fake.backward()
                total_error_D += (errD_real + errD_fake).item()
                if batch_id % args.gap == 0:
                    optimizerD.step()

                """Update G Network"""
                GNet.zero_grad()
                label.fill_(real_label) # fake labels are real for generator cost
                output = DNet(fake).view(-1)
                #print("1", output.mean().item())
                #print()
                #print(output.mean().item())
                #print()
                errG = -(output).mean()
                errG.backward()
                optimizerG.step()
                total_error_G += errG.item()
            schedulerD.step()
            schedulerG.step()
            print(total_error_D + total_error_G, total_error_D, total_error_G)
        print("Gan trainig finished") 
    
    torch.save(DNet.state_dict(), "DNet_{0}{1}.pth".format(args.task, args.file))
    torch.save(GNet.state_dict(), "GNet_{0}{1}.pth".format(args.task, args.file))
    
    params = list(encoder.parameters()) + list(CNet.parameters())

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(params, lr=args.lr_clf)

    scheduler = ReduceLROnPlateau(optimizer, 'min')
    # Encoding by complex transformer
    x_mean_tr = training_set.data_mean
    x_std_tr = training_set.data_std

    x_mean_te = test_set.data_mean
    x_std_te = test_set.data_std
    best_acc_train = best_acc_test = 0
    for epoch in range(args.epochs):
        """Training"""
        correct_train = 0
        total_bs_train = 0 # total batch size
        train_loss = 0
        for batch_id, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            #normalize data
            x = (x - x_mean_tr) / x_std_tr
            # take the real and imaginary part out
            real = x[:,:,0].reshape(batch_size, seq_len, feature_dim).float().to(device)
            imag = x[:,:,1].reshape(batch_size, seq_len, feature_dim).float().to(device)
            real, imag = encoder(real, imag)
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

        """Testing"""
        correct_test = 0
        total_bs_test = 0
        test_loss = 0
        for batch_id, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            x = (x - x_mean_te) / x_std_te
            batch_size = x.shape[0]

            CNet.eval()
            encoder.eval()

            with torch.no_grad():
                # feed into Generator (target to source)
                x = x.reshape(batch_size, seq_len, feature_dim * 2).float()
                GNet.eval()
                #print(x.mean().item(), x.std().item())
                x = GNet(x)
                #print(x.mean().item(), x.std().item())
                #print()
                x = x.reshape(batch_size, seq_len * feature_dim, 2) # reshape back

                real = x[:,:,0].reshape(x.shape[0], seq_len, feature_dim).float().to(device)
                imag = x[:,:,1].reshape(x.shape[0], seq_len, feature_dim).float().to(device)
                real, imag = encoder(real, imag)
                pred = CNet(torch.cat((real, imag), -1).reshape(x.shape[0], -1)) 
                #pred = CNet(x.reshape(x.shape[0], -1))
                #print(pred)
                loss = criterion(pred, y.argmax(-1))
                correct_test += (pred.argmax(-1) == y.argmax(-1)).sum().item()
                total_bs_test += y.shape[0]
                test_loss += loss.item() * x.shape[0]
        test_acc = float(correct_test) / total_bs_test
        test_loss = test_loss / total_bs_test

        best_acc_test = max(best_acc_test, test_acc)


        print("train loss", train_loss, "test_loss", test_loss)
        print("train acc", train_acc, "test_acc", test_acc)

    torch.save(CNet.state_dict(), "CNet_{0}{1}.pth".format(args.task, args.file))
    torch.save(encoder.state_dict(), "encoder_{0}{1}.pth".format(args.task, args.file))
    print(best_acc_train, best_acc_test)
    print(args.file)
    print(args)
    print("seed", args.seed)
    print("{0:20} | {1:9.3f}".format(args.file, best_acc_test))
        #scheduler.step(test_loss)
if __name__ == "__main__":
    main()

