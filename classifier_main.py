import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from dataset import TimeSeriesChunkDataset
from gan import Generator
from classifier import Classifier
from my_utils import classifier_inference, get_accuracy

parser = argparse.ArgumentParser(description='Time series adaptation')
parser.add_argument("--data_path", type=str, default="/projects/rsalakhugroup/complex/wifi/", help="dataset path")
parser.add_argument("--task", type=str, help='3A or 3E')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=8, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--context', type=int, default=40, help='context of time series data')
parser.add_argument('--output', type=str, default="./train_related", help="output path")
parser.add_argument('--save_period', type=int, default=5, help="how many epochs to save model")
parser.add_argument('--model_name', type=str, default="classifer", help="model name")

args = parser.parse_args()

args.task = "processed_file_3Av2.pkl" if args.task == "3A" else "processed_file_3E.pkl"
args.data_path = args.data_path + args.task

class SourceDomainClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(SourceDomainClassifier, self).__init__()
        self.net = nn.Sequential(
            Generator(**(kwargs["generator"])),
            nn.Flatten(),
            Classifier(**(kwargs['classifier']))
        )
        
    def forward(self, x):
        x = self.net(x)
        return x

def train(model, train_dataloader, vali_dataloader, lr, n_epochs, device, args):
    optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)
    loss_fn = nn.CrossEntropyLoss()
    train_loss_ = []
    train_acc_ = []
    vali_acc_ = []
    
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        train_acc = 0.0

        num_data = 0.0
        num_batches = len(train_dataloader)
        for batches, (x_batch, y_batch) in tqdm(enumerate(train_dataloader), total=num_batches):
            model.train()
            num_data += y_batch.shape[0]
            x_batch = x_batch.to(device)
            y_batch = y_batch.long().to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch.squeeze_())
            acc = get_accuracy(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc.item()
            
            # if batches > 1 and batches % int(num_batches/3) == 0:
            #     vali_acc = classifier_inference(model, vali_dataloader, device)
            #     print("validation_acc: ", vali_acc)

        vali_acc = classifier_inference(model, vali_dataloader, device)
        train_loss_.append(train_loss/num_data)
        train_acc_.append(train_acc/num_data)
        vali_acc_.append(vali_acc)
        # scheduler.step(vali_acc)

        if epoch % args.save_period == 0 and epoch > 1:
          output_path = args.output + "/" + args.model_name + "/"
          name = output_path + "model_" + str(epoch) + ".t7"   
          np.save(output_path + "train_loss_.npy",train_loss_)
          np.save(output_path + "train_acc_.npy",train_acc_)
          np.save(output_path + "vali_acc_.npy",vali_acc_)
          torch.save(model.state_dict(), name)

        print("epoch {}: train_loss: {}, train_acc: {}, vali_acc: {}".format(epoch, train_loss/num_data, train_acc/num_data, vali_acc))


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_dict = np.load(args.data_path, allow_pickle=True)
    
    # split train data and validation data
    np.random.seed(seed=0)
    indices = np.random.permutation(data_dict['tr_data'].shape[0])
    train_x = data_dict['tr_data'][indices[:int(indices.shape[0]*0.95)],:,:].astype("float32")
    train_y = data_dict['tr_lbl'][indices[:int(indices.shape[0]*0.95)],:].astype("float32")
    vali_x = data_dict['tr_data'][indices[int(indices.shape[0]*0.05):],:,:].astype("float32")
    vali_y = data_dict['tr_lbl'][indices[int(indices.shape[0]*0.05):],:].astype("float32")

    # build dataset
    train_dataset = TimeSeriesChunkDataset(train_x, train_y, args.context)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12)
    vali_dataset = TimeSeriesChunkDataset(vali_x, vali_y, args.context)
    vali_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12)
    
    # TODO: change it to read json
    model_args = {
        'classifier': {
            'layers_size': [args.context*2, args.context*3, args.context*3, args.context*2, args.context*2], 
            'dim_out': 50 if args.task == '3A' else 65
        },
        'generator':{
            'transformer_layer': {},
            'transformer': {'num_layers':2}
        }
    }
    
    # build model
    source_classifier = SourceDomainClassifier(**model_args)
    source_classifier.to(device)
    # source_classifier.apply(init_weights)
    
    # train
    train(source_classifier, train_dataloader, vali_dataloader, args.lr, args.epochs, device, args)
    
