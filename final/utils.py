import torch
import os
import sys
import logging
from torch import nn

def encoder_inference(encoder, encoder_MLP, x, seq_len=10, feature_dim=160):
    real = x[:,:,0].reshape(x.size(0), seq_len, feature_dim).float()
    imag = x[:,:,1].reshape(x.size(0), seq_len, feature_dim).float()
    real, imag = encoder(real, imag)
    cat_embedding = torch.cat((real[:,-1,:], imag[:,-1,:]), -1).reshape(x.shape[0], -1)
    cat_embedding = encoder_MLP(cat_embedding)
    return cat_embedding

def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LayerNorm:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

def get_logger(save_folder, args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if os.path.isfile(os.path.join(save_folder, 'logfile.log')):
        os.remove(os.path.join(save_folder, 'logfile.log'))
        
    file_log_handler = logging.FileHandler(os.path.join(save_folder, 'logfile.log'))
    logger.addHandler(file_log_handler)

    stdout_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_log_handler)

    attrs = vars(args)
    for item in attrs.items():
        logger.info("%s: %s"%item)
    logger.info("Saved in %s"%save_folder)

    return logger