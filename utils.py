import os
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def denorm(x):
    #out = (x + 1) / 2
    return x.clamp_(0, 1)