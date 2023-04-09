'''FrameNet based on AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import numpy as np
__all__ = ['framenet']


def wav_pool(in_channels, out_channels, kernel_size=2, stride = 2):
    "wav_pooling without padding"
    Harr_wav_L = 1/np.sqrt(2) * np.ones((1,2))
    Harr_wav_H = 1/np.sqrt(2) * np.ones((1,2))
    Harr_wav_H[0,0] = Harr_wav_H[0,0]*-1

    Harr_wav_LL = np.transpose(Harr_wav_L)*Harr_wav_L
    Harr_wav_LH = np.transpose(Harr_wav_L)*Harr_wav_H
    Harr_wav_HL = np.transpose(Harr_wav_H)*Harr_wav_L
    Harr_wav_HH = np.transpose(Harr_wav_H)*Harr_wav_H

    fixed_wav_filters = np.stack((Harr_wav_LL , Harr_wav_LH , Harr_wav_HL, Harr_wav_HH  ), axis = -1 ).transpose((2,0,1))
    fixed_wav_filters = torch.from_numpy(np.expand_dims(fixed_wav_filters, axis=1))    
    
    out = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=0, bias=False, groups=in_channels)
    
    # Set Conv weight to wav_filters
    out.weight.data = fixed_wav_filters.float().repeat(in_channels,1,1,1)
    out.weight.requires_grad = False
    return out

class FrameNet(nn.Module):

    def __init__(self, num_classes=10):
        super(FrameNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=4, padding=5),
            nn.ReLU(inplace=True),
            wav_pool(32, 32*4, kernel_size=3, stride=2),
            nn.Conv2d(32*4, 128, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            wav_pool(128, 128*4, kernel_size=3, stride=2),
            nn.Conv2d(128*4, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            wav_pool(256, 256*4, kernel_size=3, stride=2),
        )
        self.classifier = nn.Linear(256*4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def framenet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = FrameNet(**kwargs)
    #model.features[2].weight.requires_grad = False
    #model.features[5].weight.requires_grad = False
    #model.features[12].weight.requires_grad = False    
    return model
