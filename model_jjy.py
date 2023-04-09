import torch
import torch.nn as nn
import numpy as np

def get_fixed(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    template = np.eye(4)
    fixed_LL = template[0].reshape(2,2)
    fixed_LH = template[1].reshape(2,2)
    fixed_HL = template[2].reshape(2,2)
    fixed_HH = template[3].reshape(2,2)
    
    filter_LL = torch.from_numpy(fixed_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(fixed_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(fixed_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(fixed_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels, 
                    kernel_size=2, stride=2, padding=0, bias=False, 
                    groups=in_channels)
    LH = net(in_channels, in_channels, 
                    kernel_size=2, stride=2, padding=0, bias=False, 
                    groups=in_channels)
    HL = net(in_channels, in_channels, 
                    kernel_size=2, stride=2, padding=0, bias=False, 
                    groups=in_channels)
    HH = net(in_channels, in_channels, 
                    kernel_size=2, stride=2, padding=0, bias=False, 
                    groups=in_channels)  

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    
    return LL, LH, HL, HH


class Fixed_pool(nn.Module):
    def __init__(self, in_channels):
        super(Fixed_pool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_fixed(in_channels)
    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

    
class FixedP_recon(nn.Module):
    def __init__(self, in_channels):
        super(FixedP_recon, self).__init__()
        self.in_channels = in_channels

        self.LL, self.LH, self.HL, self.HH = get_fixed(self.in_channels, pool=False)
    
    def forward(self, LL, LH, HL, HH):
        return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
    
def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels, 
                    kernel_size=2, stride=2, padding=0, bias=False, 
                    groups=in_channels)
    LH = net(in_channels, in_channels, 
                    kernel_size=2, stride=2, padding=0, bias=False, 
                    groups=in_channels)
    HL = net(in_channels, in_channels, 
                    kernel_size=2, stride=2, padding=0, bias=False, 
                    groups=in_channels)
    HH = net(in_channels, in_channels, 
                    kernel_size=2, stride=2, padding=0, bias=False, 
                    groups=in_channels)  

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    
    return LL, LH, HL, HH

    
class Wav_pool(nn.Module):
    def __init__(self, in_channels):
        super(Wav_pool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

    
class Wav_recon(nn.Module):
    def __init__(self, in_channels):
        super(Wav_recon, self).__init__()
        self.in_channels = in_channels

        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)
    
    def forward(self, LL, LH, HL, HH):
        return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)

    
class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

    def forward(self, x):
        skips = {}
        unpool_idxs = {}
        pool_sizes = {}
        
        out = self.conv0(x)
        out = self.relu(self.conv1_1(self.pad(out)))
        skips['conv1_1'] = out
        out = self.relu(self.conv1_2(self.pad(out)))        
        
        pool1, pool1_idx = self.maxpool(out)
        unpool_idxs['pool1'] = pool1_idx
        pool_sizes['pool1'] = pool1.size()
        
        out = self.relu(self.conv2_1(self.pad(pool1)))
        skips['conv2_1'] = out
        out = self.relu(self.conv2_2(self.pad(out)))
        
        
        pool2, pool2_idx = self.maxpool(out)
        unpool_idxs['pool2'] = pool2_idx
        pool_sizes['pool2'] = pool2.size()
        
        out = self.relu(self.conv3_1(self.pad(pool2)))
        skips['conv3_1'] = out
        out = self.relu(self.conv3_2(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_4(self.pad(out)))
        
                
        pool3, pool3_idx = self.maxpool(out)
        unpool_idxs['pool3'] = pool3_idx
        pool_sizes['pool3'] = pool3.size()
        
        out = self.relu(self.conv4_1(self.pad(pool3)))

        return out, skips, unpool_idxs, pool_sizes

    
class VGGDecoder(nn.Module):
    def __init__(self):
        super(VGGDecoder, self).__init__()


        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
        
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
        
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
        
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)
    
    def forward(self, x, unpool_idxs, pool_sizes):
        out = x
        out = self.relu(self.conv4_1(self.pad(out)))
        
        out = self.unpool(out, unpool_idxs['pool3'], output_size=pool_sizes['pool3'])
        
        out = self.relu(self.conv3_4(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_2(self.pad(out)))
        out = self.relu(self.conv3_1(self.pad(out)))

        out = self.unpool(out, unpool_idxs['pool2'], output_size=pool_sizes['pool2'])

        out = self.relu(self.conv2_2(self.pad(out)))
        out = self.relu(self.conv2_1(self.pad(out)))
            
        out = self.unpool(out, unpool_idxs['pool1'], output_size=pool_sizes['pool1'])

        out = self.relu(self.conv1_2(self.pad(out)))
        out = self.conv1_1(self.pad(out))

        return out

    
    
class FixedP_Encoder(nn.Module):
    def __init__(self):
        super(FixedP_Encoder, self).__init__()        
        
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = Fixed_pool(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = Fixed_pool(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = Fixed_pool(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        
    def forward(self, x):
        out = x
        skips = {}
        
        out = self.conv0(out)
        out = self.relu(self.conv1_1(self.pad(out)))
        skips['conv1_1'] = out
        out = self.relu(self.conv1_2(self.pad(out)))        
        
        
        LL, LH, HL, HH = self.pool1(out)
        skips['pool1'] = [LH, HL, HH]
        out = self.relu(self.conv2_1(self.pad(LL)))
        skips['conv2_1'] = out
        out = self.relu(self.conv2_2(self.pad(out)))
        
        
        LL, LH, HL, HH = self.pool2(out)
        skips['pool2'] = [LH, HL, HH]
        out = self.relu(self.conv3_1(self.pad(LL)))
        skips['conv3_1'] = out
        out = self.relu(self.conv3_2(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_4(self.pad(out)))
        
        
        LL, LH, HL, HH = self.pool3(out)
        skips['pool3'] = [LH, HL, HH]
        
        out = self.relu(self.conv4_1(self.pad(LL)))
        
        return out, skips
    
    
class FixedP_Decoder(nn.Module):
    def __init__(self):
        super(FixedP_Decoder, self).__init__()
        
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
        
        self.recon_block3 = FixedP_recon(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        self.recon_block2 = FixedP_recon(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.recon_block1 = FixedP_recon(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, skips):        
        out = x
        out = self.relu(self.conv4_1(self.pad(out)))
        
        LH, HL, HH = skips['pool3']
        out = self.recon_block3(out, LH, HL, HH)        
        out = self.relu(self.conv3_4(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_2(self.pad(out)))
        skips['conv3_2_fixedP_recon'] = out
        out = self.relu(self.conv3_1(self.pad(out)))
        
        LH, HL, HH = skips['pool2']
        out = self.recon_block2(out, LH, HL, HH)        
        out = self.relu(self.conv2_2(self.pad(out)))
        skips['conv2_2_fixedP_recon'] = out
        out = self.relu(self.conv2_1(self.pad(out)))

        LH, HL, HH = skips['pool1']
        out = self.recon_block1(out, LH, HL, HH)        
        out = self.relu(self.conv1_2(self.pad(out)))
        skips['conv1_2_fixedP_recon'] = out
        out = self.conv1_1(self.pad(out))
        
        return out, skips
    
    
class WaveEncoder(nn.Module):
    def __init__(self):
        super(WaveEncoder, self).__init__()        
        
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = Wav_pool(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = Wav_pool(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = Wav_pool(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        
    def forward(self, x):
        out = x
        skips = {}
        
        out = self.conv0(out)
        out = self.relu(self.conv1_1(self.pad(out)))
        skips['conv1_1'] = out
        out = self.relu(self.conv1_2(self.pad(out)))        
        
        
        LL, LH, HL, HH = self.pool1(out)
        skips['pool1'] = [LH, HL, HH]
        out = self.relu(self.conv2_1(self.pad(LL)))
        skips['conv2_1'] = out
        out = self.relu(self.conv2_2(self.pad(out)))
        
        
        LL, LH, HL, HH = self.pool2(out)
        skips['pool2'] = [LH, HL, HH]
        out = self.relu(self.conv3_1(self.pad(LL)))
        skips['conv3_1'] = out
        out = self.relu(self.conv3_2(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_4(self.pad(out)))
        
        
        LL, LH, HL, HH = self.pool3(out)
        skips['pool3'] = [LH, HL, HH]
        
        out = self.relu(self.conv4_1(self.pad(LL)))
        
        return out, skips
    
    
class WaveDecoder(nn.Module):
    def __init__(self):
        super(WaveDecoder, self).__init__()
        
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
        
        self.recon_block3 = Wav_recon(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        self.recon_block2 = Wav_recon(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.recon_block1 = Wav_recon(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, skips):        
        out = x
        out = self.relu(self.conv4_1(self.pad(out)))
        
        LH, HL, HH = skips['pool3']
        out = self.recon_block3(out, LH, HL, HH)        
        out = self.relu(self.conv3_4(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_2(self.pad(out)))
        skips['conv3_2_wavrecon'] = out
        out = self.relu(self.conv3_1(self.pad(out)))
        
        LH, HL, HH = skips['pool2']
        out = self.recon_block2(out, LH, HL, HH)        
        out = self.relu(self.conv2_2(self.pad(out)))
        skips['conv2_2_wavrecon'] = out
        out = self.relu(self.conv2_1(self.pad(out)))

        LH, HL, HH = skips['pool1']
        out = self.recon_block1(out, LH, HL, HH)        
        out = self.relu(self.conv1_2(self.pad(out)))
        skips['conv1_2_wavrecon'] = out
        out = self.conv1_1(self.pad(out))
        
        return out, skips