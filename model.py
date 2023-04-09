import torch
import torch.nn as nn
import numpy as np

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
    
    def forward(self, x):
        channel_wise = x.split(self.in_channels, dim=1)
        return self.LL(channel_wise[0]) + self.LH(channel_wise[1]) + self.HL(channel_wise[2]) + self.HH(channel_wise[3])

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.pool = Wav_pool(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        LL, LH, HL, HH = self.pool(x)
        LL = self.relu(self.conv1(self.pad(LL)))
        LH = self.relu(self.conv2(self.pad(LH)))
        HL = self.relu(self.conv3(self.pad(HL)))
        HH = self.relu(self.conv4(self.pad(HH)))
        return LL, LH, HL, HH

class ReconBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReconBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels * 4, out_channels * 4, kernel_size=3, stride=1, padding=0)
        self.recon = Wav_recon(out_channels)

    def forward(self, x):
        return self.recon(self.relu(self.conv(self.pad(x))))

class VGGEncoder(nn.Module):
    def __init__(self, level):
        super(VGGEncoder, self).__init__()
        self.level = level
        
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        # 224 x 224
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        # 224 x 224
        
        if level < 2: return
        
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        # 224 x 224
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        # 112 x 112
        
        if level < 3: return
        
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        # 112 x 112
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        # 56 x 56
        
        if level < 4: return
        
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        # 56 x 56
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)     
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

        if level < 5: return

        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 0)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 0)

    def forward(self, x, end_level):
        out = self.conv0(x)
        out = self.relu(self.conv1_1(self.pad(out)))
        
        if self.level == 1 or end_level == 0:
            return out, [None, None, None, None]
    
        pool1 = self.relu(self.conv1_2(self.pad(out)))
        
        if self.level == 2 or end_level == 1: # Conv2_1
            return pool1, [None, None, None, None]

        out, pool1_idx = self.maxpool1(pool1)
        out = self.relu(self.conv2_1(self.pad(out)))
        pool2 = self.relu(self.conv2_2(self.pad(out)))

        if self.level == 3 or end_level == 2: # Conv3_1
            return pool2, [pool1_idx, pool1.size(), None, None]

        out, pool2_idx = self.maxpool2(pool2)
        out = self.relu(self.conv3_1(self.pad(out)))
        out = self.relu(self.conv3_2(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        pool3 = self.relu(self.conv3_4(self.pad(out)))

        if self.level == 4 or end_level == 3:
            return pool3, [pool1_idx, pool1.size(), pool2_idx, pool2.size()]

        out, _ = self.maxpool3(pool3)
        out = self.relu(self.conv4_1(self.pad(out)))
        out = self.relu(self.conv4_2(self.pad(out)))
        out = self.relu(self.conv4_3(self.pad(out)))
        pool4 = self.relu(self.conv4_4(self.pad(out)))

        # self.level == 5 or end_level == 4
        return pool4, [pool1_idx, pool1.size(), pool2_idx, pool2.size()]

class VGGDecoder(nn.Module):
    def __init__(self, level):
        super(VGGDecoder, self).__init__()
        self.level = level

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        if level > 4:
            self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 0)
            
            self.unpool4 = nn.Upsample(scale_factor=2)

            self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 0)
            self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 0)
            self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 0)
        if level > 3:
            self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
            # 28 x 28
            if self.level != 5:
                self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            else:
                self.unpool3 = nn.Upsample(scale_factor=2)
            # 56 x 56
            
            self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
            self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
            self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
            # 56 x 56
        if level > 2:
            self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
            # 56 x 56
            if self.level != 5:
                self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            else:
                self.unpool2 = nn.Upsample(scale_factor=2)
            self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
            # 112 x 112
        if level > 1:
            self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
            # 112 x 112
            if self.level != 5:
                self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            else:
                self.unpool1 = nn.Upsample(scale_factor=2)
            self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
            # 224 x 224
        
        if level > 0:
            self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)
    
    def forward(self, x, start_level, pool1_idx=None, pool1_size=None, pool2_idx=None, pool2_size=None):
        out = x
        if start_level > 3:
            out = self.relu(self.conv4_4(self.pad(out)))
            out = self.relu(self.conv4_3(self.pad(out)))
            out = self.relu(self.conv4_2(self.pad(out)))
            out = self.relu(self.conv4_1(self.pad(out)))
            if self.level == 5:
                out = self.unpool3(out)

        if start_level > 2: # pass conv4_1
            out = self.relu(self.conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            out = self.relu(self.conv3_2(self.pad(out)))
            out = self.relu(self.conv3_1(self.pad(out)))
            if self.level != 5:
                out = self.unpool2(out, pool2_idx, output_size=pool2_size)
            else:
                out = self.unpool2(out)

        if start_level > 1:
            out = self.relu(self.conv2_2(self.pad(out)))
            out = self.relu(self.conv2_1(self.pad(out)))
            if self.level != 5:
                out = self.unpool1(out, pool1_idx, output_size=pool1_size)
            else:
                out = self.unpool1(out)

        if start_level > 0:
            out = self.relu(self.conv1_2(self.pad(out)))
            out = self.conv1_1(self.pad(out))
        
        return out

class WaveEncoder(nn.Module):
    def __init__(self, level):
        super(WaveEncoder, self).__init__()
        self.level = level
        
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)

        if self.level < 2: return # Conv1_1
        
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.wave_conv1 = ConvBlock(64, 128)
        # self.conv2_1

        if self.level < 3: return # Conv2_1

        self.conv2_2_2 = nn.Conv2d(128 * 4, 128, 3, 1, 0)
        self.wave_conv2 = ConvBlock(128, 256)
        # self.conv3_1

        if self.level < 4: return # Conv3_1

        self.conv3_2_2 = nn.Conv2d(256 * 4, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)

        self.wave_conv3 = ConvBlock(256, 512)
        # self.conv4_1

        if self.level < 5: return
        
        self.conv4_2_2 = nn.Conv2d(512 * 4, 512, 3, 1, 0)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 0)

        self.wave_conv4 = ConvBlock(512, 512)
        # self.conv5_1

    def forward(self, x, start_level):
        out = x
        if start_level < 2: # 1
            out = self.conv0(out)
            out = self.relu(self.conv1_1(self.pad(out)))
            if self.level == 1:
                return out

            out = self.relu(self.conv1_2(self.pad(out)))
        
        if start_level < 3: # 2 
            pool1 = self.wave_conv1(out)
            if self.level == 2:
                return pool1
            
            pool1 = torch.cat(pool1, dim=1)
            out = self.relu(self.conv2_2_2(self.pad(pool1)))

        if start_level < 4: # 3
            pool2 = self.wave_conv2(out)
            if self.level == 3:
                return pool2
            pool2 = torch.cat(pool2, dim=1)
            out = self.relu(self.conv3_2_2(self.pad(pool2)))
            out = self.relu(self.conv3_3(self.pad(out)))
            out = self.relu(self.conv3_4(self.pad(out)))

        if start_level < 5: # 4
            pool3 = self.wave_conv3(out)
            if self.level == 4:
                return pool3
            out = self.relu(self.conv4_2_2(self.pad(pool3)))
            out = self.relu(self.conv4_3(self.pad(out)))
            out = self.relu(self.conv4_4(self.pad(out)))
        
        if start_level < 6:
            pool4 = self.wave_conv4(out)
            return pool4
    
    def freeze(self, level):
        for param in self.conv0.parameters():
            param.requires_grad = False
        for param in self.conv1_1.parameters():
            param.requires_grad = False
        
        if level == 5:
            for param in self.wave_conv4.parameters():
                param.requires_grad = False
        elif level == 4:
            for param in self.wave_conv3.parameters():
                param.requires_grad = False
            if self.level > 4:
                for param in self.conv4_2_2.parameters():
                    param.requires_grad = False
                for param in self.conv4_3.parameters():
                    param.requires_grad = False
                for param in self.conv4_4.parameters():
                    param.requires_grad = False
        elif level == 3:
            for param in self.wave_conv2.parameters():
                param.requires_grad = False
            if self.level > 3:
                for param in self.conv3_2_2.parameters():
                    param.requires_grad = False
                for param in self.conv3_3.parameters():
                    param.requires_grad = False
                for param in self.conv3_4.parameters():
                    param.requires_grad = False
        elif level == 2:
            for param in self.wave_conv1.parameters():
                param.requires_grad = False
            if self.level > 2:
                for param in self.conv2_2_2.parameters():
                    param.requires_grad = False
        elif level == 1:
            if self.level > 1:
                for param in self.conv1_2.parameters():
                    param.requires_grad = False

class WaveDecoder(nn.Module):
    def __init__(self, level):
        super(WaveDecoder, self).__init__()
        self.level = level
        
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        if self.level > 4: # level 5
            self.recon_block4 = ReconBlock(512, 512)
            self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 0)
            self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 0)
            self.conv4_2_2 = nn.Conv2d(512, 512 * 4, 3, 1, 0)
        if self.level > 3: # level 4
            self.recon_block3 = ReconBlock(512, 256) # Conv4_1, unpool
            self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
            self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
            self.conv3_2_2 = nn.Conv2d(256, 256 * 4, 3, 1, 0)

        if self.level > 2: # level 3
            self.recon_block2 = ReconBlock(256, 128) # Conv3_1, unpool
            self.conv2_2_2 = nn.Conv2d(128, 128 * 4, 3, 1, 0)

        if self.level > 1: # level 2
            self.recon_block1 = ReconBlock(128, 64) # Conv2_1, unpool
            self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        
        if self.level > 0:
            self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, end_level):
        out = x
        if self.level > 4:
            out = self.recon_block4(out)
            if end_level == 5:
                return out
            out = self.relu(self.conv4_4(self.pad(out)))
            out = self.relu(self.conv4_3(self.pad(out)))
            out = self.relu(self.conv4_2_2(self.pad(out)))

        if self.level > 3:
            out = self.recon_block3(out)
            if end_level == 4:
                return out
            out = self.relu(self.conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            out = self.relu(self.conv3_2_2(self.pad(out)))
        
        if self.level > 2:
            out = self.recon_block2(out)
            if end_level == 3:
                return out
            out = self.relu(self.conv2_2_2(self.pad(out)))

        if self.level > 1:
            out = self.recon_block1(out)
            if end_level == 2:
                return out
            out = self.relu(self.conv1_2(self.pad(out)))
        
        if self.level > 0:
            out = self.conv1_1(self.pad(out))
            return out
    
    def freeze(self, level):
        for param in self.conv1_1.parameters():
            param.requires_grad = False
        if level == 5:
            for param in self.recon_block4.parameters():
                param.requires_grad = False
        elif level == 4:
            for param in self.recon_block3.parameters():
                param.requires_grad = False
            if self.level > 4:
                for param in self.conv4_2_2.parameters():
                    param.requires_grad = False
                for param in self.conv4_3.parameters():
                    param.requires_grad = False
                for param in self.conv4_4.parameters():
                    param.requires_grad = False
        elif level == 3:
            for param in self.recon_block2.parameters():
                param.requires_grad = False
            if self.level > 3:
                for param in self.conv3_2_2.parameters():
                    param.requires_grad = False
                for param in self.conv3_3.parameters():
                    param.requires_grad = False
                for param in self.conv3_4.parameters():
                    param.requires_grad = False
        elif level == 2:
            for param in self.recon_block1.parameters():
                param.requires_grad = False
            if self.level > 2:
                for param in self.conv2_2_2.parameters():
                    param.requires_grad = False
        elif level == 1:
            if self.level > 1:
                for param in self.conv1_2.parameters():
                    param.requires_grad = False