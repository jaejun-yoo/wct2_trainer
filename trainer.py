import os
import time
import torch
import datetime

import torch.nn as nn
from torchvision.utils import save_image

from model import VGGDecoder, VGGEncoder
from model import WaveDecoder, WaveEncoder

from logger import Logger
from utils import denorm

import nsml

class Trainer():
    def __init__(self, loader, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger = Logger()
        self.loader = loader

        self.epoch = config.epoch

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.feature_weight = config.feature_weight
        self.recon_weight = config.recon_weight

        self.block = config.block
        self.model_save_path = config.model_save_path

        self.build_models()

    def build_models(self):
        self.encoder = VGGEncoder(self.block).to(self.device)
        self.decoder = VGGDecoder(self.block).to(self.device)

        self.encoder.load_state_dict(
            torch.load("./lua_models/vgg_normalised_conv{}.pth".format(self.block))
        )
        self.decoder.load_state_dict(
            torch.load("./lua_models/feature_invertor_conv{}.pth".format(self.block))
        )
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.wave_encoder = WaveEncoder(self.block).to(self.device)
        self.wave_decoder = WaveDecoder(self.block).to(self.device)
        # Warmstart
        self.wave_encoder.load_state_dict(
            torch.load("./lua_models/vgg_normalised_conv{}.pth".format(self.block)),
            strict=False
        )
        self.wave_decoder.load_state_dict(
            torch.load("./lua_models/feature_invertor_conv{}.pth".format(self.block)),
            strict=False
        )
        self.MSE_loss = nn.MSELoss().to(self.device)

        self.freeze(-1)
    
    def reset_grad(self):
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        
    def freeze(self, level):
        self.wave_encoder.freeze(level)
        self.wave_decoder.freeze(level)
        self.enc_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.wave_encoder.parameters()),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.dec_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.wave_decoder.parameters()),
            lr = self.lr,
            betas=(self.beta1, self.beta2)
        )
        
    def train(self):
        fixed_sample = next(iter(self.loader))
        fixed_sample = fixed_sample.to(self.device)
        self.logger.image_summary(denorm(fixed_sample),
                opts=dict(title='sample', caption='sample'))

        start_time = time.time()

        for level in range(self.block, 1, -1):
            for epoch in range(self.epoch):
                for num, real_image in enumerate(self.loader):
                    real_image = real_image.to(self.device)

                    feature, unpool_list = self.encoder(real_image, level-1)
                    wav_decomp = self.wave_encoder(feature, level)

                    decode_feature = self.wave_decoder(torch.cat(wav_decomp, dim=1), level)
                    recon_image = self.decoder(decode_feature, level-1, *unpool_list)

                    recon_loss = self.MSE_loss(recon_image, real_image) * self.recon_weight
                    recon_feature, _ = self.encoder(recon_image, level-1)

                    feature_loss = self.MSE_loss(recon_feature, feature) * self.feature_weight

                    loss = recon_loss + feature_loss
                    self.reset_grad()
                    loss.backward()
                    self.enc_optim.step()
                    self.dec_optim.step()

                    if (num + 1) % 100 == 0:
                        elapsed = time.time() - start_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print("Elapsed: [{}], Level: [{}], Epoch: [{}/{}], Iter: [{}/{}] \
                            Recon Loss: {:.4f}, Feature loss: {:.4f}".format(elapsed, level, epoch + 1, self.epoch, num + 1, len(self.loader),
                                recon_loss.item(), feature_loss.item()))
                        info = {
                            "Level": level,
                            "Loss": loss.item()
                        }
                        for tag, value in info.items():
                            self.logger.scalar_summary(tag, value, (len(self.loader) * epoch) + num + 1)
                        
                        if (num + 1) % 500 == 0:
                            with torch.no_grad():
                                feature, unpool_list = self.encoder(fixed_sample, level-1)
                                wav_decomp = self.wave_encoder(feature, level)

                                decode_feature = self.wave_decoder(torch.cat(wav_decomp, dim=1), level)
                                recon_image = self.decoder(decode_feature, level-1, *unpool_list)
                                self.logger.image_summary(denorm(recon_image),
                                            opts=dict(title='transferd{}_{}_{}'.format(level, epoch+1, num+1),
                                                        caption='transfered{}_{}_{}'.format(level, epoch+1, num+1)
                                                )
                                )
                            torch.save(self.wave_encoder.state_dict(),
                                        os.path.join(self.model_save_path, "conv{}_1_at_{}.pth".format(self.block, level))
                            )
                            torch.save(self.wave_decoder.state_dict(),
                                        os.path.join(self.model_save_path, "dec{}_1_at_{}.pth".format(self.block, level))
                            )
            
            torch.save(self.wave_encoder.state_dict(),
                        os.path.join(self.model_save_path, "conv{}_1_at_{}.pth".format(self.block, level))
            )
            torch.save(self.wave_decoder.state_dict(),
                        os.path.join(self.model_save_path, "dec{}_1_at_{}.pth".format(self.block, level))
            )
            self.freeze(level)
