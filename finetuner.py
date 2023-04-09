import os
import time
import torch
import datetime

import torch.nn as nn

from model import WaveDecoder, WaveEncoder

from logger import Logger
from utils import denorm


class Finetuner():
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
        self.pretrained_path = config.pretrained_path
        self.model_save_path = config.model_save_path

        self.build_models()

    def build_models(self):
        self.wave_encoder = WaveEncoder(self.block).to(self.device)
        self.wave_decoder = WaveDecoder(self.block).to(self.device)
        # Warmstart
        fname = os.path.join(self.pretrained_path, 'conv{}_1.pth'.format(self.block))
        self.wave_encoder.load_state_dict(
            torch.load(fname),
            strict=False
        )
        print('weights from {} are loaded'.format(fname))
        fname = os.path.join(self.pretrained_path, 'dec{}_1.pth'.format(self.block))
        self.wave_decoder.load_state_dict(
            torch.load(fname),
            strict=False
        )
        print('weights from {} are loaded'.format(fname))

        self.MSE_loss = nn.MSELoss().to(self.device)

        self.enc_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.wave_encoder.parameters()),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.dec_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.wave_decoder.parameters()),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )

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
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )

    def train(self):
        level = 1

        fixed_sample = next(iter(self.loader))
        fixed_sample = fixed_sample.to(self.device)
        self.logger.image_summary(denorm(fixed_sample),
                                  opts=dict(title='sample', caption='sample'))

        with torch.no_grad():
            wav_decomp = self.wave_encoder(fixed_sample, level)
            feature = torch.cat(wav_decomp, dim=1)
            recon_image = self.wave_decoder(feature, level)
            self.logger.image_summary(denorm(recon_image),
                                      opts=dict(title='recon_start', caption='recon_start')
                                      )
        start_time = time.time()
        print('training start')
        for epoch in range(self.epoch):
            for num, real_image in enumerate(self.loader):
                real_image = real_image.to(self.device)

                # loss: recon image
                with torch.no_grad():
                    wave_decomp = self.wave_encoder(real_image, level)
                    feature = torch.cat(wave_decomp, dim=1)
                recon_image = self.wave_decoder(feature, level)
                recon_loss = self.MSE_loss(recon_image, real_image) * self.recon_weight

                # loss: recon feature
                recon_wave_decomp = self.wave_encoder(recon_image, level)
                recon_feature = torch.cat(recon_wave_decomp, dim=1)
                feature_loss = self.MSE_loss(recon_feature, feature.detach()) * self.feature_weight

                # sum loss and update
                loss = recon_loss + feature_loss
                self.reset_grad()
                loss.backward()
                # self.enc_optim.step()
                self.dec_optim.step()

                if (num + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed: [{}], Level: [{}], Epoch: [{}/{}], Iter: [{}/{}] \
                        Recon Loss: {:.4f}, Feature loss: {:.4f}".format(elapsed, level, epoch + 1, self.epoch, num + 1, len(self.loader),
                                                                         recon_loss.item(), feature_loss.item()))
                    info = {
                        'loss/recon': recon_loss.item(),
                        'loss/feature': feature_loss.item(),
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, num + 1, scope=locals())
                    with torch.no_grad():
                        wav_decomp = self.wave_encoder(fixed_sample, level)
                        feature = torch.cat(wav_decomp, dim=1)
                        recon_image = self.wave_decoder(feature, level)
                        self.logger.image_summary(denorm(recon_image),
                                                  opts=dict(title='recon{}_{}_{}'.format(level, epoch+1, num+1),
                                                            caption='recon{}_{}_{}'.format(level, epoch+1, num+1)
                                                            )
                                                  )
                if (num + 1) % 500 == 0:
                    torch.save(self.wave_encoder.state_dict(),
                               os.path.join(self.model_save_path, "e{}_i{}_conv{}_1.pth".format(epoch+1, num+1, self.block))
                               )
                    torch.save(self.wave_decoder.state_dict(),
                               os.path.join(self.model_save_path, "e{}_i{}_dec{}_1.pth".format(epoch+1, num+1, self.block))
                               )

        torch.save(self.wave_encoder.state_dict(),
                   os.path.join(self.model_save_path, "conv{}_1.pth".format(self.block))
                   )
        torch.save(self.wave_decoder.state_dict(),
                   os.path.join(self.model_save_path, "dec{}_1.pth".format(self.block))
                   )
