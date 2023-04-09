import os
import time
import torch
import datetime

import torch.nn as nn

from torchvision.utils import save_image

from model_jjy import VGGDecoder, VGGEncoder
from model_jjy import FixedP_Decoder, FixedP_Encoder

from logger import Logger
from utils import denorm

import nsml


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)        
    
    
class Trainer():
    def __init__(self, loader, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger = Logger()
        self.loader = loader

        self.epoch = config.epoch
        self.fixedP_enc_freeze = config.fixedP_enc_freeze
        self.loss_option = config.loss_option

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.feature_weight = config.feature_weight
        self.recon_weight = config.recon_weight

        self.model_save_path = config.model_save_path

        self.build_models()

    def build_models(self):
        self.encoder = VGGEncoder().to(self.device)
        self.decoder = VGGDecoder().to(self.device)

        self.encoder.load_state_dict(
            torch.load("./lua_models/vgg_normalised_conv{}.pth".format(4))
        )
        self.decoder.load_state_dict(
            torch.load("./lua_models/feature_invertor_conv{}.pth".format(4))
        )
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.fixedP_encoder = FixedP_Encoder().to(self.device)
        self.fixedP_decoder = FixedP_Decoder().to(self.device)
        # Warmstart
        self.fixedP_encoder.load_state_dict(
            torch.load("./lua_models/vgg_normalised_conv{}.pth".format(4)),
            strict=False
        )
        self.fixedP_decoder.load_state_dict(
            torch.load("./lua_models/feature_invertor_conv{}.pth".format(4)),
            strict=False
        )
            
        self.MSE_loss = nn.MSELoss().to(self.device)
        
        if self.fixedP_enc_freeze:
            for param in self.fixedP_encoder.parameters():
                param.requires_grad = False
        else:            
            self.enc_optim = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.fixedP_encoder.parameters()),
                lr=self.lr,
                betas=(self.beta1, self.beta2)
            )
            
        self.dec_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.fixedP_decoder.parameters()),
            lr = self.lr,
            betas=(self.beta1, self.beta2)
        )
    
    def reset_grad(self):
        if not self.fixedP_enc_freeze:
            self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()

    
    def train(self):
        fixed_sample = next(iter(self.loader))
        fixed_sample = fixed_sample.to(self.device)
        self.logger.image_summary(denorm(fixed_sample),
                opts=dict(title='sample', caption='sample'))

        start_time = time.time()
        with torch.no_grad():
            fixed_feature, fixed_skips_fixedP = self.fixedP_encoder(fixed_sample)
            fixed_recon_image, _ = self.fixedP_decoder(fixed_feature, fixed_skips_fixedP)
            self.logger.image_summary(denorm(fixed_recon_image),
                        opts=dict(title='recon_initial',
                                    caption='recon_initial'
                            )
            )

        for epoch in range(self.epoch):
            for num, real_image in enumerate(self.loader):
                real_image = real_image.to(self.device)

                feature_vgg_4_1, skips_vgg, unpool_idxs, unpool_sizes = self.encoder(real_image)

                feature_4_1, skips_fixedP = self.fixedP_encoder(real_image)
                recon_image, skips_fixedP = self.fixedP_decoder(feature_4_1, skips_fixedP)
                _, skips_fixedP_re = self.fixedP_encoder(recon_image)

                recon_loss = self.MSE_loss(recon_image, real_image) * self.recon_weight
                recon_feature, recon_skips, _, _ = self.encoder(recon_image)               
                
                loss = recon_loss 
                
                feature_loss = torch.zeros(1).to(self.device)
                mirror_feature_loss = torch.zeros(1).to(self.device)
                if 'perceptual' in self.loss_option:                    
                    feature_loss += self.MSE_loss(feature_4_1, feature_vgg_4_1)
                    feature_loss += self.MSE_loss(skips_fixedP_re['conv3_1'], skips_vgg['conv3_1'])
                    feature_loss += self.MSE_loss(skips_fixedP_re['conv2_1'], skips_vgg['conv2_1'])
                    feature_loss += self.MSE_loss(skips_fixedP_re['conv1_1'], skips_vgg['conv1_1'])
                if 'gram' in self.loss_option:
                    feature_loss += self.MSE_loss(gram_matrix(feature_4_1), gram_matrix(feature_vgg_4_1))                    
                    feature_loss += self.MSE_loss(gram_matrix(skips_fixedP_re['conv3_1']), gram_matrix(skips_vgg['conv3_1']))
                    feature_loss += self.MSE_loss(gram_matrix(skips_fixedP_re['conv2_1']), gram_matrix(skips_vgg['conv2_1']))
                    feature_loss += self.MSE_loss(gram_matrix(skips_fixedP_re['conv1_1']), gram_matrix(skips_vgg['conv1_1']))
                
                if 'mirror_per' in self.loss_option:
                    mirror_feature_loss += self.MSE_loss(skips_fixedP['conv3_2_fixedP_recon'], skips_fixedP['conv3_1'])
                    mirror_feature_loss += self.MSE_loss(skips_fixedP['conv2_2_fixedP_recon'], skips_fixedP['conv2_1'])
                    mirror_feature_loss += self.MSE_loss(skips_fixedP['conv1_2_fixedP_recon'], skips_fixedP['conv1_1'])                    
#                     mirror_feature_loss = self.MSE_loss(skips_wave['conv4_1_wavrecon'], skips_wave['conv3_4'])
#                     mirror_feature_loss += self.MSE_loss(skips_wave['conv3_1_wavrecon'], skips_wave['conv2_2'])
#                     mirror_feature_loss += self.MSE_loss(skips_wave['conv2_1_wavrecon'], skips_wave['conv1_2'])                  
                if 'mirror_gram' in self.loss_option:
                    mirror_feature_loss += self.MSE_loss(gram_matrix(skips_fixedP['conv3_2_fixedP_recon']), gram_matrix(skips_vgg['conv3_1']))
                    mirror_feature_loss += self.MSE_loss(gram_matrix(skips_fixedP['conv2_2_fixedP_recon']), gram_matrix(skips_vgg['conv2_1']))
                    mirror_feature_loss += self.MSE_loss(gram_matrix(skips_fixedP['conv1_2_fixedP_recon']), gram_matrix(skips_vgg['conv1_1']))
                    
                loss = recon_loss + feature_loss * self.feature_weight + mirror_feature_loss * self.feature_weight
                
                self.reset_grad()
                loss.backward()
                if not self.fixedP_enc_freeze:
                    self.enc_optim.step()
                self.dec_optim.step()

                if (num + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed: [{}], Epoch: [{}/{}], Iter: [{}/{}] \
                        Recon Loss: {:.4f}, Feature loss: {:.4f}".format(elapsed, epoch + 1, self.epoch, num + 1, len(self.loader),
                            recon_loss.item(), feature_loss.item() + mirror_feature_loss.item()))
                    info = {
                        "Loss": loss.item()
                    }
                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, (len(self.loader) * epoch) + num + 1)

                    if (num + 1) % 500 == 0:
                        with torch.no_grad():
                            fixed_feature, fixed_skips_fixedP = self.fixedP_encoder(fixed_sample)
                            fixed_recon_image, _= self.fixedP_decoder(fixed_feature, fixed_skips_fixedP)
                            self.logger.image_summary(denorm(fixed_recon_image),
                                        opts=dict(title='recon{}_{}'.format(epoch+1, num+1),
                                                    caption='recon{}_{}'.format(epoch+1, num+1)
                                            )
                            )
                        torch.save(self.fixedP_encoder.state_dict(),
                                    os.path.join(self.model_save_path, 
                                                 "fixedP_encoder_{}_{}_{:.4f}_{:.4f}.pth".format(epoch+1, num+1, recon_loss.item(), feature_loss.item()))
                        )
                        torch.save(self.fixedP_decoder.state_dict(),
                                    os.path.join(self.model_save_path, 
                                                 "fixedP_decoder_{}_{}_{:.4f}_{:.4f}.pth".format(epoch+1, num+1, recon_loss.item(), feature_loss.item()))
                        )

        torch.save(self.fixedP_encoder.state_dict(),
                    os.path.join(self.model_save_path, "fixedP_encoder.pth")
        )
        torch.save(self.fixedP_decoder.state_dict(),
                    os.path.join(self.model_save_path, "fixedP_decoder.pth")
        )
