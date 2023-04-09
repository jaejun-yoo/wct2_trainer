import os
import time
import argparse

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image

from utils import denorm, make_folder
from data_loader import get_img
from model import WaveEncoder, WaveDecoder
from photo_smooth import Propagator
from smooth_filter import smooth_filter

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DPST', type=str2bool, default=False)
    parser.add_argument('--content_path', default=None)
    parser.add_argument('--style_path', type=str, default="./picture/night2.jpg")
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--transform_index', 
            type=lambda s: [int(item.strip()) for item in s.split(',')], default=[0])
    parser.add_argument('--index_weight', help="determine how much weight to selected index",
            type=lambda s: [float(item.strip()) for item in s.split(',')], default=[1, 1, 1, 1])
    parser.add_argument('--content_seg_path', default=None)
    parser.add_argument('--style_seg_path', default=None)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--conv_level', type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--post_process', type=str2bool, default=True)
    parser.add_argument('--no_seg', type=str2bool, default=False)
    return parser.parse_args()

def change_seg(seg):
    color_dict = {
        (0, 0, 255): 3, # blue
        (0, 255, 0): 2, # green
        (0, 0, 0): 0, # black
        (255, 255, 255): 1, # white
        (255, 0, 0): 4, # red
        (255, 255, 0): 5, # yellow
        (128, 128, 128): 6, # grey
        (0, 255, 255): 7, # lightblue
        (255, 0, 255): 8 # purple
    }
    arr_seg = np.asarray(seg)
    new_seg = np.zeros(arr_seg.shape[:-1])
    for x in range(arr_seg.shape[0]):
        for y in range(arr_seg.shape[1]):
            if tuple(arr_seg[x, y, :]) in color_dict:
                new_seg[x, y] = color_dict[tuple(arr_seg[x, y, :])]
            else:
                min_dist_index = 0
                min_dist = 99999
                for key in color_dict:
                    dist = np.sum(np.abs(np.asarray(key) - arr_seg[x, y, :]))
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_index = color_dict[key]
                    elif dist == min_dist:
                        try:
                            min_dist_index = new_seg[x, y-1, :]
                        except:
                            pass
                new_seg[x, y] = min_dist_index
    return new_seg.astype(np.uint8)
    
class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

class Transformer(nn.Module):
    def __init__(self, alpha, conv_level):
        """NOTE: content_seg and style_seg will be np.array"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.conv_level = conv_level
        self.encoder = dict()
        self.decoder = dict()
        for i in range(4, self.conv_level - 1, -1):
            self.encoder[i] = WaveEncoder(i).to(self.device)
            self.decoder[i] = WaveDecoder(i).to(self.device)
            self.encoder[i].load_state_dict(
                torch.load("./models/conv{}_1.pth".format(i))
            )
            self.decoder[i].load_state_dict(
                torch.load("./models/dec{}_1.pth".format(i))
            )
    
    def transform(self, cont_img, styl_img, cont_seg, styl_seg, num):
        with torch.no_grad():
            self.compute_label_info(cont_seg, styl_seg)
            img = cont_img
            for i in range(4, self.conv_level - 1, -1):
                style_decomp = self.encoder[i](styl_img, 1)
                content_decomp = self.encoder[i](img, 1)
                if i != 1:
                    lst = []
                    for j in range(0, 4):
                        if j in config.transform_index:
                            transformed = self.feature_wct(content_decomp[j], style_decomp[j], cont_seg, styl_seg, config.index_weight[j])
                            lst.append(self.alpha * transformed + (1 - self.alpha) * content_decomp[j])
                        else:
                            lst.append(content_decomp[j])
                
                    transformed = torch.cat(lst, dim=1)
                else:
                    transformed = self.feature_wct(content_decomp, style_decomp, cont_seg, styl_seg, 1)
                    transformed = self.alpha * transformed + (1 - self.alpha) * content_decomp

                img = self.decoder[i](transformed, 1)
                save_image(denorm(img), "./result/{}/level{}.png".format(num, i))
        return img
    def compute_label_info(self, cont_seg, styl_seg):
        if cont_seg.size == False or styl_seg.size == False:
            return
        max_label = np.max(cont_seg) + 1
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(max_label)
        for l in self.label_set:
            # if l==0:
            #   continue
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
            self.label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)

    def feature_wct(self, cont_feat, styl_feat, cont_seg, styl_seg, weight):
        cont_feat = cont_feat.squeeze(0)
        styl_feat = styl_feat.squeeze(0)
        cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2)
        styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2)
        cont_feat_view = cont_feat.view(cont_c, -1).clone()
        styl_feat_view = styl_feat.view(styl_c, -1).clone()

        if cont_seg.size == False or styl_seg.size == False:
            target_feature = self.wct_core(cont_feat_view, styl_feat_view, weight)
        else:
            target_feature = cont_feat.view(cont_c, -1).clone()
            if len(cont_seg.shape) == 2:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg).resize((cont_w, cont_h), Image.NEAREST))
            else:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg, mode='RGB').resize((cont_w, cont_h), Image.NEAREST))
            if len(styl_seg.shape) == 2:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg).resize((styl_w, styl_h), Image.NEAREST))
            else:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg, mode='RGB').resize((styl_w, styl_h), Image.NEAREST))

            for l in self.label_set:
                if self.label_indicator[l] == 0:
                    continue
                cont_mask = np.where(t_cont_seg.reshape(t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l)
                styl_mask = np.where(t_styl_seg.reshape(t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l)
                if cont_mask[0].size <= 0 or styl_mask[0].size <= 0:
                    continue

                cont_indi = torch.LongTensor(cont_mask[0])
                styl_indi = torch.LongTensor(styl_mask[0])
                if torch.cuda.is_available():
                    cont_indi = cont_indi.to(self.device)
                    styl_indi = styl_indi.to(self.device)

                cFFG = torch.index_select(cont_feat_view, 1, cont_indi)
                sFFG = torch.index_select(styl_feat_view, 1, styl_indi)
                # print(len(cont_indi))
                # print(len(styl_indi))
                tmp_target_feature = self.wct_core(cFFG, sFFG, weight)
                # print(tmp_target_feature.size())
                if torch.__version__ >= "0.4.0":
                    # This seems to be a bug in PyTorch 0.4.0 to me.
                    new_target_feature = torch.transpose(target_feature, 1, 0)
                    new_target_feature.index_copy_(0, cont_indi, \
                            torch.transpose(tmp_target_feature,1,0))
                    target_feature = torch.transpose(new_target_feature, 1, 0)
                else:
                    target_feature.index_copy_(1, cont_indi, tmp_target_feature)

        target_feature = target_feature.view_as(cont_feat)
        ccsF = target_feature.float().unsqueeze(0)
        return ccsF

    def wct_core(self, cont_feat, styl_feat, weight):
        cFSize = cont_feat.size()
        c_mean = torch.mean(cont_feat, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean
        
        iden = torch.eye(cFSize[0])  # .double()
        if torch.cuda.is_available():
            iden = iden.to(self.device)
        try:
            contentConv = torch.mm(cont_feat, cont_feat.t()).div(cFSize[1] - 1) + iden
        except:
            contentConv = torch.mm(cont_feat, cont_feat.t()) + iden

        # del iden
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
        # c_e2, c_v = torch.eig(contentConv, True)
        # c_e = c_e2[:,0]
        
        k_c = cFSize[0]
        for i in range(cFSize[0] - 1, -1, -1):
            if c_e[i] >= 0.00001:
                k_c = i + 1
                break
        
        sFSize = styl_feat.size()
        s_mean = torch.mean(styl_feat, 1)
        styl_feat = styl_feat - s_mean.unsqueeze(1).expand_as(styl_feat)
        try:
            styleConv = torch.mm(styl_feat, styl_feat.t()).div(sFSize[1] - 1)
        except:
            styleConv = torch.mm(styl_feat, styl_feat.t())
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
        
        k_s = sFSize[0]
        for i in range(sFSize[0] - 1, -1, -1):
            if s_e[i] >= 0.00001:
                k_s = i + 1
                break
        
        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cont_feat)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d) * weight), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    def size_fix(self, encode_feature, decode_feature):
        encode_h, encode_w = list(encode_feature.shape)[-2:]
        decode_h, decode_w = list(decode_feature.shape)[-2:]
        if encode_h == decode_h and encode_w == decode_w:
            return decode_feature
        else:
            diff_h = encode_h - decode_h
            diff_w = encode_w - decode_w
            pad = nn.ZeroPad2d((0, diff_w, 0, diff_h))
            return pad(decode_feature)

    def get_gram(self, cont_img, styl_img, cont_seg, styl_seg):
        result = []

        self.compute_label_info(cont_seg, styl_seg)
        with torch.no_grad():
            cont_features = self.encoder[4].get_feature_map(cont_img)
            styl_features = self.encoder[4].get_feature_map(styl_img)

            for level, cont_feat in enumerate(cont_features):
                level_gram = []
                cont_feat = cont_feat.squeeze(0)
                styl_feat = styl_features[level].squeeze(0)

                cont_c, cont_h, cont_w = cont_feat.size()
                styl_c, styl_h, styl_w = styl_feat.size()

                cont_feat_view = cont_feat.view(cont_c, -1).clone()
                styl_feat_view = styl_feat.view(styl_c, -1).clone()

                t_cont_seg = np.asarray(Image.fromarray(cont_seg).resize((cont_w, cont_h),
                                                                         Image.NEAREST))
                t_styl_seg = np.asarray(Image.fromarray(styl_seg).resize((styl_w, styl_h),
                                                                         Image.NEAREST))

                for l in self.label_set:
                    if self.label_indicator[l] == 0:
                        continue
                    cont_mask = np.where(t_cont_seg.reshape(t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l)
                    styl_mask = np.where(t_styl_seg.reshape(t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l)
                    if cont_mask[0].size <= 0 or styl_mask[0].size <= 0:
                        continue

                    cont_indi = torch.LongTensor(cont_mask[0])
                    styl_indi = torch.LongTensor(styl_mask[0])
                    if torch.cuda.is_available():
                        cont_indi = cont_indi.to(self.device)
                        styl_indi = styl_indi.to(self.device)

                    cFFG = torch.index_select(cont_feat_view, 1, cont_indi)
                    sFFG = torch.index_select(styl_feat_view, 1, styl_indi)

                    cont_G = torch.mm(cFFG, cFFG.t()).div(cont_h * cont_w)
                    styl_G = torch.mm(sFFG, sFFG.t()).div(styl_h * styl_w)

                    level_gram.append(nn.functional.mse_loss(cont_G.unsqueeze(0),
                                                             styl_G.unsqueeze(0)).item())
                result.append(np.sum(np.asarray(level_gram)))
        return result

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = get_parameters()
    print(config)

    transformer = Transformer(config.alpha, config.conv_level)
    p_pro = Propagator()

    content_list, style_list = [], []
    content_seg_list, style_seg_list = [], []

    if config.DPST:
        content_path = "./picture/input/in{}.png"
        style_path = "./picture/style/tar{}.png"
        content_seg_path = "./picture/segmentation/in{}.png"
        style_seg_path = "./picture/segmentation/tar{}.png"
        for i in range(1, 61):
            content_list.append(content_path.format(i))
            style_list.append(style_path.format(i))
            content_seg_list.append(content_seg_path.format(i))
            style_seg_list.append(style_seg_path.format(i))
    else:
        content_list.append(config.content_path)
        style_list.append(config.style_path)
        content_seg_list.append(config.content_seg_path)
        style_seg_list.append(config.style_seg_path)
    
    print(len(content_list), len(style_list), len(content_seg_list), len(style_seg_list))
    
    for num in range(len(content_list)):
        try:
            print("{}th Image".format(num))
            style = get_img(style_list[num], config.img_size).to(device)
            try:
                content = get_img(content_list[num], config.img_size).to(device)
            except:
                print("NO CONTENT! USE GAUSSIAN NOISE AS INPUT")
                content = torch.randn_like(style).to(device)

            try:
                if config.no_seg:
                    raise NotImplementedError
                cont_seg = Image.open(content_seg_list[num])
                styl_seg = Image.open(style_seg_list[num])
                if config.img_size:
                    cont_seg.resize((config.img_size, config.img_size), Image.NEAREST)
                    styl_seg.resize((config.img_size, config.img_size), Image.NEAREST)
                if len(np.asarray(cont_seg).shape) == 3:
                    cont_seg = change_seg(cont_seg)
                if len(np.asarray(styl_seg).shape) == 3:
                    styl_seg = change_seg(styl_seg)
            except Exception as e:
                print("{}th Image: Failed to load segmentation map: {}".format(num, e))
                cont_seg = []
                styl_seg = []
                
            cont_seg = np.asarray(cont_seg)
            styl_seg = np.asarray(styl_seg)
            
            make_folder('./result/{}'.format(num))
            save_image(denorm(content), "./result/{}/content.png".format(num))
            save_image(denorm(style), "./result/{}/style.png".format(num))

            with Timer("Elapsed time in WCT: %f"):
                out_img = transformer.transform(content, style, cont_seg, styl_seg, num)
            save_image(denorm(out_img), "./result/{}/transfered.png".format(num))

            if config.post_process:
                with Timer("Elapsed time in propagation: %f"):
                    out_img = p_pro.process("./result/{}/level{}.png".format(num, config.conv_level),
                                content_list[num])
                out_img.save("./result/{}/smoothed.png".format(num))

                with Timer("Elapsed time in post processing: %f"):
                    out_img = smooth_filter("./result/{}/smoothed.png".format(num), 
                                content_list[num])
                out_img.save("./result/{}/result.png".format(num))

                with Timer("Elapsed time in post processing: %f"):
                    out_img = smooth_filter("./result/{}/level{}.png".format(num, config.conv_level),
                                content_list[num])
                out_img.save("./result/{}/only_post.png".format(num))
        except Exception as e:
            print("ERR! {}th image: {}".format(num, e))