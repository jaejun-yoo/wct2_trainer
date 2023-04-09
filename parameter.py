import os
import nsml
import argparse


def str2bool(v):
    return v.lower() in ('true')


def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--wavenc_freeze', type=str2bool, default=True)
    parser.add_argument('--fixedP_enc_freeze', type=str2bool, default=True)
    parser.add_argument('--loss_option', type=str, default='')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    parser.add_argument('--feature_weight', type=float, default=10)
    parser.add_argument('--recon_weight', type=float, default=1000)

    parser.add_argument('--dataset', type=str, default=os.path.join(nsml.DATASET_PATH, 'train'))
    parser.add_argument('--img_size', type=int, default=256)

    parser.add_argument('--block', type=int, default=4)
    parser.add_argument('--model_save_path', type=str, default='./models')
    
    config = parser.parse_args()
    
    config.loss_option = config.loss_option.split(',')
    config.loss_option = [op.strip() for op in config.loss_option]
    
    return config
