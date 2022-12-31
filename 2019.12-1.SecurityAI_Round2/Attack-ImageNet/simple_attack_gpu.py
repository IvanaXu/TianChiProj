# Helper function for extracting features from pre-trained models
# /data/code/gproj/code/SecurityAI_Round2/code/Attack-ImageNet

import sys, os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
import glob

from attacker import Attacker
from loader import ImageNet_A
from utils.Resnet import resnet152_denoise, resnet101_denoise, resnet152
from utils.Normalize import Normalize, Permute


class Ensemble(nn.Module):
    def __init__(self, model1, model2, model3):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        logits3 = self.model3(x)

        # fuse logits
        logits_e = (logits1 + logits2 + logits3) / 3

        return logits_e


def load_model():
    i_path = "/data/data/SecurityAI_Round2/"
    pretrained_model1 = resnet101_denoise()
    loaded_state_dict = torch.load(f'{i_path}Adv_Denoise_Resnext101.pytorch')
    pretrained_model1.load_state_dict(loaded_state_dict, strict=True)

    pretrained_model2 = resnet152_denoise()
    loaded_state_dict = torch.load(f'{i_path}Adv_Denoise_Resnet152.pytorch')
    pretrained_model2.load_state_dict(loaded_state_dict)

    pretrained_model3 = resnet152()
    loaded_state_dict = torch.load(f'{i_path}Adv_Resnet152.pytorch')
    pretrained_model3.load_state_dict(loaded_state_dict)

    model1 = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model1
        )
    
    model2 = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model2
        )

    model3 = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model3
        )

    return model1, model2, model3


if __name__ == '__main__':
    m_path = "/data/code/gproj/code/SecurityAI_Round2/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=f'{m_path}/data/', type=str, help='path to data')
    parser.add_argument('--output_dir', default=f'{m_path}/out/season2/t00001/', type=str, help='path to results')

    parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
    # 100
    parser.add_argument('--steps', default=2000, type=int, help='iteration steps')
    parser.add_argument('--max_norm', default=32, type=float, help='Linf limit')
    parser.add_argument('--div_prob', default=0.9, type=float, help='probability of diversity')
    
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, 'images')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # ensemble model
    model1, model2, model3 = load_model()
    model = Ensemble(model1, model2, model3)

    model.cuda()
    model.eval()

    # set dataset
    dataset = ImageNet_A(args.input_dir)
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=args.batch_size, 
                                         shuffle=False)

    # set attacker
    attacker = Attacker(steps=args.steps, 
                        max_norm=args.max_norm/255.0,
                        div_prob=args.div_prob,
                        device=torch.device('cuda'))
    # RuntimeError: expected device cuda:0 but got device cpu
    
    for ind, (img, label_true, label_target, filenames) in enumerate(loader):
        # run attack
        adv = attacker.attack(model, img.cuda(), label_true.cuda(), label_target.cuda())

        # save results
        for bind, filename in enumerate(filenames):
            out_img = adv[bind].detach().cpu().numpy()
            delta_img = np.abs(out_img - img[bind].numpy()) * 255.0

            print('Attack on {}:'.format(os.path.split(filename)[-1]))
            print('Max: {0:.0f}, Mean: {1:.2f}'.format(np.max(delta_img), np.mean(delta_img)))

            out_img = np.transpose(out_img, axes=[1, 2, 0]) * 255.0
            out_img = out_img[:, :, ::-1]

            out_filename = os.path.join(output_dir, os.path.split(filename)[-1])
            cv2.imwrite(out_filename, out_img)



