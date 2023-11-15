import torch
from utils import *
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from kornia.color import rgb_to_hsv

def gram_matrix(input):
    a, b, c, d = input.size()  # batch size(=1), feature map number, dimensions
    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product

    # normalize the values of the gram matrix by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)



def compute_style_loss(self, generated_features, style_features):
    style_loss = 0.0
    WD=SinkhornDistance(eps=0.1, max_iter=100)
    for gen_feat, style_feat in zip(generated_features, style_features):
        G_gen = gram_matrix(gen_feat)
        G_style = gram_matrix(style_feat)

        # G_gen = gen_feat.view(gen_feat.shape[0],-1)
        # G_style = style_feat.view(style_feat.shape[0],-1)
        wd,_,_=WD(G_gen,G_style)
        # wd,_,_=self.WD(gen_feat,style_feat)
        style_loss += F.mse_loss(G_gen, G_style)
        style_loss += wd
    return style_loss/5


def compute_euclidean_loss(generated_features, style_features):
    loss = 0.0
    for gen_feat, style_feat in zip(generated_features, style_features):
        gen_feat=F.normalize(gen_feat,dim=1)
        style_feat=F.normalize(style_feat,dim=1)
        dis = euclidean_distance(gen_feat,style_feat)
        loss += dis

    return loss/3

def euclidean_distance(img1, img2):
    return torch.sqrt(torch.sum((img1 - img2) ** 2))




class ColorLoss(torch.nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, generated_img, original_img):
        # 转换颜色空间
        original_hsv = rgb_to_hsv(original_img)
        generated_hsv = rgb_to_hsv(generated_img)

        # 计算HSV通道的差异
        hue_loss = F.l1_loss(original_hsv[:, :, 0], generated_hsv[:, :, 0])
        saturation_loss = F.l1_loss(original_hsv[:, :, 1], generated_hsv[:, :, 1])
        value_loss = F.l1_loss(original_hsv[:, :, 2], generated_hsv[:, :, 2])

        # 综合三个通道的损失
        total_loss = hue_loss + saturation_loss + value_loss
        return total_loss