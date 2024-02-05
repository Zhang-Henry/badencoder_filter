# %%
import sys
sys.path.append('../')

import torch
from torchvision import transforms, models
from torchvision.io import read_image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from models import get_encoder_architecture_usage
import argparse
from optimize_filter.tiny_network import U_Net_tiny
from util import clamp_batch_images
# %%
import torch.nn.functional as F
import os,random,copy
import kornia.augmentation as A
from PIL import Image, ImageDraw, ImageFont

args=argparse.Namespace(
    pretrained_encoder='../output/cifar10/clean_encoder/model_1000.pth',
    encoder_usage_info='cifar10',
    filter='../output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32',
)

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

# Define the transform
finetune_transform_cifar10 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=1),
    transforms.RandomGrayscale(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# Load the image
img_path = '../data/cifar10/test/image_9922_label_[5].jpeg'  # Replace with the path to the image file
original_image = Image.open(img_path).convert('RGB')

# Let's assume we want to generate 100 transformed versions of the image
num_samples = 1000
all_transformed_images = []

def add_text_to_image(image, text, position=(0, 0), font_size=20, font_color=(255, 255, 255)):
    """
    在给定图像上添加文本。
    :param image: PIL.Image对象。
    :param text: 要添加的文本字符串。
    :param position: 文本添加的位置，形式为(x, y)。
    :param font_size: 文本的字体大小。
    :param font_color: 文本的颜色，形式为(R, G, B)。
    :return: 添加了文本的图像。
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", font_size)  # 可能需要调整字体路径
    draw.text(position, text, fill=font_color, font=font)
    return image

def generate_tsne(args):
    # 假设其他函数如get_encoder_architecture_usage等已经定义

    all_tsne_images = []  # 用于存储所有t-SNE图像

    for epoch in range(0, 201, 25):  # 假设模型训练了1000个epochs
        print('Ploting t-SNE for epoch:', epoch)
        encoder_file = os.path.join(args.filter, f'model_{epoch}.pth')
        filter_file = os.path.join(args.filter, f'unet_filter_{epoch}_trained.pt')

        if not os.path.exists(encoder_file) or not os.path.exists(filter_file):
            continue  # 如果文件不存在，跳过此次循环

        # 加载模型和滤镜
        checkpoint = torch.load(encoder_file)
        encoder = get_encoder_architecture_usage(args).cuda()
        encoder.load_state_dict(checkpoint['state_dict'])

        net = U_Net_tiny(img_ch=3, output_ch=3)
        state_dict = torch.load(filter_file, map_location=torch.device('cuda:0'))
        net.load_state_dict(state_dict['model_state_dict'])
        net = net.cuda().eval()

        # 省略了处理图像和生成t-SNE图的代码，假设你已经有了一个函数来处理这个
        # Apply the transformation multiple times to the image
        for _ in range(num_samples):
            transformed_image = finetune_transform_cifar10(original_image)
            all_transformed_images.append(transformed_image)

        transformed_image_clean = test_transform_cifar10(original_image)
        all_transformed_images.append(transformed_image_clean) # clean image

        img_backdoor_filter=net(transformed_image_clean.unsqueeze(0).cuda())
        img_backdoor_filter=clamp_batch_images(img_backdoor_filter,args).squeeze(0).detach().cpu() # filter image

        all_transformed_images.append(img_backdoor_filter)

        all_transformed_images_tensor = torch.stack(all_transformed_images).cuda()
        # Disable gradient computation since we are only doing inference
        with torch.no_grad():
            # Pass the transformed images through the encoder
            encoded_features = encoder(all_transformed_images_tensor)

        features_np = encoded_features[0].detach().cpu().numpy()
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        features_tsne = tsne.fit_transform(features_np)
        transformed_aug = features_tsne[:-2]
        transformed_clean = features_tsne[-2:-1]
        transformed_filter = features_tsne[-1:]

        plt.figure(figsize=(10, 10))
        plt.scatter(transformed_aug[:, 0], transformed_aug[:, 1], c='blue', label='Augmented')
        plt.scatter(transformed_clean[:, 0], transformed_clean[:, 1], c='green', label='Clean')
        plt.scatter(transformed_filter[:, 0], transformed_filter[:, 1], c='red', label='FilterAttack', marker='x')

        plt.legend()
        plt.tight_layout()


        # 保存t-SNE图像
        tsne_image_path = f'TSNE/insight/tsne_{epoch}.png'
        plt.savefig(tsne_image_path)
        all_tsne_images.append(tsne_image_path)

    epochs = range(0, 201, 25)  # 根据你的循环范围
    images_and_epochs = zip(all_tsne_images, epochs)  # 将图像路径和对应的epoch配对

    total_height = 0
    max_width = 0
    for img_path, _ in images_and_epochs:
        with Image.open(img_path) as img:
            width, height = img.size
            total_height += height
            max_width = max(max_width, width)

    # 创建一个足够大的空白图像来容纳所有图像和文本
    combined_image = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))

    # 准备在图像上绘制文本
    draw = ImageDraw.Draw(combined_image)
    # 尝试加载系统字体，或者替换为你系统中可用的字体路径
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    y_offset = 0
    for img_path, epoch in zip(all_tsne_images, epochs):
        with Image.open(img_path) as img:
            combined_image.paste(img, (0, y_offset))
            # 在每张图像的顶部添加epoch标注
            draw.text((10, y_offset + 10), f'Epoch: {epoch}', fill="black", font=font)
            y_offset += img.height

    combined_image.save(f'TSNE/aug/combined_tsne_with_epochs_25epoch.png')

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--filter_dir", type=str, required=True, help="Directory containing filter models.")
    # args = parser.parse_args()
    args=argparse.Namespace(
        pretrained_encoder='../output/cifar10/clean_encoder/model_1000.pth',
        encoder_usage_info='cifar10',
        filter='../output/cifar10/svhn_backdoored_encoder/2023-12-26-13:50:32',
    )
    generate_tsne(args)
