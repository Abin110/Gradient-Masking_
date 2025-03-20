import os
import random
from io import BytesIO
from math import sqrt
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from skimage.transform import rotate
from skimage.util import view_as_blocks
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from rs_process import MessageEcodeDecode
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm




# 设置设备 GPU
device = "cuda"
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
        transforms.Resize(256),                      # Resize the image to 256x256
        transforms.CenterCrop(224),                  # Crop the center 224x224 region
        transforms.ToTensor(),                       # Convert PIL Image to PyTorch Tensor
        transforms.Normalize(                        # Normalize the image
        mean=[0.485, 0.456, 0.406],  # Mean of ImageNet dataset
        std=[0.229, 0.224, 0.225]  # Std deviation of ImageNet dataset
    )
    # Std deviation of ImageNet dataset

    ])


class Attacks:

    @staticmethod
    def mgs(model, x, target, mask, eps=0.1, alpha=0.05, iters=40):
        device = x.device
        noise = ((torch.randn_like(x) + 1e-6) * mask).to(device)

        for i in range(iters):
            alpha = alpha - (alpha - alpha / 100) / iters * i

            noise.requires_grad_(True)
            output = model(x + noise)
            loss = F.cross_entropy(output, target)
            loss.backward(retain_graph=True)

            grad, = torch.autograd.grad(loss, [noise])
            noise = noise - grad.detach().sign() * mask * alpha

            noise = torch.minimum(torch.maximum(noise, torch.zeros_like(x) - eps), torch.zeros_like(x) + eps)

            noise.grad = None

        x_adv = x + noise

        return x_adv


def denorm(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):

    mean = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()

    # 反归一化操作
    denormalized_tensor = tensor * std + mean

    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1).cuda()

    return denormalized_tensor

def compute_ssim_psnr(img1, img2):


    img1 = img1.permute(0, 2, 3, 1).cpu().detach().numpy()
    img2 = img2.permute(0, 2, 3, 1).cpu().detach().numpy()

    ssim_value = calculate_ssim(img1, img2)
    psnr_value = calculate_psnr(img1, img2)

    return ssim_value, psnr_value


def calculate_ssim(img1, img2, win_size=11, channel_axis=-1):

    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)

    img1_mean = img1.mean(dim=(0, 1, 2))
    img2_mean = img2.mean(dim=(0, 1, 2))

    img1_var = ((img1 - img1_mean) ** 2).mean(dim=(0, 1, 2))
    img2_var = ((img2 - img2_mean) ** 2).mean(dim=(0, 1, 2))
    img_covar = ((img1 - img1_mean) * (img2 - img2_mean)).mean(dim=(0, 1, 2))

    c1 = (0.01 * 1) ** 2
    c2 = (0.03 * 1) ** 2
    ssim_val = ((2 * img1_mean * img2_mean + c1) * (2 * img_covar + c2)) / \
               ((img1_mean ** 2 + img2_mean ** 2 + c1) * (img1_var + img2_var + c2))

    ssim_val = ssim_val.mean()

    return ssim_val.item()


def calculate_psnr(img1, img2):

    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)

    mse = F.mse_loss(img1, img2)

    psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse))

    return psnr_val.item()


def evaluation(dataloader, model, config):
    top1_accuracy_before_attack = 0
    top5_accuracy_before_attack = 0
    top1_accuracy_after_attack = 0
    top5_accuracy_after_attack = 0
    total = 0.0
    total_ssim = 0.0
    total_psnr = 0.0

    targeted_label = torch.tensor([0]).to(device)
    temperdetec = MessageEcodeDecode(config)
    mask = temperdetec.enmask(config['watermark_path'])

    with open(config['watermark_path'], 'rb') as f:
        original_bytes = bytearray(f.read())

    pbar = tqdm(dataloader)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        adv_images = Attacks.mgs(model, images, targeted_label, mask=mask,
                                 eps=config['attack_params']['eps'],
                                 alpha=config['attack_params']['alpha'],
                                 iters=config['attack_params']['iters']).to(device)
        outputs = model(images).to(device)
        adv_outputs = model(adv_images).to(device)

        #perturbation = adv_images - images
        #bytes_data = temperdetec.demask(perturbation,config['output_path'])

        pbar.set_description('Processing')

        _, predicted = torch.max(outputs, 1)
        _, adv_predicted = torch.max(adv_outputs, 1)
        _, top5_predicted = torch.topk(outputs, 5, dim=1)
        _, adv_top5_predicted = torch.topk(adv_outputs, 5, dim=1)
        total += labels.size(0)

        top1_accuracy_before_attack += (predicted == labels.to(device)).sum().item()
        top5_accuracy_before_attack += torch.sum(torch.eq(top5_predicted, labels.unsqueeze(1).expand_as(top5_predicted))).item()
        top1_accuracy_after_attack += (adv_predicted == labels.to(device)).sum().item()
        top5_accuracy_after_attack += torch.sum(torch.eq(adv_top5_predicted, labels.unsqueeze(1).expand_as(adv_top5_predicted))).item()
        total_ssim += compute_ssim_psnr(denorm(adv_images), denorm(images))[0]
        total_psnr += compute_ssim_psnr(denorm(adv_images), denorm(images))[1]

    return {
        'top1_accuracy_before_attack': top1_accuracy_before_attack / total * 100,
        'top5_accuracy_before_attack': top5_accuracy_before_attack / total * 100,
        'top1_accuracy_after_attack': top1_accuracy_after_attack / total * 100,
        'top5_accuracy_after_attack': top5_accuracy_after_attack / total * 100,
        'average_ssim': total_ssim / len(dataloader),
        'average_psnr': total_psnr / len(dataloader)
    }