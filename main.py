import random
import time
import matplotlib.pyplot as plt
from attack import evaluation
import torch
import torchvision
from torchvision.transforms import transforms
from matplotlib.ticker import ScalarFormatter

device = 'cuda'

config = {
    'watermark_path': 'logo/text_2.txt',
    'message_type': 'text',
    'host_image_size': (224, 224),
    'ecc_size': 85,
    'batch_size': 1,
    'attack_params': {
        'eps': 0.1,
        'alpha': 0.05,
        'iters': 40
    },
    'dataset_path': "F:\\ILSVRC2012_img_\\ILSVRC2012_img_val",
    'output_path': "decoded_text.txt"
}

# 定义数据集和模型
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cifar_10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])


dataset_path = "F:\ILSVRC2012_img_\ILSVRC2012_img_val"
dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

random.seed(42)
selected_indices = random.sample(range(len(dataset)), 1000)
subset_dataset = torch.utils.data.Subset(dataset, selected_indices)


cifarset = torchvision.datasets.CIFAR10('GMM_experiment-main/cifar_10', train=False, transform=cifar_10_transform, download=True)

dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=config['batch_size'], shuffle=True)

#model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
model= torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.to(device)
model.eval()

# 初始化数据存储
top1_after = []
top5_after = []
ssim_values = []
psnr_values = []

# 模拟 evaluation 函数的调用
for ecc in [85]:  # ECC 的取值范围
    print(f'++++++++ECC_simbol:{ecc}++++++++')
    config['ecc_size'] = ecc  # 更新配置字典中的 ECC 大小
    result = evaluation(dataloader, model, config)

    # 将结果存储到列表中
    top1_after.append(result['top1_accuracy_after_attack'])
    top5_after.append(result['top5_accuracy_after_attack'])
    ssim_values.append(result['average_ssim'])
    psnr_values.append(result['average_psnr'])

print(top1_after, top5_after, ssim_values, psnr_values)


model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a1", pretrained=True)
model.to(device)
model.eval()

# 初始化数据存储
top1_after = []
top5_after = []
ssim_values = []
psnr_values = []

# 模拟 evaluation 函数的调用
for ecc in [0]:  # ECC 的取值范围
    print(f'++++++++ECC_simbol:{ecc}++++++++')
    config['ecc_size'] = ecc  # 更新配置字典中的 ECC 大小
    result = evaluation(dataloader, model, config)

    # 将结果存储到列表中
    top1_after.append(result['top1_accuracy_after_attack'])
    top5_after.append(result['top5_accuracy_after_attack'])
    ssim_values.append(result['average_ssim'])
    psnr_values.append(result['average_psnr'])

print(top1_after, top5_after, ssim_values, psnr_values)