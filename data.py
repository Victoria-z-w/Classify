import os
from tqdm import tqdm
import pandas as pd
import shutil
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms


if not os.path.exists(r'F:\train_path'):
    DatasetPath = r'F:\train_path'
    if not os.path.exists(DatasetPath):
        os.makedirs(DatasetPath)

    data_csv = pd.read_csv(r'F:\train.csv')

    for index in tqdm(data_csv.index):
        imagePath = os.path.join(r'F:\train_images', data_csv.loc[index]['image'])
        label = data_csv.loc[index]['labels'].replace(' ', '-')
        labelDir = os.path.join(DatasetPath, label)
        if not os.path.exists(labelDir):
            os.makedirs(labelDir)

        shutil.copy(imagePath, labelDir)

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def calculate(dataset, batch_size=64, num_workers=6):

    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)

    # 初始化累加器
    total_images = 0
    mean = 0.0
    var = 0.0

    for data in loader:
        images, _ = data
        batch_size, channels, height, width = images.shape
        total_images += batch_size

        # 计算当前batch的均值和方差
        batch_mean = images.mean(dim=[0, 2, 3])  # shape: (channels,)
        batch_var = images.var(dim=[0, 2, 3], unbiased=False)  # 有偏估计

        # 更新全局统计量
        mean += batch_mean * batch_size
        var += (batch_var + batch_mean.pow(2)) * batch_size

    # 计算最终结果
    mean /= total_images
    var = var / total_images - mean.pow(2)
    std = torch.sqrt(var + 1e-8)  # 防止除零错误

    return mean, std

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])

    dataset = ImageFolder(root = r'F:\train_path',transform=transform)

    mean, std = calculate(dataset)
    print(mean)
    print(std)



