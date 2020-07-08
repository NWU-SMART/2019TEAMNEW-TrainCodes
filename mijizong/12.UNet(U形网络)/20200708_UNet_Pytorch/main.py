# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/7/8
#  @Author: MiJizong
#  @Content: UNet——Pytorch
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: Null
# ----------------------开发者信息-----------------------------------------

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from DataHelper import *
from tqdm import tqdm
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1个GPU
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'  # 允许副本存在
import skimage.io as io

PATH = './model/unet_model.pt'

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 将数据归一化到[-1,1]
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()


def train_model(model, criterion, optimizer, dataload, num_epochs=10):
    best_model = model
    min_loss = 1000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in tqdm(dataload):  # 循环中添加一个进度提示信息用法
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch + 1, epoch_loss / step))
        if (epoch_loss / step) < min_loss:
            min_loss = (epoch_loss / step)
            best_model = model
    torch.save(best_model.state_dict(), PATH)
    return best_model


# 训练模型
def train():
    model = Unet(1, 1).to(device)
    batch_size = 1
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    # 获取训练数据集
    train_dataset = TrainDataset("D:/Office_software/PyCharm/Test/UNet/dataset/train/image",
                                 "D:/Office_software/PyCharm/Test/UNet/dataset/train/label",
                                 transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)


# 预测模型
def test():
    model = Unet(1, 1)
    model.load_state_dict(torch.load(PATH))
    test_dataset = TestDataset("D:/Office_software/PyCharm/Test/UNet/dataset/test/orient",
                               transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(test_dataset, batch_size=1)
    model.eval()  # 接受一个字符串str作为参数，并把这个参数作为脚本代码来执行。（不是很明白）
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for index, x in enumerate(dataloaders):
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            img_y = img_y[:, :, np.newaxis]
            img = labelVisualize(2, COLOR_DICT, img_y) if False else img_y[:, :, 0]
            # io.imsave("D:/Office_software/PyCharm/Test/UNet/dataset/test" + str(index) + "_predict.png", img)
            io.imsave("D:/Office_software/PyCharm/Test/UNet/dataset/test/predict/" + str(index) + "_predict.png", img)
            plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    print("开始训练")
    train()
    print("训练完成，保存模型")
    print("-" * 20)
    print("开始预测")
    test()
