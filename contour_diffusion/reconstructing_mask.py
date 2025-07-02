import os.path

import cv2
import numpy as np
from torch import nn
from contour_diffusion.dataset.mask_data import MaskDataset
from contour_diffusion.contour_encoder import ContourTransformer, xf_convert_module_to_f16
import torch
from contour_diffusion.contour_diffusion_unet import Upsample
from torch.utils.data import DataLoader
from torchvision import transforms
from contour_encoder import LayerNorm


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class UpSampleModule(nn.Module):
    def __init__(self):
        super(UpSampleModule, self).__init__()
        self.conv_transpose0 = Upsample(256, True, 2, 128)
        self.conv_transpose1 = Upsample(128, True, 2, 64)
        self.conv_transpose2 = Upsample(64, True, 2, 32)

        self.act = SiLU()
        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_transpose0(x)
        x = self.act(x)
        x = self.conv_transpose1(x)
        x = self.act(x)
        x = self.conv_transpose2(x)

        x = self.conv_out(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Linear(2, 96)
        self.encoder = ContourTransformer(96, 96, layers=6, heads=8)

        self.proj = nn.Linear(96, 256)
        self.conv_in = nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1)
        self.decoder = UpSampleModule()

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)

        x = self.proj(x)
        x = x.reshape(x.shape[0], 1, 16, 16)
        x = self.conv_in(x)
        x = self.decoder(x)

        return x


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-8

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


model = Model()
# model.load_state_dict(torch.load('model2.pt'))
model = model.to('cuda')
dice = DiceLoss()
image_folder_path = 'S:/data/coco/128/mask_data/mask'
label_folder_path = 'S:/data/coco/128/mask_data/poly'
bbox_folder_path = 'S:/data/coco/128/mask_data/bbox'
# 创建数据集实例
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = MaskDataset(image_folder_path, label_folder_path,bbox_folder_path ,transform)
batch_size = 768
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
epoch = 2500
ac = nn.Sigmoid().to('cuda')
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-04,
                              betas=(0.9, 0.999),
                              weight_decay=0.01,
                              eps=1e-8)
for i in range(epoch):

    for j, data in enumerate(data_loader):
        x, mask = data
        x = x / 128
        x = x.to('cuda', dtype=torch.float32)
        x = x.view(batch_size, 96, 2)
        mask = mask.to('cuda', dtype=torch.float32)
        y = model(x)

        loss = dice(y, mask)

        print(f'epoch=={i + 1}---step=={j + 1}---loss=={loss}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

torch.save(model.state_dict(), 'model2.pt')
