import cv2
import numpy as np
import torch
from accelerate import Accelerator
from diffusers import UNet2DModel
import torch.nn as nn
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from contour_diffusion.dataset.dataset import CustomDataset
from contour_diffusion.resample import build_schedule_sampler
from contour_diffusion.respace import build_diffusion

from torchvision import transforms


class Model(nn.Module):
    def __init__(self, unet, hidden_dim=32):
        super(Model, self).__init__()
        self.unet = unet
        self.mask_proj = nn.Embedding(80, hidden_dim)

    def forward(self, x, mask, step):
        B = mask.shape[0]
        W = mask.shape[2]
        mask = mask.reshape(B, 1, -1)
        mask = mask.permute(0, 2, 1).int().to('cuda')
        mask_embed = torch.squeeze(self.mask_proj(mask), dim=2)
        mask_embed = mask_embed.permute(0, 2, 1)
        mask_embed = mask_embed.reshape(B, 32, W, -1)

        x = torch.cat((x, mask_embed), dim=1)

        model_output = self.unet(x, step)['sample']
        noise_ = model_output[:, :-1, :, :]
        mask_ = model_output[:, -1, :, :]
        return noise_, mask_


parser = argparse.ArgumentParser()

parser.add_argument("--config_file", type=str,
                    default='E:/ContourDiffusion_unet_coco2017/configs/COCO-stuff_256x256/ContourDiffusion_large.yaml')
known_args, unknown_args = parser.parse_known_args()

known_args = OmegaConf.create(known_args.__dict__)
cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
if unknown_args:
    unknown_args = OmegaConf.from_dotlist(unknown_args)
    cfg = OmegaConf.merge(cfg, unknown_args)


def get_mask(data):
    batch_size = data['obj_contour'].shape[0]
    arr = None
    for i in range(batch_size):
        h = 256
        w = 256
        mask = np.zeros((h, w), dtype=np.uint8)
        for j in range(data['obj_num'][0][i]):
            seg = data['obj_contour'][i][j + 1]
            seg[0::2] = seg[0::2] * 256
            seg[1::2] = seg[1::2] * 256
            seg = seg.to('cpu')
            seg = np.array(seg).reshape(-1, 2)  # [n_points, 2]
            cv2.fillPoly(mask, seg.astype(np.int32)[np.newaxis, :, :], 1)
        tensor = torch.unsqueeze(torch.tensor(mask), dim=0)

        if arr is None:
            arr = tensor
        else:
            arr = torch.cat((arr, tensor), dim=0)
    return arr


unet = UNet2DModel(sample_size=256, in_channels=35, out_channels=4, block_out_channels=(256, 512, 768, 1024)).to(
    'cuda')

# 创建扩散模型
diffusion = build_diffusion(cfg)

# 创建扩散进度采样器
schedule_sampler = build_schedule_sampler(cfg, diffusion)

model = Model(unet=unet)
model.mask_proj.to('cuda')
opt = torch.optim.AdamW(model.parameters(),
                        lr=1e-05,
                        betas=(0.9, 0.999),
                        weight_decay=0.01,
                        eps=1e-8)

get_MseLoss = nn.MSELoss()
epoch = 1

# 定义转换，例如将图像转换为张量
transform = transforms.Compose([
    transforms.ToTensor()
])

image_folder_path = 'S:/data/coco/val/images'
label_folder_path = 'S:/data/coco/val/conds'
# 创建数据集实例
dataset = CustomDataset(image_folder_path, label_folder_path, transform=transform)
batch_size = 1000
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
accumulation_steps = 4
accelerator = Accelerator()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


get_DiceLoss = DiceLoss()

model, data_loader, diffusion, opt = accelerator.prepare(model, data_loader, diffusion, opt)


def train():
    for e in range(epoch):
        print(f'epoch===={e}')
        for i, s in enumerate(data_loader):
            batch, cond = s
            batch = batch.to('cuda', dtype=torch.float32)
            mask = get_mask(cond).to('cuda', dtype=torch.float32)
            t, weights = schedule_sampler.sample(batch.shape[0], 'cuda')
            noise = torch.randn_like(batch)
            x_t = diffusion.q_sample(batch, t, noise=noise).to(dtype=torch.float32)
            model_output, model_mask = model(x_t, mask, t)

            loss = get_MseLoss(noise, model_output) + get_DiceLoss(model_mask, mask) * 0.01

            print(loss)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (i + 1) % accumulation_steps == 0:
                # 累积梯度并更新模型参数
                opt.step()
                opt.zero_grad()  # 清除累积的梯度


if __name__ == '__main__':
    train()
    torch.save(model.state_dict(), f'S:/weights/model.pt')
