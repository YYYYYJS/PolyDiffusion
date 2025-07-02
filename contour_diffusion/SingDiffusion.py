import numpy as np
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from diffusers.models.resnet import ResnetBlock2D
from torch.utils.data import DataLoader

from contour_encoder import SingDiffusionEncoder, generate_image_patch_boundary_points
import torch
import os
from contour_diffusion.dataset.dataset import CustomDataset
from torchvision import transforms
from contour_diffusion.dataset.util import image_normalize
import torch.nn as nn


def torch_dfs(model):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


def get_rid_of_time(model):
    for module in torch_dfs(model):
        if isinstance(module, ResnetBlock2D):
            module.time_emb_proj = None
            module.time_embedding_norm = None


class SingDiffusion(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        self.sing_diffusion = UNet2DConditionModel.from_config(os.path.join(config_path, 'config.json'))
        self.contour_encoder = SingDiffusionEncoder(contour_length=10, num_heads=8, hidden_dim=128,
                                                    use_positional_embedding=True, num_layers=6)

    def forward(self, x, t, c):
        contour_output = self.contour_encoder(
            obj_description=c['obj_description'],
            obj_contour=c['obj_contour'],
            is_valid_obj=c['is_valid_obj'],
            obj_bbox=c['obj_bbox'],
        )
        c_ = contour_output
        model_pred = self.sing_diffusion(x, t, c_).sample

        return model_pred


transform = transforms.Compose([
    transforms.ToTensor(),
    image_normalize()
])
contour_ = torch.tensor(generate_image_patch_boundary_points([1])['boundary_points_resolution1']).view(1, 97, 2)
INITIAL_LOG_LOSS_SCALE = 20.0
obj_description = torch.tensor([0, 184, 184, 184, 184, 184, 184, 184, 184, 184]).int()
obj_bbox_ = torch.tensor([0, 0, 1, 1], dtype=torch.float16)
image_folder_path = 'S:/data/coco/256/val/images'
label_folder_path = 'S:/data/coco/256/val/conds'
# 创建数据集实例
batch_size = 64
dataset = CustomDataset(image_folder_path, label_folder_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
vae = AutoencoderKL.from_pretrained("G:/models-weight/stable-diffusion-models/v-1-5", subfolder='vae',
                                    torch_dtype=torch.float32)

model = SingDiffusion('S:\singdiffusion')

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-05,
                              betas=(0.9, 0.999),
                              weight_decay=0.01,
                              eps=1e-8)

criterion = torch.nn.MSELoss()

scheduler = DDIMScheduler()


def train(scheduler, model, vae, epochs):
    get_rid_of_time(model)
    model = model.to('cuda')
    vae = vae.to('cuda')
    vae.requires_grad_(False)
    scheduler.set_timesteps(100, device='cuda')

    for e in range(epochs):
        for i, data in enumerate(data_loader):
            batch, dictionary = data
            batch = batch.to('cuda')
            batch_size = batch.shape[0]
            size = batch.shape[2]
            batch = batch.to(dtype=torch.float32).reshape(batch_size, 3, size, size)

            vae_out = vae.encode(batch).latent_dist.sample()

            noisy_latents = torch.rand_like(vae_out).to('cuda')
            place_holder_time = torch.ones((batch_size,)).to(device='cuda', dtype=torch.long)

            obj_num = dictionary['obj_num']
            cond = {'obj_description': dictionary['obj_class'].to(dtype=torch.float32).reshape(batch_size, -1),
                    'obj_contour': dictionary['obj_contour'].to(dtype=torch.float32).reshape(batch_size, 10, 97, 2),
                    'is_valid_obj': dictionary['is_valid_obj'].to(dtype=torch.float32).reshape(batch_size, 10),
                    'obj_bbox': dictionary['obj_bbox'].to(dtype=torch.float32).reshape(batch_size, 10, 4),
                    'is_use_bbox': torch.tensor(False, dtype=torch.bool),
                    'obj_num': obj_num

                    }
            p = np.random.rand()
            if p < 0.10:
                cond['obj_description'] = obj_description.repeat(batch.shape[0], 1)
                cond['obj_contour'] = torch.zeros_like(cond['obj_contour'])
                cond['obj_contour'][:, 0] = contour_
                cond['obj_bbox'] = torch.zeros_like(cond['obj_bbox'])
                cond['obj_bbox'][:, 0] = obj_bbox_.repeat(batch.shape[0], 1)
                cond['is_valid_obj'] = torch.zeros_like(cond['is_valid_obj'])
                cond['is_valid_obj'][:, 0] = 1.0

            for key, value in cond.items():
                cond[key] = cond[key].to('cuda')

            model_pred = model(noisy_latents, place_holder_time, cond)
            loss = criterion(model_pred, vae_out)
            print(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()


train(scheduler, model, vae, 100)
