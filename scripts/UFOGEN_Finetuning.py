import copy
import math
import random
import torch
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader
import argparse
from omegaconf import OmegaConf
from contour_diffusion.contour_diffusion_unet import build_model
from torchvision import transforms
from contour_diffusion.dataset.dataset import CustomDataset
from contour_diffusion.dataset.util import image_normalize
from contour_diffusion.resample import build_schedule_sampler
from contour_diffusion.respace import build_diffusion
from contour_diffusion.train_util import get_mask

batch_size = 2
epoch = 10000

checkpoint = "F:/models-weight/stable-diffusion-models/v-1-5"
# 加载两个工具类
# scheduler = DDIMScheduler.from_pretrained(checkpoint, subfolder='scheduler', torch_dtype=torch.float32)
transform = transforms.Compose([
    transforms.ToTensor(),
    image_normalize()
])

image_folder_path = 'S:/data/coco/128/val/images'
label_folder_path = 'S:/data/coco/128/val/conds'
dataset = CustomDataset(image_folder_path, label_folder_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

parser = argparse.ArgumentParser()

parser.add_argument("--config_file", type=str,
                    default='E:/ContourDiffusion_unet_coco2017/configs/COCO-stuff_128x128/ContourDiffusion_small.yaml')
known_args, unknown_args = parser.parse_known_args()

known_args = OmegaConf.create(known_args.__dict__)
cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
if unknown_args:
    unknown_args = OmegaConf.from_dotlist(unknown_args)
    cfg = OmegaConf.merge(cfg, unknown_args)

    # 创建模型
generator = build_model(cfg).to('cuda')
# generator.convert_to_fp16()
generator.load_state_dict(torch.load('F:/contour_weight/ContourDiffusion0010000.pt'))
discriminator = copy.deepcopy(generator).to('cuda')

optimizer1 = torch.optim.AdamW(generator.parameters(),
                               lr=1e-04,
                               betas=(0.9, 0.999),
                               weight_decay=0.01,
                               eps=1e-8)
optimizer2 = torch.optim.AdamW(discriminator.parameters(),
                               lr=1e-04,
                               betas=(0.0, 0.999),
                               weight_decay=0.01,
                               eps=1e-8)
criterion = torch.nn.MSELoss()

used_condition_types = [
    'obj_description', 'obj_contour', 'is_valid_obj', 'prompt', 'mask', 'mask_'
]

scheduler = build_diffusion(cfg)


def update(data):
    batch, dictionary = data
    batch_size = batch.shape[0] * 3
    batch = batch.to(dtype=torch.float32).reshape(batch_size, 3, 128, 128)
    obj_num = dictionary['obj_num']
    cond = {'obj_description': dictionary['obj_class'].to(dtype=torch.float32).reshape(batch_size, -1),
            'obj_contour': dictionary['obj_contour'].to(dtype=torch.float32).reshape(batch_size, 10, 192),
            'is_valid_obj': dictionary['is_valid_obj'].to(dtype=torch.float32).reshape(batch_size, 10),
            'obj_num': obj_num
            }

    mask, mask_ = get_mask(cond, 128)
    cond['mask'] = mask
    cond['mask_'] = mask_
    for i in range(0, batch.shape[0], 1):
        x_0 = batch[i: i + 1].to('cuda')
        micro_cond = {
            k: v[i: i + 1].to('cuda')
            for k, v in cond.items() if k in used_condition_types
        }

        noise = torch.randn_like(x_0, dtype=torch.float32)
        noise_step = torch.tensor([max(0, random.randint(1, 999) - 250)]).to('cuda')
        fix_step = torch.tensor([250]).to('cuda')
        x_t_1 = scheduler.q_sample(x_0, noise_step, noise)
        x_t = scheduler.q_sample(x_t_1, fix_step, noise)
        # noise_pred = generator(x_t, noise_step, **micro_cond)
        x_0_pred = scheduler.ddim_sample(generator, x_t, noise_step, model_kwargs=micro_cond)['pred_xstart']

        x_t_1_pred = scheduler.q_sample(x_0_pred, noise_step, noise)

        Adversarial_loss = torch.tensor(
            -torch.log(discriminator(x_t_1, noise_step, **micro_cond, flag=False)) - torch.log(
                1 - discriminator(x_t_1_pred, noise_step, **micro_cond, flag=False)), requires_grad=True)
        print(f'Adversarial_loss=={Adversarial_loss}')
        Adversarial_loss.backward()
        if i == batch_size - 1:
            optimizer2.step()
            optimizer2.zero_grad()

        loss = torch.tensor(
            -torch.log(discriminator(x_t_1_pred, noise_step, **micro_cond, flag=False)) + criterion(x_0,
                                                                                                    x_0_pred),
            requires_grad=True)
        print(f'loss=={loss}')
        loss.backward()
        if i == batch_size - 1:
            optimizer1.step()
            optimizer1.zero_grad()

    return


def train():
    generator.train()
    discriminator.train()
    for j in range(epoch):
        print(f'epoch=={j + 1}')
        for i, data in enumerate(dataloader):
            update(data)


train()
torch.save(generator.state_dict(), f'F:/models-weight/my_model_weight/Unet_weight/generator.pt')
torch.save(discriminator.state_dict(), f'F:/models-weight/my_model_weight/Unet_weight/discriminator.pt')
