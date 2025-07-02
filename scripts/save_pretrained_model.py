import argparse
from omegaconf import OmegaConf

from contour_diffusion.contour_diffusion_unet import build_model

import torch

parser = argparse.ArgumentParser()

parser.add_argument("--config_file", type=str,
                    default='E:/LayoutDiffusion-master/configs/COCO-stuff_256x256/ContourDiffusion_large.yaml')
known_args, unknown_args = parser.parse_known_args()

known_args = OmegaConf.create(known_args.__dict__)
cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
if unknown_args:
    unknown_args = OmegaConf.from_dotlist(unknown_args)
    cfg = OmegaConf.merge(cfg, unknown_args)
print(OmegaConf.to_yaml(cfg))

# 创建模型
model = build_model(cfg)
checkpoint = torch.load('E:/pre_trained/COCO-stuff_256x256_LayoutDiffusion_large_ema_1150000.pt')
model.load_state_dict(checkpoint)
model.to('cuda')  # 将模型移到GPU上，如果有的话

# 遍历模型中的每个子模块并保存其参数
for name, module in model.named_children():
    # 生成文件名
    file_name = f'F:/layout_weight/{name}_parameters.pt'
    torch.save(module.state_dict(), file_name)
