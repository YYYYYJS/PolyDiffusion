from PIL import Image
import argparse
from omegaconf import OmegaConf
from torchvision import transforms
import os
from contour_diffusion.dataset.data_loader import build_loaders
import torch
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--config_file", type=str,
                    default='E:\ContourDiffusion_unet_coco2017_relative_position\configs\COCO-stuff_128x128\ContourDiffusion_small.yaml')
known_args, unknown_args = parser.parse_known_args()

known_args = OmegaConf.create(known_args.__dict__)
cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
if unknown_args:
    unknown_args = OmegaConf.from_dotlist(unknown_args)
    cfg = OmegaConf.merge(cfg, unknown_args)

train_loader = build_loaders(cfg, mode='train')

img_save_path = 'S:/data/coco/128/train/images'
cond_save_path = 'S:/data/coco/128/train/conds'

if __name__ == '__main__':
    for i, s in enumerate(train_loader):
        tensor_image, cond = s
        tensor_to_pil = transforms.ToPILImage()
        pil_image = tensor_to_pil(tensor_image[0])
        save_path1 = os.path.join(img_save_path, f'{i +0}.png')
        pil_image.save(save_path1)

        save_path2 = os.path.join(cond_save_path, f'{i + 0}.pkl')
        with open(save_path2, 'wb') as file:
            pickle.dump(cond, file)


