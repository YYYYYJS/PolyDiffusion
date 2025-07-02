import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from contour_diffusion.dataset.dataset import CustomDataset
from contour_diffusion.train_util import TrainLoop
from contour_diffusion.util import loopy
from contour_diffusion.dataset.util import image_normalize
from contour_diffusion.contour_diffusion_unet import build_model
from contour_diffusion.resample import build_schedule_sampler
from contour_diffusion.respace import build_diffusion
from omegaconf import OmegaConf
import torch


def main():
    parser = argparse.ArgumentParser()
    path = '../configs/COCO-stuff_128x128/ContourDiffusion_small.yaml'
    parser.add_argument("--config_file", type=str,
                        default=path)
    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    # 创建模型
    model = build_model(cfg)
    model.convert_to_fp16()

    # model.contour_encoder.obj_contour_embedding.half()
    model.load_state_dict(torch.load('G:/contour_weight_128/ContourDiffusion0000000y.pt'))

    model.to('cuda')  # 将模型移到GPU上，如果有的话
    model.contour_encoder.requires_grad_(False)
    model.time_embedding.requires_grad_(False)
    # model.input_blocks.requires_grad_(False)
    # model.middle_block.requires_grad_(False)
    # model.output_blocks.requires_grad_(False)

    # 创建扩散模型
    diffusion = build_diffusion(cfg)

    # 创建扩散进度采样器
    schedule_sampler = build_schedule_sampler(cfg, diffusion)

    transform = transforms.Compose([
        transforms.ToTensor(),
        image_normalize()
    ])

    image_folder_path = 'S:/data/coco/128/train/images'
    label_folder_path = 'S:/data/coco/128/train/conds'
    # 创建数据集实例
    dataset = CustomDataset(image_folder_path, label_folder_path, transform)
    batch_size = 256
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # 训练
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        schedule_sampler=schedule_sampler,
        data=loopy(data_loader),
        batch_size=cfg.data.parameters.train.batch_size,
        flag=False,
        **cfg.train
    )

    trainer.run_loop()


if __name__ == "__main__":
    main()
