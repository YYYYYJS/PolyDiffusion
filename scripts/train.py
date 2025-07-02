import argparse
import math

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from contour_diffusion.dataset.dataset import CustomDataset
from contour_diffusion.fp16_util import get_param_groups_and_shapes, make_master_params
from contour_diffusion.train_util import TrainLoop, get_mask
from contour_diffusion.util import loopy
from contour_diffusion.dataset.util import image_normalize
from contour_diffusion.contour_diffusion_unet import build_model
from contour_diffusion.resample import build_schedule_sampler
from contour_diffusion.respace import build_diffusion
import torch


def main():
    parser = argparse.ArgumentParser()
    path = 'E:/ContourDiffusion_unet_coco2017_relative_position_2/configs/COCO-stuff_128x128/ContourDiffusion_small.yaml'
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

    param_groups_and_shapes = get_param_groups_and_shapes(
        model.named_parameters()
    )
    master_params = make_master_params(param_groups_and_shapes)
    model.convert_to_fp16()

    model.load_state_dict(torch.load('G:/contour_weight_128/ContourDiffusion0000013.pt'))
    model.to('cuda')  # 将模型移到GPU上，如果有的话

    # model.contour_encoder.requires_grad_(False)
    # model.time_embedding.requires_grad_(False)

    # 创建扩散模型
    diffusion = build_diffusion(cfg)
    schedule_sampler = build_schedule_sampler(cfg, diffusion)
    transform = transforms.Compose([
        transforms.ToTensor(),
        image_normalize()
    ])

    image_folder_path = 'S:/data/coco/128/small/images'
    label_folder_path = 'S:/data/coco/128/small/conds'
    # 创建数据集实例
    dataset = CustomDataset(image_folder_path, label_folder_path, transform=transform)
    batch_size = 8
    epoch = 200
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(master_params,
                                  lr=1e-02,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.01,
                                  eps=1e-8)

    loss_scale = 100
    count = 1
    for j in range(epoch):
        print(f'epoch=={j + 1}')
        for k, data in enumerate(data_loader):
            batch, dictionary = data

            batch = batch.to('cuda', dtype=torch.float16).reshape(batch_size, 3, 128, 128)
            obj_num = dictionary['obj_num']
            cond = {'obj_description': dictionary['obj_class'].to('cuda', dtype=torch.float16).reshape(batch_size, -1),
                    'obj_contour': dictionary['obj_contour'].to('cuda', dtype=torch.float16).reshape(batch_size, 10, 97,
                                                                                                     2
                                                                                                     ),
                    'obj_bbox': dictionary['obj_bbox'].to('cuda', dtype=torch.float16).reshape(batch_size, 10, 4),
                    'is_valid_obj': dictionary['is_valid_obj'].to('cuda', dtype=torch.float16).reshape(batch_size, 10),
                    'obj_num': obj_num
                    }

            micro_batch_size = 8
            accum_iter = math.ceil(batch.shape[0] / micro_batch_size)
            for i in range(0, batch.shape[0], micro_batch_size):

                if i + micro_batch_size < batch.shape[0]:
                    micro = batch[i: i + micro_batch_size].to('cuda')

                    micro_cond = {
                        k: v[i: i + micro_batch_size].to('cuda')
                        for k, v in cond.items() if k in model.contour_encoder.used_condition_types
                    }
                else:

                    micro = batch[i:].to('cuda')

                    micro_cond = {
                        k: v[i:].to('cuda')
                        for k, v in cond.items() if k in model.contour_encoder.used_condition_types
                    }
                size = batch.shape[2]
                mask = get_mask(micro_cond, size)
                micro_cond['mask'] = mask.to('cuda')
                micro_cond['train_step'] = k + 1

                t, weights = schedule_sampler.sample(micro.shape[0], 'cuda')
                loss = diffusion.training_losses(model=model, x_start=micro, t=t, model_kwargs=micro_cond)[
                    'loss'].mean()
                loss = loss / accum_iter
                print(loss)
                (loss * loss_scale).backward()



            '''
                        for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name} grad norm: {param.grad.norm().item():.4f}")
            print('--------------')
            
            '''
            optimizer.step()
            optimizer.zero_grad()
            count += 1

    torch.save(model.state_dict(), f'G:/contour_weight_128/ContourDiffusion{count}.pt')


if __name__ == "__main__":
    main()
