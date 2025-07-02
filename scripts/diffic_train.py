import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from contour_diffusion.dataset.dataset import CustomDataset
from contour_diffusion.train_util import get_mask

from contour_diffusion.dataset.util import image_normalize

from contour_diffusion.resample import build_schedule_sampler
from contour_diffusion.respace import build_diffusion
import torch

from contour_diffusion.DiffiC import UViT
from contour_diffusion.contour_encoder import ContourTransformerEncoder


def main():
    used_condition_types = [
        'obj_description', 'obj_contour', 'is_valid_obj', 'prompt', 'mask', 'mask_'
    ]

    encoder = ContourTransformerEncoder(hidden_dim=160,
                                        output_dim=640,
                                        num_layers=4,
                                        num_heads=8,
                                        use_final_ln=True,
                                        use_positional_embedding=False,
                                        resolution_to_attention=[8],
                                        use_key_padding_mask=False, contour_length=10, mask_size_for_contour_object=32,
                                        used_condition_types=[
                                            'obj_description', 'obj_contour', 'is_valid_obj', 'prompt', 'mask', 'mask_'
                                        ], num_classes_for_contour_object=185)
    model = UViT(img_size=128, contour_encoder=encoder).to('cuda')

    batch_size = 56
    epoch = 10000
    max_grad_norm = 1.0
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

    diffusion = build_diffusion(cfg)
    schedule_sampler = build_schedule_sampler(cfg, diffusion)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-05,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.01,
                                  eps=1e-8)

    for j in range(epoch):
        print(f'epoch=={j + 1}')
        for i, data in enumerate(dataloader):
            batch, dictionary = data
            batch_size = batch.shape[0] * 3
            batch = batch.to('cuda', dtype=torch.float32).reshape(batch_size, 3, 128, 128)
            obj_num = dictionary['obj_num']
            cond = {'obj_description': dictionary['obj_class'].to('cuda', dtype=torch.float32).reshape(batch_size, -1),
                    'obj_contour': dictionary['obj_contour'].to('cuda', dtype=torch.float32).reshape(batch_size, 10,
                                                                                                     192),
                    'is_valid_obj': dictionary['is_valid_obj'].to('cuda', dtype=torch.float32).reshape(batch_size, 10),
                    'obj_num': obj_num
                    }

            mask, mask_ = get_mask(cond, 128)
            cond['mask'] = mask_.to('cuda')
            t, weights = schedule_sampler.sample(batch_size, 'cuda')
            loss = diffusion.training_losses(model=model, x_start=batch, t=t, model_kwargs=cond, flag=False)[
                'loss'].mean()
            print(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    main()
