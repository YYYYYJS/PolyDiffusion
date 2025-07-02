"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import pickle

import argparse

import torch
import torch as th
from omegaconf import OmegaConf
from PIL import Image
from contour_diffusion.contour_diffusion_unet import build_model
from contour_diffusion.contour_encoder import generate_image_patch_boundary_points
from contour_diffusion.util import fix_seed
from repositories.dpm_solver.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper
import numpy as np
from contour_diffusion.respace import build_diffusion

object_name_to_idx = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7,
                      'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
                      'parking meter': 14,
                      'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21,
                      'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 'umbrella': 28,
                      'handbag': 31,
                      'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37,
                      'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42,
                      'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49,
                      'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56,
                      'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63,
                      'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73,
                      'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79,
                      'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86,
                      'scissors': 87,
                      'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90, 'banner': 92, 'blanket': 93, 'branch': 94,
                      'bridge': 95, 'building-other': 96, 'bush': 97, 'cabinet': 98, 'cage': 99, 'cardboard': 100,
                      'carpet': 101, 'ceiling-other': 102, 'ceiling-tile': 103, 'cloth': 104, 'clothes': 105,
                      'clouds': 106, 'counter': 107, 'cupboard': 108, 'curtain': 109, 'desk-stuff': 110, 'dirt': 111,
                      'door-stuff': 112, 'fence': 113, 'floor-marble': 114, 'floor-other': 115, 'floor-stone': 116,
                      'floor-tile': 117, 'floor-wood': 118, 'flower': 119, 'fog': 120, 'food-other': 121, 'fruit': 122,
                      'furniture-other': 123, 'grass': 124, 'gravel': 125, 'ground-other': 126, 'hill': 127,
                      'house': 128, 'leaves': 129, 'light': 130, 'mat': 131, 'metal': 132, 'mirror-stuff': 133,
                      'moss': 134,
                      'mountain': 135, 'mud': 136, 'napkin': 137, 'net': 138, 'paper': 139, 'pavement': 140,
                      'pillow': 141, 'plant-other': 142, 'plastic': 143, 'platform': 144, 'playingfield': 145,
                      'railing': 146,
                      'railroad': 147, 'river': 148, 'road': 149, 'rock': 150, 'roof': 151, 'rug': 152, 'salad': 153,
                      'sand': 154, 'sea': 155, 'shelf': 156, 'sky-other': 157, 'skyscraper': 158, 'snow': 159,
                      'solid-other': 160, 'stairs': 161, 'stone': 162, 'straw': 163, 'structural-other': 164,
                      'table': 165, 'tent': 166, 'textile-other': 167, 'towel': 168, 'tree': 169, 'vegetable': 170,
                      'wall-brick': 171, 'wall-concrete': 172, 'wall-other': 173, 'wall-panel': 174, 'wall-stone': 175,
                      'wall-tile': 176, 'wall-wood': 177, 'water-other': 178, 'waterdrops': 179,
                      'window-blind': 180, 'window-other': 181, 'wood': 182, 'other': 183, '__image__': 0,
                      '__null__': 184}

contour_ = torch.tensor(generate_image_patch_boundary_points([1])['boundary_points_resolution1']).view(-1, 2)


@torch.no_grad()
def layout_to_image_generation(cfg, model_fn, noise_schedule, custom_layout_dict):
    print(custom_layout_dict)

    model_wrapper(
        model_fn,
        noise_schedule,
        is_cond_classifier=False,
        total_N=1000,
        model_kwargs=custom_layout_dict
    )
    for key in custom_layout_dict.keys():
        if key != 'obj_num':
            custom_layout_dict[key] = custom_layout_dict[key].cuda()

    bsize = custom_layout_dict['obj_bbox'].shape[0]

    diffusion = build_diffusion(cfg, timestep_respacing=cfg.sample.timestep_respacing)
    x_T = th.randn((bsize, 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size)).cuda()

    sample_fn = (diffusion.p_sample_loop if cfg.sample.sample_method == 'ddpm' else diffusion.ddim_sample_loop)
    all_results = sample_fn(
        model_fn, (x_T.shape[0], 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size),
        clip_denoised=cfg.sample.clip_denoised, model_kwargs=custom_layout_dict, cond_fn=None, device='cuda'
    )  # (B, 3, H, W)
    last_result = all_results[-1]
    sample = last_result['sample'].clamp(-1, 1)

    img_list = []
    for i in range(bsize):
        generate_img = np.array(sample[i].cpu().permute(1, 2, 0) * 127.5 + 127.5, dtype=np.uint8)
        img_list.append(generate_img)
    # generate_img = np.transpose(generate_img, (1,0,2))

    print("sampling complete")

    return img_list


@torch.no_grad()
def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str,
                        default='../configs/COCO-stuff_128x128/ContourDiffusion_small.yaml')
    parser.add_argument("--share", action='store_true')

    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    print("creating generator...")
    model = build_model(cfg)
    model.convert_to_fp16()

    cfg.sample.pretrained_model_path = 'G:/contour_weight_128/ContourDiffusion201.pt'
    if cfg.sample.pretrained_model_path:
        print("loading generator from {}".format(cfg.sample.pretrained_model_path))
        checkpoint = torch.load(cfg.sample.pretrained_model_path, map_location="cpu")

        try:
            model.load_state_dict(checkpoint, strict=True)
            print('successfully load the entire generator')
        except:
            print('not successfully load the entire generator, try to load part of generator')

            model.load_state_dict(checkpoint, strict=False)

    model.cuda()
    if cfg.sample.use_fp16:
        model.convert_to_fp16()
        model.contour_encoder.obj_contour_embedding.half()
    model.eval()

    def model_fn(x, t, obj_description=None, obj_contour=None, obj_mask=None, is_valid_obj=None, prompt=None, **kwargs):
        assert obj_description is not None
        assert obj_contour is not None
        bsize = x.shape[0]

        cond_image, cond_extra_outputs, _ = model(
            x, t,
            obj_description=obj_description, obj_contour=obj_contour, obj_mask=obj_mask, prompt=prompt,
            is_valid_obj=is_valid_obj, **kwargs
        )
        cond_mean, cond_variance = th.chunk(cond_image, 2, dim=1)

        obj_description = torch.tensor([[0, 184, 184, 184, 184, 184, 184, 184, 184, 184]]).to('cuda')
        obj_description = obj_description.repeat(bsize, 1).view(bsize, -1)

        obj_contour = torch.zeros_like(obj_contour)
        obj_contour[:, 0] = contour_

        is_valid_obj = th.zeros_like(is_valid_obj)
        is_valid_obj[:, 0] = 1.0

        uncond_image, uncond_extra_outputs, _ = model(
            x, t,
            obj_description=obj_description, obj_contour=obj_contour, obj_mask=obj_mask, prompt=prompt,
            is_valid_obj=is_valid_obj, **kwargs
        )
        uncond_mean, uncond_variance = th.chunk(uncond_image, 2, dim=1)

        mean = cond_mean + cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

        if cfg.sample.sample_method in ['ddpm', 'ddim']:
            return [th.cat([mean, cond_variance], dim=1), cond_extra_outputs]
        else:
            return mean

    print("creating diffusion...")

    noise_schedule = NoiseScheduleVP(schedule='linear')

    print('sample method = {}'.format(cfg.sample.sample_method))
    print("sampling...")

    return cfg, model_fn, noise_schedule


if __name__ == "__main__":
    path = 'G:/demo4'
    cfg, model_fn, noise_schedule = init()
    with open('my_dict.pkl', 'rb') as file:
        custom_layout_dict = pickle.load(file)

    bsize = custom_layout_dict['obj_bbox'].shape[0]
    samper_num = 5
    step_l = 16
    for j in range(samper_num):
        for k in range(0, bsize, step_l):
            cond = {
                'obj_description': custom_layout_dict['obj_description'][k:k + step_l, :],
                'obj_contour': custom_layout_dict['obj_contour'][k:k + step_l, :, :, :],
                'is_valid_obj': custom_layout_dict['is_valid_obj'][k:k + step_l, :],
                'mask': custom_layout_dict['mask'][k:k + step_l, :, :],
                'obj_bbox': custom_layout_dict['obj_bbox'][k:k + step_l, :, :]
            }

            cfg.sample.timestep_respacing[0] = 30
            # cfg.sample.sample_method = 'dpm_solver'
            cfg.classifier_free_scale = 7

            # cond['obj_description'][4][1] = 184
            # cond['obj_contour'][4:5, 1:2, :, :] = torch.zeros((1, 1, 97, 2))
            # print(cond['obj_contour'][0:1, 1:2, :, :])
            img_list = layout_to_image_generation(cfg, model_fn, noise_schedule, cond)

            for i in range(step_l):
                image = Image.fromarray(img_list[i])
                image.save(f"{path}/{j}_{k + i}.png")
