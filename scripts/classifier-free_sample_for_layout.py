"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import json
import os
import time

import imageio
import torch
import torch as th
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision import utils

from contour_diffusion import dist_util, logger
from contour_diffusion.dataset.data_loader import build_loaders
from contour_diffusion.contour_diffusion_unet import build_model
from contour_diffusion.respace import build_diffusion
from contour_diffusion.util import fix_seed
from contour_diffusion.dataset.util import image_unnormalize_batch, get_cropped_image
from dpm_solver.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

class_ = torch.load('class.pt').to('cuda', dtype=torch.float32)
class_ = torch.rand_like(class_).to('cuda', dtype=torch.float32)
bbox_ = torch.load('bbox.pt').to('cuda', dtype=torch.float32)
#bbox_ = torch.rand_like(bbox_).to('cuda', dtype=torch.float32)


def imageio_save_image(img_tensor, path):
    '''
    :param img_tensor: (C, H, W) torch.Tensor
    :param path:
    :param args:
    :param kwargs:
    :return:
    '''
    # tmp_img = image_unnormalize_batch(img_tensor).clamp(0.0, 1.0)
    tmp_img = img_tensor
    imageio.imsave(
        uri=path,
        im=tmp_img.cpu().detach().numpy().transpose(1, 2, 0),  # (H, W, C) numpy
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config_file", type=str,
                        default='E:/LayoutDiffusion-master/configs/COCO-stuff_256x256/ContourDiffusion_large.yaml')

    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    cfg.sample.pretrained_model_path = 'F:/layout_weight/model0030000.pt'
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    # dist_util.setup_dist(local_rank=cfg.local_rank)

    # if cfg.sample.fix_seed:
    #    fix_seed()

    data_loader = build_loaders(cfg, mode='test')

    total_num_samples = len(data_loader.dataset)
    log_dir = os.path.join(cfg.sample.log_root, 'conditional_{}'.format(cfg.sample.timestep_respacing),
                           'sample{}x{}'.format(total_num_samples, int(cfg.sample.sample_times)),
                           cfg.sample.sample_suffix)
    # logger.configure(dir=log_dir)
    # logger.log('current rank == {}, total_num = {}, \n, {}'.format(dist.get_rank(), dist.get_world_size(), cfg))
    # logger.log(OmegaConf.to_yaml(cfg))

    # logger.log("creating generator...")
    model = build_model(cfg)
    model.to('cuda')
    # logger.log(generator)

    if cfg.sample.pretrained_model_path:
        # logger.log("loading generator from {}".format(cfg.sample.pretrained_model_path))
        checkpoint = dist_util.load_state_dict(cfg.sample.pretrained_model_path, map_location="cpu")
        # if 'layout_encoder.obj_box_embedding.weight' in list(checkpoint.keys()):
        #     logger.log('pop layout_encoder.obj_box_embedding.weight')
        #     return
        #     checkpoint['layout_encoder.obj_bbox_embedding.weight'] = checkpoint.pop('layout_encoder.obj_box_embedding.weight')
        #     checkpoint['layout_encoder.obj_bbox_embedding.bias'] = checkpoint.pop('layout_encoder.obj_box_embedding.bias')
        try:
            model.load_state_dict(checkpoint, strict=True)
            # logger.log('successfully load the entire generator')
        except:
            # logger.log('not successfully load the entire generator, try to load part of generator')
            model.load_state_dict(checkpoint, strict=False)

    model.to('cuda')
    if cfg.sample.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, text=None, **kwargs):
        assert obj_class is not None
        assert obj_bbox is not None

        cond_image, cond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj, text=text
        )
        cond_mean, cond_variance = th.chunk(cond_image, 2, dim=1)

        obj_class = th.ones_like(obj_class)
        obj_class[:, 1:, :] = class_[1].repeat(9, 1, 1)
        obj_class[:, 0] = class_[0]

        obj_bbox = th.zeros_like(obj_bbox)
        obj_bbox[:, 0] = bbox_

        is_valid_obj = th.zeros_like(obj_class)
        is_valid_obj[:, 0] = 1.0
        text[:, :, :] = class_[1]

        if obj_mask is not None:
            obj_mask = th.zeros_like(obj_mask)
            obj_mask[:, 0] = th.ones(obj_mask.shape[-2:])

        uncond_image, uncond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj, text=text
        )
        uncond_mean, uncond_variance = th.chunk(uncond_image, 2, dim=1)
        cfg.sample.classifier_free_scale = 0.18725
        mean = cond_mean + cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

        if cfg.sample.sample_method in ['ddpm', 'ddim']:
            return [th.cat([mean, cond_variance], dim=1), cond_extra_outputs]
        else:
            return mean

    dir_names = ['generated_imgs', 'real_imgs', 'gt_annotations']
    if cfg.sample.save_cropped_images:
        dir_names.extend(['generated_cropped_imgs', 'real_cropped_imgs'])
    if cfg.sample.save_images_with_bboxs:
        dir_names.extend(['real_imgs_with_bboxs', 'generated_imgs_with_bboxs', 'generated_images_with_each_bbox'])
    if cfg.sample.save_sequence_of_obj_imgs:
        dir_names.extend(['obj_imgs_from_unresized_gt_imgs', 'obj_imgs_from_resized_gt_imgs'])

    for dir_name in dir_names:
        os.makedirs(os.path.join(log_dir, dir_name), exist_ok=True)

    if cfg.sample.save_cropped_images:
        if cfg.data.type == 'COCO-stuff':
            for class_id in range(1, 183):  # 1-182
                if class_id not in [12, 183, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:
                    os.makedirs(os.path.join(log_dir, 'generated_cropped_imgs', str(class_id)), exist_ok=True)
                    os.makedirs(os.path.join(log_dir, 'real_cropped_imgs', str(class_id)), exist_ok=True)
        elif cfg.data.type == 'VG':
            for class_id in range(1, 179):  # 1-178
                os.makedirs(os.path.join(log_dir, 'generated_cropped_imgs', str(class_id)), exist_ok=True)
                os.makedirs(os.path.join(log_dir, 'real_cropped_imgs', str(class_id)), exist_ok=True)
        else:
            raise NotImplementedError

    # logger.log("creating diffusion...")
    if cfg.sample.sample_method == 'dpm_solver':
        noise_schedule = NoiseScheduleVP(schedule='linear')
    elif cfg.sample.sample_method in ['ddpm', 'ddim']:
        diffusion = build_diffusion(cfg, timestep_respacing=cfg.sample.timestep_respacing)
    else:
        raise NotImplementedError

    # logger.log('sample method = {}'.format(cfg.sample.sample_method))
    # logger.log("sampling...")
    start_time = time.time()
    total_time = 0.0

    for batch_idx, batch in enumerate(data_loader):
        total_time += (time.time() - start_time)

        # print('rank={}, batch_id={}'.format(dist.get_rank(), batch_idx))

        imgs, cond = batch
        imgs = imgs.to('cuda')
        model_kwargs = {
            'obj_class': cond['obj_class'].to('cuda'),
            'obj_bbox': cond['obj_bbox'].to('cuda'),
            'is_valid_obj': cond['is_valid_obj'].to('cuda')
        }
        if 'obj_mask' in cfg.data.parameters.used_condition_types:
            model_kwargs['obj_mask']: cond['obj_mask'].to('cuda')

        for sample_idx in range(cfg.sample.sample_times):
            start_time = time.time()
            if cfg.sample.sample_method == 'dpm_solver':
                wrappered_model_fn = model_wrapper(
                    model_fn,
                    noise_schedule,
                    is_cond_classifier=False,
                    total_N=1000,
                    model_kwargs=model_kwargs
                )

                dpm_solver = DPM_Solver(wrappered_model_fn, noise_schedule)

                x_T = th.randn((imgs.shape[0], 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size),
                               device=dist_util.dev())
                sample = dpm_solver.sample(
                    x_T,
                    steps=int(cfg.sample.timestep_respacing[0]),
                    eps=float(cfg.sample.eps),
                    adaptive_step_size=cfg.sample.adaptive_step_size,
                    fast_version=cfg.sample.fast_version,
                    clip_denoised=False,
                    rtol=cfg.sample.rtol
                )  # (B, 3, H, W)
                sample = sample.clamp(-1, 1)
            elif cfg.sample.sample_method in ['ddpm', 'ddim']:
                file_path = 'my_dict.pt'
                model_kwargs = torch.load(file_path)
                for key, value in model_kwargs.items():
                    model_kwargs[key] = value.to('cuda')  # 将张量
                sample_fn = (
                    diffusion.p_sample_loop if cfg.sample.sample_method == 'ddpm' else diffusion.ddim_sample_loop)
                all_results = sample_fn(
                    model_fn, (imgs.shape[0], 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size),
                    clip_denoised=cfg.sample.clip_denoised, model_kwargs=model_kwargs, cond_fn=None, device='cuda'
                )  # (B, 3, H, W)
                last_result = all_results[-1]
                sample = last_result['sample'].clamp(-1, 1)
                for i in range(8):
                    imageio_save_image(sample[i], f'{i}.png')
            else:
                raise NotImplementedError

    # logger.log("sampling complete")


if __name__ == "__main__":
    main()
