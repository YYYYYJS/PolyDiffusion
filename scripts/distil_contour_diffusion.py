import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from diffusers import (

    DDPMScheduler,

)
from torchvision import transforms
import argparse

from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from contour_diffusion.dataset.util import image_normalize
from contour_diffusion.contour_diffusion_unet import build_model
from contour_diffusion.dataset.dataset import CustomDataset
from contour_diffusion.fp16_util import MixedPrecisionTrainer
from contour_diffusion.train_util import get_mask, generate_image_patch_boundary_points

obj_bbox_ = torch.tensor([0, 0, 1, 1], dtype=torch.float16)


def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data ** 2 / ((timestep / 0.1) ** 2 + sigma_data ** 2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def save(mp_trainer):
    def save_checkpoint(rate, params):
        state_dict = mp_trainer.master_params_to_state_dict(params)

        torch.save(state_dict, 'F:/contour_weight/unet.pt')

    save_checkpoint(0, mp_trainer.master_params)


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


accelerator = Accelerator(
    split_batches=True,
    # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
)
transform = transforms.Compose([
    transforms.ToTensor(),
    image_normalize()
])
image_folder_path = 'S:/data/coco/128/small_val/images'
label_folder_path = 'S:/data/coco/128/small_val/conds'
dataset = CustomDataset(image_folder_path, label_folder_path, transform=transform)
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
used_condition_types = [
    'obj_description', 'obj_contour', 'is_valid_obj', 'prompt', 'mask', 'obj_bbox'
]
obj_description = torch.tensor([0, 184, 184, 184, 184, 184, 184, 184, 184, 184]).int()
contour_ = torch.tensor(generate_image_patch_boundary_points([1])['boundary_points_resolution1']).view(1, 97, 2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str,
                        default='../configs/COCO-stuff_128x128/ContourDiffusion_small.yaml')
    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    teacher_unet = build_model(cfg).to('cuda')
    teacher_unet.convert_to_fp16()
    teacher_unet.contour_encoder.obj_contour_embedding.half()
    teacher_unet.load_state_dict(torch.load('F:/contour_weight/unet.pt'))
    target_unet = copy.deepcopy(teacher_unet).to('cuda')
    unet = copy.deepcopy(teacher_unet).to('cuda')
    unet.convert_to_fp16()
    unet.contour_encoder.obj_contour_embedding.half()
    teacher_unet.requires_grad_(False)
    teacher_unet.contour_encoder.obj_contour_embedding.half()
    noise_scheduler = DDPMScheduler.from_pretrained("F:/models-weight/stable-diffusion-models/v-1-5",
                                                    subfolder="scheduler"
                                                    )

    # The scheduler calculates the alpha and sigma schedule for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod).to('cuda')
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod).to('cuda')
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=50,
    ).to('cuda')
    mp_trainer = MixedPrecisionTrainer(
        model=unet,
        use_fp16=True,
        fp16_scale_growth=1e-3,
        only_update_parameters_that_require_grad=False
    )
    optimizer = torch.optim.AdamW(
        mp_trainer.master_params, lr=1e-5, weight_decay=0
    )
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=100,
    )
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    for j in range(1400):
        print(f'epoch=={j + 1}')
        for n, data in enumerate(dataloader):
            with accelerator.accumulate(unet):
                batch, dictionary = data
                batch_size = batch.shape[0]
                batch = batch.to(dtype=torch.float16).reshape(batch_size, 3, 128, 128)
                obj_num = dictionary['obj_num']
                cond = {'obj_description': dictionary['obj_class'].to(dtype=torch.float16).reshape(batch_size, -1),
                        'obj_contour': dictionary['obj_contour'].to(dtype=torch.float16).reshape(batch_size, 10, 97, 2),
                        'is_valid_obj': dictionary['is_valid_obj'].to(dtype=torch.float16).reshape(batch_size, 10),
                        'obj_bbox': dictionary['obj_bbox'].to(dtype=torch.float16).reshape(batch_size, 10, 4),
                        'obj_num': obj_num
                        }

                mask = get_mask(cond, 128)
                cond['mask'] = mask

                uncond = {'obj_description': obj_description.repeat(batch.shape[0], 1),
                          'obj_contour': torch.zeros_like(cond['obj_contour']),
                          'is_valid_obj': torch.zeros_like(cond['is_valid_obj']),
                          'obj_bbox': torch.zeros_like(cond['obj_bbox']),
                          'obj_num': obj_num,
                          'mask': mask

                          }
                uncond['obj_contour'][:, 0] = contour_
                uncond['is_valid_obj'][:, 0] = 1.0
                uncond['obj_bbox'][:, 0] = obj_bbox_.repeat(batch.shape[0], 1)
                latents = batch.to('cuda')

                for key, value in cond.items():
                    cond[key] = cond[key].to('cuda')
                for key, value in uncond.items():
                    uncond[key] = uncond[key].to('cuda')

                noise = torch.randn_like(latents, dtype=torch.float16)
                bsz = latents.shape[0]

                # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
                topk = noise_scheduler.config.num_train_timesteps // 50
                index = torch.randint(0, 50, (bsz,), device='cpu').long()
                start_timesteps = solver.ddim_timesteps[index].to('cuda')
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(timesteps)
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                # 20.4.5. Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
                noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

                # 20.4.6. Sample a random guidance scale w from U[w_min, w_max] and embed it
                w = (15 - 5) * torch.rand((bsz,)) + 5

                w_embedding = guidance_scale_embedding(w, embedding_dim=512)
                w_embedding = w_embedding.to(device=latents.device, dtype=torch.float32)
                w = w.reshape(bsz, 1, 1, 1)
                # Move to U-Net device and dtype
                w = w.to(device=latents.device, dtype=latents.dtype)

                # noise_pred = unet(latents, start_timesteps, timestep_cond=w_embedding, **micro_cond)
                noise_pred, _, _ = unet(noisy_model_input, start_timesteps, time_cond=w_embedding, **cond)
                noise_pred, _ = torch.split(noise_pred, 3, dim=1)

                pred_x_0 = predicted_origin(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )
                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

                with torch.no_grad():
                    cond_teacher_output, _, _ = teacher_unet(
                        noisy_model_input,
                        start_timesteps,
                        time_cond=w_embedding,
                        **cond
                    )
                    cond_teacher_output, _ = torch.split(cond_teacher_output, 3, dim=1)

                    cond_pred_x0 = predicted_origin(
                        cond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    # Get teacher model prediction on noisy_latents and unconditional embedding
                    uncond_teacher_output, _, _ = teacher_unet(
                        noisy_model_input,
                        start_timesteps,
                        time_cond=w_embedding,
                        **uncond
                    )
                    uncond_teacher_output, _ = torch.split(uncond_teacher_output, 3, dim=1)
                    uncond_pred_x0 = predicted_origin(
                        uncond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    # 20.4.11. Perform "CFG" to get x_prev estimate (using the LCM paper's CFG formulation)
                    pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                    pred_noise = cond_teacher_output + w * (cond_teacher_output - uncond_teacher_output)
                    index = index.to('cuda')
                    x_prev = solver.ddim_step(pred_x0, pred_noise, index).to(dtype=torch.float16)

                with torch.no_grad():
                    target_noise_pred, _, _ = target_unet(
                        x_prev,
                        timesteps,
                        time_cond=w_embedding,
                        **cond
                    )
                    target_noise_pred, _ = torch.split(target_noise_pred, 3, dim=1)
                    pred_x_0 = predicted_origin(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target = c_skip * x_prev + c_out * pred_x_0
                    # print(c_skip, x_prev, c_out, pred_x_0)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                print(loss)
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                mp_trainer.optimize(optimizer)
                lr_scheduler.step()
                optimizer.zero_grad()

    save(mp_trainer)

if __name__ == "__main__":
    main()
