
from contour_diffusion.SingDiffusionPipeline import SingDiffusionPipeline
from diffusers import DDIMScheduler

sd15_path = "G:/models-weight/stable-diffusion-models/v-1-5"
sing_diffusion_path = 'S:/singdiffusion'
scheduler = DDIMScheduler.from_pretrained(sd15_path, subfolder="scheduler")
pipe = SingDiffusionPipeline(sd15_path=sd15_path, sing_diffusion_path=sing_diffusion_path, scheduler=scheduler)
prompt = ['a  cat']
pipe(prompt=prompt, prompt_uncond=None, height=16, width=16)
