"""make variations of input image"""

import os, sys, glob
import PIL
import torch
import numpy as np
import einops
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
from pathlib import Path

from skimage.color import rgb2gray
from modules.feature_removal.detectors import canny

controlnet_path = 'sources/ControlNet'
# Add ControlNet dir to path, so references to module 'ldm' still work.
sys.path.append(controlnet_path)

sys.path.append(os.path.dirname(sys.path[0]))

from instldm.util import instantiate_from_config
from controlled_inst.ddim import DDIMSampler

# ControlNet imports
from cldm.model import load_state_dict
from annotator.util import HWC3
from annotator.canny import CannyDetector

apply_canny = CannyDetector()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    
    sd = load_state_dict(ckpt)
    if "global_step" in sd:
        print(f"Global Step: {sd['global_step']}")

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model

def get_state_dict_from_checkpoint(ckpt):
    print(f"Loading model state dict from {ckpt}.")
    return load_state_dict(ckpt)

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((256, 256), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

model = None
sampler = None

def main(prompt = '', content_image_path = '', style_image_path='',ddim_steps = 50,strength = 0.5, model = None, seed=42, outdir_name='', scale=10.0, guess_mode = False, controlled = False, control_only = False):
    ddim_eta=0.0
    n_iter=1
    C=4
    f=8
    n_samples=1
    n_rows=0
    scale=scale
    
    precision="autocast"
    outdir="outputs/img2img-samples/"
    if outdir_name:
        outdir += outdir_name
    else:
        outdir += str(seed)+'/'+str(ddim_steps)
    seed_everything(seed)


    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) + 10
    
    style_name = style_image_path.split('/')[-1].split('.')[0]
    style_image = load_img(style_image_path).to(device)
    style_image = repeat(style_image, '1 ... -> b ...', b=batch_size)
    style_latent = model.get_first_stage_encoding(model.encode_first_stage(style_image))  # move to latent space

    content_name =  content_image_path.split('/')[-1].split('.')[0]
    content_image = load_img(content_image_path).to(device)

    if control_only:
        content_name += '--control_only'
    
    content_image = repeat(content_image, '1 ... -> b ...', b=batch_size)

    if controlled:
        canny_image = (content_image[0].permute(1,2,0) + 1) / 2
        canny_image_uint8 = (canny_image.cpu().numpy() * 255).astype(np.uint8)
        canny_image = rgb2gray(canny_image_uint8)
        canny_image = canny(canny_image).astype(np.uint8) * 255
        canny_image = HWC3(canny_image)

        control = torch.from_numpy(canny_image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(n_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    if control_only:
        random_noise_image = np.random.normal(0, 1, (256, 256, 3)).astype(np.float32)
        content_image = (random_noise_image - np.min(random_noise_image)) / (np.max(random_noise_image) - np.min(random_noise_image)) # Map noise between -1 and 1
        content_image = content_image[None].transpose(0, 3, 1, 2)
        content_image = (torch.from_numpy(content_image) * 2 - 1).to(device)
    content_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))  # move to latent space
    init_latent = content_latent

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            if controlled:
                                uc = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning(batch_size * [""], style_image)]}
                            else:
                                uc = model.get_learned_conditioning(batch_size * [""], style_image)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        if controlled:
                            c = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(prompts, style_image)]}
                        else:
                            c = model.get_learned_conditioning(prompts, style_image)

                        # img2img

                        # stochastic inversion
                        t_enc = int(strength * 1000)
                        x_noisy = model.q_sample(x_start=init_latent, t=torch.tensor([t_enc]*batch_size).to(device))

                        if controlled:
                            model_output = model.apply_ldm(x_noisy, torch.tensor([t_enc]*batch_size).to(device), c)
                        else:
                            model_output = model.apply_model(x_noisy, torch.tensor([t_enc]*batch_size).to(device), c)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device),\
                                                          noise = model_output, use_original_steps = True)
            
                        if controlled:
                            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
                            
                        t_enc = int(strength * ddim_steps)
                        samples = sampler.decode(z_enc, c, t_enc, 
                                                unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)

                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            base_count += 1
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                output = Image.fromarray(grid.astype(np.uint8))
                output.save(
                    os.path.join(
                        outpath, 
                        "--".join([
                            content_name, 
                            style_name, 
                            'str' + str(strength),
                            'scale' + str(scale),
                            str(seed),
                            str(ddim_steps), 
                            str(time.time())
                        ]) + '.jpg'
                    )
                )
                
                grid_count += 1

                toc = time.time()
    return output


"""
TESTING/INFERENCE
"""
from config import GlobalConfig, TestConfig
from pathlib import Path

def run_test(test_config: TestConfig, run_directory: str):
    GlobalConfig.set(test_config)
    global model

    model = model.to('cpu')
    model.embedding_manager.load(test_config.embedding_path)
    model = model.to(device)
    
    for guidance_scale in test_config.guidance_scales:
        for strength in test_config.strengths:
            for content_image_path in test_config.content_image_paths:
                output_path = str(Path(
                    run_directory, 
                    test_config.test_name, 
                    Path(test_config.style_image_path).name, 
                    Path(content_image_path).name
                ))

                main(
                    prompt = test_config.prompt,
                    content_image_path = content_image_path,
                    style_image_path = test_config.style_image_path,
                    outdir_name = output_path,
                    ddim_steps = 50,
                    strength = strength,
                    scale = guidance_scale,
                    model = model,
                    seed = 23,
                    controlled = test_config.controlled,
                    control_only= test_config.control_only
                )


if __name__ == '__main__':
    from config.test.current import Tests

    run_directory = time.strftime('%Y-%m-%d_%H-%M')

    last_checkpoint = None
    last_model_config = None
    for test_config in Tests().test_configs:
        if not test_config.sd_checkpoint:
            print('A Stable Diffusion checkpoint needs to be provided. Skipping test "' + test_config.test_name + '".')
            continue

        if not test_config.model_config:
            print('A model config needs to be provided. Skipping test "' + test_config.test_name + '".')
            continue
        
        if test_config.model_config != last_model_config:
            config = OmegaConf.load(f"{test_config.model_config}")
            model = load_model_from_config(config, f"{test_config.sd_checkpoint}")
            sampler = DDIMSampler(model)
        elif test_config.sd_checkpoint != last_checkpoint:
            sd = load_state_dict(test_config.sd_checkpoint)
            model.load_state_dict(sd, strict=False)

        run_test(test_config, run_directory)

        last_checkpoint = test_config.sd_checkpoint
        last_model_config = test_config.model_config