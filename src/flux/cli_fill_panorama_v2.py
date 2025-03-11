import os
import re
import time
from dataclasses import dataclass
import numpy as np
import math
import torchvision
from glob import iglob
import io

import torch
from fire import Fire
from PIL import Image
from transformers import pipeline

from flux.sampling import denoise, get_noise, get_schedule, prepare_fill, unpack
from flux.util import (
    configs,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    save_image
)

from flux.gpu_split import split_flux_model_to_gpus, GPUSplitConfig


def display_tensor(tensor, title=None):
    """Display a tensor as an image using kitty graphics protocol"""
    import base64
    import sys

    # Convert tensor to PIL Image
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor + 1) / 2  # Convert from [-1,1] to [0,1]
    pil_img = torchvision.transforms.ToPILImage()(tensor)

    # Convert PIL image to PNG bytes
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_data = buffer.getvalue()

    # Encode in base64
    img_base64 = base64.b64encode(img_data).decode('ascii')

    # Print title if provided
    if title:
        print(f"\n--- {title} ---")

    # Send kitty graphics command
    sys.stdout.buffer.write(b'\033_Ga=T,f=100,s=' + str(pil_img.size[0]).encode(
    ) + b',v=' + str(pil_img.size[1]).encode() + b';' + img_base64.encode() + b'\033\\')
    sys.stdout.buffer.flush()
    print()  # Add newline after image


def create_sinusoidal_pos_enc(h: int, w: int,
                              pos_enc_channels: int = 4,
                              pos_enc_cross: int = 0) -> torch.Tensor:
    assert pos_enc_channels in [
        4, 8, 12], "pos_enc_channels must be 4, 8, or 12"
    assert pos_enc_cross in [0, 1, 2], "pos_enc_cross must be 0, 1, or 2"
    assert pos_enc_channels >= 2 * \
        pos_enc_cross, "pos_enc_channels must be >= 2 * pos_enc_cross"
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    pos_enc = torch.zeros((pos_enc_channels, h, w))
    pairs_per_dim = (pos_enc_channels - 2 * pos_enc_cross) // 4
    freqs = torch.exp(torch.linspace(
        math.log(1.0), math.log(10.0), pairs_per_dim))
    for i in range(pairs_per_dim):
        pos_enc[2*i] = torch.sin(X * np.pi * freqs[i])
        pos_enc[2*i + 1] = torch.cos(X * np.pi * freqs[i])
    y_start = 2 * pairs_per_dim
    for i in range(pairs_per_dim):
        pos_enc[y_start + 2*i] = torch.sin(Y * np.pi * freqs[i])
        pos_enc[y_start + 2*i + 1] = torch.cos(Y * np.pi * freqs[i])
    if pos_enc_cross > 0:
        cross_start = 4 * pairs_per_dim
        pos_enc[cross_start] = torch.sin(X * Y * np.pi * 4.2)
        pos_enc[cross_start + 1] = torch.cos(X * Y * np.pi * 4.2)
        if pos_enc_cross == 2:
            pos_enc[cross_start + 2] = torch.sin((X**2 + Y**2) * np.pi * 2.5)
            pos_enc[cross_start + 3] = torch.cos((X**2 + Y**2) * np.pi * 2.5)
    return pos_enc


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    img_cond_path: str
    img_mask_path: str


@torch.inference_mode()
def main(
    lora_path=None,
    seed: int | None = None,
    prompt: str = "a photo of sks",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 50,
    loop: bool = False,
    guidance: float = 30.0,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/cup.png",
    img_mask_path: str = "assets/cup_mask.png",
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image. This demo assumes that the conditioning image and mask have
    the same shape and that height and width are divisible by 32.

    Args:
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
        img_cond_path: path to conditioning image (jpeg/png/webp)
        img_mask_path: path to conditioning mask (jpeg/png/webp
    """
    nsfw_classifier = pipeline(
        "image-classification", model="Falconsai/nsfw_image_detection", device=device)

    name = "flux-dev-fill"
    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(
            f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(
            idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # init all components
    t5 = load_t5(torch_device, max_length=128)
    clip = load_clip(torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    # MY LOADING
    mdevice = "cpu" if offload else torch_device
    if False:
        model = load_flow_model(name, device=mdevice)
    else:
        name = "flux-dev-fill"
        # lora_path = "shapes-inpaint-sanity-check/lora_last.safetensors"
        verbose = True
        from flux.util import optionally_expand_state_dict, print_load_warning
        from huggingface_hub import hf_hub_download
        from flux.model import Flux, FluxLoraWrapper
        from safetensors.torch import load_file as load_sft
        ckpt_path = hf_hub_download(
            configs[name].repo_id, configs[name].repo_flow)
        with torch.device("meta" if ckpt_path is not None else mdevice):
            if lora_path is not None:
                print("Loading LORA")
                model = FluxLoraWrapper(
                    lora_rank=128,
                    params=configs[name].params).to(torch.bfloat16)
            else:
                print("Loading normal model (no LORA)")
                model = Flux(configs[name].params).to(torch.bfloat16)

        if ckpt_path is not None:
            print("Loading checkpoint")
            # load_sft doesn't support torch.device
            sd = load_sft(ckpt_path, device=str(device))
            sd = optionally_expand_state_dict(model, sd)
            missing, unexpected = model.load_state_dict(
                sd, strict=False, assign=True)
            if verbose:
                print_load_warning(missing, unexpected)

        if lora_path is not None:
            print("Loading LoRA")
            lora_sd = load_sft(lora_path, device=str(device))
            # loading the lora params + overwriting scale values in the norms
            missing, unexpected = model.load_state_dict(
                lora_sd, strict=False, assign=True)
            if verbose:
                print_load_warning(missing, unexpected)

    ids = [1, 2, 3, 4, 5] if torch.cuda.device_count() > 1 else [0]
    base_id = 1 if torch.cuda.device_count() > 1 else 0
    gpu_config = GPUSplitConfig(
        gpu_ids=ids,
        max_params_per_gpu=5e9,  # Maximum parameters per GPU
        base_gpu=base_id
    )
    model = split_flux_model_to_gpus(model, gpu_config)
    model = model.to()
    # ----------

    rng = torch.Generator(device="cpu")

    pos_enc = create_sinusoidal_pos_enc(
        2048, 2048, 8, 0)
    pano_path = img_cond_path
    tidx = 0
    for ypos in torch.arange(0, 2048, 512):
        for xpos in torch.arange(0, 2048, 512):
            xpos = int(xpos)
            ypos = int(ypos)
            img = Image.open(pano_path).convert("RGB")
            img = img.resize((2048, 2028))
            image_crop = img.crop((xpos, ypos, xpos + 512, ypos + 512))
            image_tensor = torchvision.transforms.ToTensor()(image_crop)
            sphere_coords = pos_enc[:, ypos:ypos +
                                    512, xpos:xpos+512].unsqueeze(0)
            img_cond_path = f"output/img_{tidx}_{tidx}.png"
            tidx += 1
            display_tensor(image_tensor, "Inference Image")
            image_crop.save(img_cond_path)

            width, height = 512, 512
            opts = SamplingOptions(
                prompt=prompt,
                width=width,
                height=height,
                num_steps=num_steps,
                guidance=guidance,
                seed=seed,
                img_cond_path=img_cond_path,
                img_mask_path=img_mask_path,
            )

            while opts is not None:
                if opts.seed is None:
                    opts.seed = rng.seed()
                print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
                t0 = time.perf_counter()

                # prepare input
                x = get_noise(
                    1,
                    opts.height,
                    opts.width,
                    device=torch_device,
                    dtype=torch.bfloat16,
                    seed=opts.seed,
                )
                opts.seed = None
                if offload:
                    t5, clip, ae = t5.to(torch_device), clip.to(
                        torch_device), ae.to(torch_device)
                inp = prepare_fill(
                    t5,
                    clip,
                    x,
                    prompt=opts.prompt,
                    ae=ae,
                    img_cond_path=opts.img_cond_path,
                    mask_path=opts.img_mask_path,
                )

                timesteps = get_schedule(
                    opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

                # offload TEs and AE to CPU, load model to gpu
                if offload:
                    t5, clip, ae = t5.cpu(), clip.cpu(), ae.cpu()
                    torch.cuda.empty_cache()
                    model = model.to(torch_device)

                # x = 1
                # torch.Tensor([[0, 0], [x, x]]).unsqueeze(0)
                # sphere_coords = torch.tensor([0 / 4, 0 / 4, 512 / 1536, 512 / 1536])
                inp["sphere_coords"] = sphere_coords
                # denoise initial noise
                x = denoise(model, **inp, timesteps=timesteps,
                            guidance=opts.guidance)

                # offload model, load autoencoder to gpu
                if offload:
                    model.cpu()
                    torch.cuda.empty_cache()
                    ae.decoder.to(x.device)

                # decode latents to pixel space
                x = unpack(x.float(), opts.height, opts.width)
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                print(f"Done in {t1 - t0:.1f}s")

                idx = save_image(nsfw_classifier, name, output_name,
                                 idx, x, add_sampling_metadata, prompt)
                x = x.clamp(-1, 1)
                display_tensor(x.to(torch.float32), "Prediction")
                opts = None


def app():
    Fire(main)


if __name__ == "__main__":
    app()
