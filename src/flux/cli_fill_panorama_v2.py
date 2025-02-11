import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import cv2
import torch
from fire import Fire
from PIL import Image
from transformers import pipeline

from flux.sampling import denoise, get_noise, get_schedule, prepare_fill, unpack
from flux.util import (
    configs,
    load_ae,
    load_clip,
    load_t5,
    save_image
)

from flux.gpu_split import split_flux_model_to_gpus, GPUSplitConfig
from flux.train_utils import PanoramaSphereMapper, SphereConfig


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
    seed: int | None = None,
    prompt: str = "a photo of sks",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 50,
    guidance: float = 30.0,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/cup.png",
):
    """
    Sample the flux model. Run for a
    single image. This demo assumes that the conditioning image and mask have
    the same shape and that height and width are divisible by 32.

    Args:
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
        img_cond_path: path to conditioning image (jpeg/png/webp)
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
    # model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)
    # MY LOADING
    from flux.util import optionally_expand_state_dict, print_load_warning
    from flux.modules.lora import LinearLora, replace_linear_with_lora
    from safetensors.torch import load_file as load_sft
    from flux.model import Flux
    model = Flux(
        params=configs[name].params).to(torch.bfloat16)

    lora_rank = 16
    lora_scale = 1.0
    replace_linear_with_lora(model, lora_rank, lora_scale, recursive=True)
    # ckpt_path = configs[name].ckpt_path
    # lora_done.safetensors"
    ckpt_path = "cube-inpaint-sanity-check/lora_last.safetensors"
    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(
            ckpt_path, device="cpu")
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(
            sd, strict=False, assign=True)
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

    panorama_pil = Image.open(img_cond_path).convert("RGB")
    width, height = panorama_pil.size
    crop_size = (512, 512)
    sphere_mapper = PanoramaSphereMapper(SphereConfig(patch_size=2, target_size=crop_size))
    stride = (crop_size[0] // 2, crop_size[1] // 2)
    stride = (512, 512)

    x_positions = torch.arange(0, 1500, stride[0])[:1]
    y_positions = torch.arange(0, 1500, stride[1])[:1]

    first_row = True
    idx = 0
    for i, ypos in enumerate(y_positions):
        for j, xpos in enumerate(x_positions):
            if first_row:
                img_mask_path = "left_mask.png"
            else:
                img_mask_path = "bottom_mask.png"

            box = (xpos.item(), ypos.item(),
                   xpos.item() + crop_size[0], ypos.item() + crop_size[1])
            crop = panorama_pil.crop(box)
            img_cond_path = os.path.join(output_dir, f"{idx:02d}_cond.png")
            output_name = os.path.join(output_dir, f"{idx:02d}_pred.png")
            mask_oname = os.path.join(output_dir, f"{idx:02d}_mask.png")
            crop.save(img_cond_path)
            Image.open(img_mask_path).save(mask_oname)

            # crop_pos = (box[0], box[1])
            # sphere_coords = sphere_mapper.get_coords_for_crop(
            #     panorama_size=(height, width),
            #     crop_size=crop_size,
            #     crop_position=crop_pos,
            #     rotation=0
            # )
            sphere_coords = i * len(x_positions) + j
            assert idx == sphere_coords, f"{idx} but sphere coords are {sphere_coords}, ({i}, {j})"
            sphere_coords = torch.tensor([sphere_coords])
            idx += 1

            opts = SamplingOptions(
                prompt=prompt,
                width=crop_size[0],
                height=crop_size[1],
                num_steps=num_steps,
                guidance=guidance,
                seed=seed,
                img_cond_path=img_cond_path,
                img_mask_path=img_mask_path,
            )

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
            # import pdb; pdb.set_trace()

            timesteps = get_schedule(
                opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

            # offload TEs and AE to CPU, load model to gpu
            if offload:
                t5, clip, ae = t5.cpu(), clip.cpu(), ae.cpu()
                torch.cuda.empty_cache()
                model = model.to(torch_device)

            x = 1
            # torch.Tensor([[0, 0], [x, x]]).unsqueeze(0)
            inp["sphere_coords"] = sphere_coords
            # denoise initial noise
            x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

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

            _ = save_image(nsfw_classifier, name, output_name,
                             idx, x, add_sampling_metadata, prompt)
        first_row = False


def app():
    Fire(main)


if __name__ == "__main__":
    app()
