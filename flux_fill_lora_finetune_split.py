from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

from safetensors.torch import load_file as load_sft, save_file
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from einops import rearrange, repeat
from PIL import Image

import bitsandbytes
import numpy as np
import random
# import math
import torch

from flux.util import (
    load_t5, load_clip, load_ae,
    configs, optionally_expand_state_dict,
    print_load_warning
)
from flux.modules.lora import LinearLora, replace_linear_with_lora
from flux.modules.autoencoder import AutoEncoder
from flux.modules.conditioner import HFEmbedder
from flux.model import FluxLoraWrapper
from flux.gpu_split import split_flux_model_to_gpus, GPUSplitConfig


def make_rec_mask(images, resolution, times=30):
    mask, times = torch.ones_like(images[0:1, :, :]), np.random.randint(
        1, times
    )
    min_size, max_size, margin = np.array([0.03, 0.25, 0.01]) * resolution
    max_size = min(max_size, resolution - margin * 2)

    for _ in range(times):
        width = np.random.randint(int(min_size), int(max_size))
        height = np.random.randint(int(min_size), int(max_size))

        x_start = np.random.randint(
            int(margin), resolution - int(margin) - width + 1
        )
        y_start = np.random.randint(
            int(margin), resolution - int(margin) - height + 1
        )
        mask[:, y_start: y_start + height, x_start: x_start + width] = 0

    mask = 1 - mask if random.random() < 0.5 else mask
    # mask = mask * 0 if random.random() < 0.5 else mask
    return mask


@dataclass
class TrainingConfig:
    outdir = "house-512-testing"
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_epochs: int = 1000
    save_every: int = -1
    num_steps: int = 50
    guidance: float = 1.0
    seed: int = 420


class FluxFillDataset(Dataset):
    def __init__(self,
                 root: Path | str,
                 prompt: str = "a photo of sks",
                 height: int = 512,
                 width: int = 512):
        root = root if isinstance(root, Path) else Path(root)
        self.image_paths = [f for f in root.iterdir()]
        # self.mask_paths = [f for f in root.iterdir()
        #                    if str(f).endswith(('mask.jpg', 'mask.png', 'mask.jpeg'))]
        # assert len(self.image_paths) == len(self.mask_paths)
        self.prompt = prompt
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[0]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # x_positions = [x.item() for x in torch.arange(0, min(w, 1500) - 256, 256)]
        # y_positions = [x.item() for x in torch.arange(0, min(h, 1500) - 256, 256)]

        x_positions = torch.arange(0, 1500, 512)[:3]
        y_positions = torch.arange(0, 1500, 512)[:1]

        xidx = np.random.randint(len(x_positions))
        yidx = np.random.randint(len(y_positions))
        xpos = x_positions[xidx].item()
        ypos = y_positions[yidx].item()
        # print(xidx, yidx, xpos, ypos)
        # box = (xpos, ypos, xpos + 512, ypos + 512)
        # img = img.crop(box)
        # sphere_coords = yidx * len(x_positions) + xidx
        sphere_coords = 0
        img = img.resize((512, 512))
        img = np.array(img)
        img = torch.from_numpy(img).float() / 127.5 - 1.0
        img = rearrange(img, "h w c -> c h w")

        mask = make_rec_mask(
            img.unsqueeze(0), resolution=512, times=30).squeeze(0)

        if sphere_coords is None:
            return img, mask
        return img, mask, sphere_coords


class OptimalTransportPath:
    def __init__(self, sig_min: float = 1e-5, num_timesteps: int = 1000) -> None:
        self.sig_min = sig_min
        self.num_timesteps = num_timesteps
        self.timesteps = torch.linspace(1, 0, num_timesteps + 1)

        m = (1.15 - 0.5) / (4096 - 256)
        b = 0.5 - m * 256
        def mu(x): return m * x + b
        mu = mu((1024 // 8) * (1024 // 8) // 4)
        # self.timesteps = math.exp(mu) / math.exp(mu) + (1 / self.timesteps - 1) ** 1.

        x = torch.arange(num_timesteps, dtype=torch.float32)
        y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)
        y_shifted = y - y.min()
        self.weights = y_shifted * (num_timesteps / y_shifted.sum())
        # half_weights = y_shifted * (num_timesteps / y_shifted.sum())
        # half_weights[num_timesteps / 2:] = half_weighing[num_timesteps / 2:].max()

    def get_timesteps(self, shape):
        indices = torch.randint(
            low=0, high=self.num_timesteps, size=(shape[0],))
        t = self.timesteps[indices]
        weights = self.weights[indices]
        t = t.to("cuda:0")
        weights = weights.to("cuda:0")
        return t, weights

    def sample(self,
               x1: torch.Tensor,
               x0: torch.Tensor,
               t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # REVISIT:xlabs says to flip these
        t = t.view(-1, 1, 1)
        xt = x0 * t + (1 - (1 - self.sig_min) * t) * x1
        vt = x0 - (1 - self.sig_min) * x1
        return xt, vt


def prepare(t5: HFEmbedder,
            clip: HFEmbedder,
            img: torch.Tensor,
            prompt: str | list[str]) -> dict[str, torch.Tensor]:
    bs, c, h, w = img.shape
    # if bs == 1 and not isinstance(prompt, str):
    #     bs = len(prompt)

    # img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    # if img.shape[0] == 1 and bs > 1:
    #   img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        # "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def prepare_fill(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: torch.Tensor,
    mask: torch.Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
) -> dict[str, torch.Tensor]:
    """
    This will be done, per image
    """

    with torch.no_grad():
        img = img.to(img.device)
        mask = mask.to(img.device)
        mask_img = img * (1 - mask)
        img = ae.encode(img)
        mask_img = ae.encode(mask_img)
        mask = mask[:, 0, :, :]
        mask = mask.to(torch.bfloat16)
        mask = rearrange(
            mask,
            "b (h ph) (w pw) -> b (ph pw) h w",
            ph=8,
            pw=8,
        )
        mask = rearrange(
            mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    original_img = img.clone()
    img = img.to(torch.bfloat16)
    mask_img = mask_img.to(torch.bfloat16)
    img = rearrange(
        img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    mask_img = rearrange(
        mask_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    # NOTE: this is now the actual img cond
    img_cond = torch.cat((mask_img, mask), dim=-1)

    path = OptimalTransportPath(sig_min=0)
    x0 = torch.randn_like(img, device=img.device)
    t, weights = path.get_timesteps(original_img.shape)
    print(t)
    xt, vt = path.sample(img, x0, t)
    # NOTE: actually get the image based on noise and path
    return_dict = prepare(t5, clip, original_img, prompt)
    return_dict["img"] = xt
    return_dict["img_cond"] = img_cond
    return_dict["t"] = t
    return_dict["weights"] = weights
    return_dict["vt"] = vt
    return return_dict


def main():
    config = TrainingConfig()
    config.outdir = Path(config.outdir)
    config.outdir.mkdir(exist_ok=True, parents=True)
    offload = False
    device = torch.device("cuda:0")

    # Initialize models
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    ae = load_ae(
        "flux-dev-fill", device="cpu" if offload else device)

    name = "flux-dev-fill"
    hf_download = True
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path  # should be None
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(
            configs[name].repo_id, configs[name].repo_flow)

    with torch.device("cpu" if ckpt_path is not None else device):
        model = FluxLoraWrapper(lora_rank=128,
                                params=configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(
            ckpt_path, device="cpu")
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(
            sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)

    model.requires_grad_(False)

    def set_requires_grad_recursive(module, name=''):
        if isinstance(module, LinearLora):
            module.requires_grad_(False)
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
        for child_name, child in module.named_children():
            child_full_name = f"{name}.{child_name}" if name else child_name
            set_requires_grad_recursive(child, child_full_name)

    set_requires_grad_recursive(model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    gpu_config = GPUSplitConfig(
        gpu_ids=[0],  # List of GPU IDs to use
        max_params_per_gpu=5e9,  # Maximum parameters per GPU
        base_gpu=0  # GPU to place non-distributed components
    )
    model = split_flux_model_to_gpus(model, gpu_config)
    model = model.to()

    dataset = FluxFillDataset(root="datasets/house")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

    # Training setup
    # optimizer = bitsandbytes.optim.AdamW8bit(
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.75)

    # Training loop
    grad_accum_steps = max(1, len(dataset))
    print("Setting grad accum step:", grad_accum_steps)
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.
        for step, batch in enumerate(dataloader):

            if offload:
                t5, clip, ae = t5.to(device), clip.to(device), ae.to(device)
            if len(batch) == 3:
                img, mask, sphere_coords = batch
            else:
                img, mask = batch
                sphere_coords = None
            img, mask = img.to(device), mask.to(device)
            inputs = prepare_fill(
                t5=t5, clip=clip, ae=ae,
                img=img, mask=mask,
                prompt="a photo of sks"
            )
            guids = torch.ones(inputs["img"].shape[0], dtype=torch.bfloat16,
                               device=device)
            if offload:
                t5, clip, ae = t5.cpu(), clip.cpu(), ae.cpu()
                torch.cuda.empty_cache()
                model = model.to(device)

            for k in inputs.keys():
                inputs[k] = inputs[k].to(device)
                inputs[k] = inputs[k].to(torch.bfloat16)
            pred = model(
                img=torch.cat((inputs["img"], inputs["img_cond"]), dim=-1),
                img_ids=inputs["img_ids"],
                txt=inputs["txt"],
                txt_ids=inputs["txt_ids"],
                y=inputs["vec"],
                timesteps=inputs["t"],
                guidance=guids,
                sphere_coords=sphere_coords,
            ).to("cuda:0")
            # pred = unpack(pred, 512, 512).to("cuda:0")
            loss = torch.nn.functional.mse_loss(
                pred, inputs["vt"], reduction="mean")
            # loss += (loss * inputs["weights"] / grad_accum_steps)
            loss += loss / grad_accum_steps
            del inputs
            loss.backward()
            print(f"Step loss: {loss.item():.4f}")
            if ((step + 1) % grad_accum_steps) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()

            if offload:
                model.cpu()
                torch.cuda.empty_cache()
        print(
            f"Epoch {epoch+1}/{config.num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        if config.save_every > 0 and (epoch + 1) % config.save_every == 0:
            # torch.save(
            #     model.state_dict(),
            #     f"lora_checkpoint_epoch_{epoch+1}.safetensors"
            # )
            sd = model.state_dict()
            lora_dict = {k: sd[k] for k in sd.keys() if "lora_" in k}
            # save_file(model.state_dict(), config.outdir /
            save_file(lora_dict, config.outdir /
                      f"lora_checkpoint_epoch_{epoch+1}.safetensors")
    sd = model.state_dict()
    lora_dict = {k: sd[k] for k in sd.keys() if "lora_" in k}
    save_file(lora_dict, config.outdir / "lora_last.safetensors")


if __name__ == "__main__":
    main()
