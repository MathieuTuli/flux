from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

from safetensors.torch import load_file as load_sft
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from einops import rearrange, repeat
from torch.optim import AdamW
from PIL import Image

import numpy as np
import torch

from flux.util import (
    load_t5, load_clip, load_ae,
    configs, optionally_expand_state_dict,
    print_load_warning
)
from flux.modules.lora import LinearLora, replace_linear_with_lora
from flux.modules.autoencoder import AutoEncoder
from flux.modules.conditioner import HFEmbedder
from flux.model import FluxLoraWrapper, Flux


@dataclass
class TrainingConfig:
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_epochs: int = 100
    save_every: int = 50
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
        self.image_paths = [f for f in root.iterdir()
                            if str(f).endswith(('img.jpg', 'img.png', 'img.jpeg'))]
        self.mask_paths = [f for f in root.iterdir()
                           if str(f).endswith(('mask.jpg', 'mask.png', 'mask.jpeg'))]
        assert len(self.image_paths) == len(self.mask_paths)
        self.prompt = prompt
        self.height = height
        self.width = width

        assert len(self.image_paths) == len(
            self.mask_paths), "Number of images and masks must match"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        img = np.array(img)
        img = torch.from_numpy(img).float() / 127.5 - 1.0
        img = rearrange(img, "h w c -> c h w")

        mask = Image.open(mask_path).convert("L").resize((512, 512))
        mask = np.array(mask)
        mask = torch.from_numpy(mask).float() / 255.0
        mask = rearrange(mask, "h w -> 1 h w")

        return img, mask


class OptimalTransportPath:
    def __init__(self, sig_min: float = 1e-5) -> None:
        self.sig_min = sig_min

    def sample(self,
               x1: torch.Tensor,
               x0: torch.Tensor,
               t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = t.view(-1, 1, 1, 1)
        xt = x1 * t + (1 - (1 - self.sig_min) * t) * x0
        vt = x1 - (1 - self.sig_min) * x0
        return xt, vt


def prepare(t5: HFEmbedder,
            clip: HFEmbedder,
            img: torch.Tensor,
            prompt: str | list[str]) -> dict[str, torch.Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

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
        "img": img,
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
        img = img * (1 - mask)
        img = ae.encode(img)
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
    img = rearrange(
        img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    # NOTE: this is now the actual img cond
    img_cond = torch.cat((img, mask), dim=-1)

    path = OptimalTransportPath()
    x0 = torch.randn_like(original_img, device=img.device)
    t = torch.rand(img.shape[0], device=img.device)
    xt, vt = path.sample(original_img, x0, t)
    # NOTE: actually get the image based on noise and path
    return_dict = prepare(t5, clip, xt, prompt)
    return_dict["img_cond"] = img_cond
    return_dict["t"] = t
    return_dict["vt"] = vt
    return return_dict


def main():
    config = TrainingConfig()
    offload = True
    device = torch.device("cuda")

    # Initialize models
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    ae = load_ae(
        "flux-dev-fill", device="cpu" if offload else device)

    name = "flux-dev-fill"
    hf_download = True
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(
            configs[name].repo_id, configs[name].repo_flow)

    model = Flux(
        params=configs[name].params)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(
            ckpt_path, device="cpu" if offload else str(device))
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(
            sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)

    lora_rank = 16
    lora_scale = 1.0
    replace_linear_with_lora(model, lora_rank, lora_scale)

    def set_requires_grad_recursive(module, name=''):
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            if isinstance(module, LinearLora):
                param.requires_grad = True
                print(f"Setting grad for {full_name} to True")
            else:
                param.requires_grad = False

        for child_name, child in module.named_children():
            child_full_name = f"{name}.{child_name}" if name else child_name
            set_requires_grad_recursive(child, child_full_name)

    set_requires_grad_recursive(model)

    dataset = FluxFillDataset(root="cube-dataset")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

    # Training setup
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            if offload:
                t5, clip, ae = t5.to(device), clip.to(device), ae.to(device)
            img, mask = batch
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
                guidance=guids
            )
            if offload:
                model.cpu()
                torch.cuda.empty_cache()
            loss = torch.pow(pred - inputs["vt"], 2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        print(
            f"Epoch {epoch+1}/{config.num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        if (epoch + 1) % config.save_every == 0:
            torch.save(
                model.state_dict(),
                f"lora_checkpoint_epoch_{epoch+1}.safetensors"
            )


if __name__ == "__main__":
    main()
