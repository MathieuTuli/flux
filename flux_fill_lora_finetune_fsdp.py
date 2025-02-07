from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import functools

from safetensors.torch import load_file as load_sft, save_file
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from einops import rearrange, repeat
from torch.optim import AdamW
from PIL import Image

import bitsandbytes
import numpy as np
import random
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
from flux.sampling import unpack

from flux.modules.layers import (
    LastLayer,
    MLPEmbedder,
    DoubleStreamBlock,
    SingleStreamBlock,
)
from flux.model import DoubleStreamSequential, SingleStreamSequential

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os


@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 200
    save_every: int = 50
    num_steps: int = 50
    guidance: float = 1.0
    seed: int = 420


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
    return mask


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
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Randomly select quadrant (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)
        quadrant = random.randint(0, 3)

        # Calculate crop coordinates based on quadrant
        crop_w, crop_h = w // 2, h // 2
        if quadrant == 0:  # top-left
            x1, y1 = 0, 0
        elif quadrant == 1:  # top-right
            x1, y1 = crop_w, 0
        elif quadrant == 2:  # bottom-left
            x1, y1 = 0, crop_h
        else:  # bottom-right
            x1, y1 = crop_w, crop_h
        x2, y2 = x1 + crop_w, y1 + crop_h

        # Store crop coordinates and perform crop
        img = img.crop((x1, y1, x2, y2)).resize((512, 512))

        x = 1
        if quadrant == 0:  # top-left
            crop_coords = torch.Tensor([[0, 0], [x, x]])
        elif quadrant == 1:  # top-right
            crop_coords = torch.Tensor([[x, 0], [x * 2, x]])
        elif quadrant == 2:  # bottom-left
            crop_coords = torch.Tensor([[0, x], [x, x * 2]])
        else:  # bottom-right
            crop_coords = torch.Tensor([[x, x], [x * 2, x * 2]])
        img = np.array(img)
        img = torch.from_numpy(img).float() / 127.5 - 1.0
        img = rearrange(img, "h w c -> c h w")

        # mask_path = self.mask_paths[idx]
        # mask = Image.open(mask_path).convert("L").resize((512, 512))
        # mask = np.array(mask)
        # mask = torch.from_numpy(mask).float() / 255.0
        # mask = rearrange(mask, "h w -> 1 h w")

        mask = make_rec_mask(img.unsqueeze(
            0), resolution=512, times=30).squeeze(0)

        return img, mask, crop_coords


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


def setup_distribution():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    print("Setting rank ", local_rank)
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def get_mixed_precision_policy():
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )


def wrap_flux_model(model, local_rank):
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            DoubleStreamBlock,
            DoubleStreamSequential,
            SingleStreamBlock,
            SingleStreamSequential,
            LastLayer,
            MLPEmbedder,
        },
    )

    # We can also add a size-based policy as backup
    size_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1_500_000  # Shard any other large layers
    )

    # Combine policies
    def combined_policy(module, recurse, nonwrapped_numel):
        # Try transformer policy first
        if transformer_wrap_policy(module, recurse, nonwrapped_numel):
            return True
        # If that doesn't match, try size policy
        return size_policy(module, recurse, nonwrapped_numel)

    print("Wrapping: ", local_rank, torch.cuda.current_device())
    fsdp_config = dict(
        auto_wrap_policy=transformer_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=get_mixed_precision_policy(),
        device_id=torch.cuda.current_device(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )
    model = FSDP(model, **fsdp_config)
    return model


def print_gpu_memory(rank):
    if rank == 0:  # Only print from first process
        for i in range(torch.cuda.device_count()):
            memory = torch.cuda.memory_allocated(i) / 1024**3
            print(f"GPU {i} memory allocated: {memory:.2f} GB")


def main():
  # Initialize distributed setup
    local_rank, world_size = setup_distribution()
    config = TrainingConfig()

    # Device assignments
    encoder_device = torch.device("cuda:0")
    flux_device = torch.device(f"cuda:{local_rank}")

    # Initialize models with proper device placement
    if local_rank == 0:
        t5 = load_t5(encoder_device, max_length=128)
        clip = load_clip(encoder_device)
        ae = load_ae("flux-dev-fill", device=encoder_device)
    else:
        t5, clip, ae = None, None, None

    # Load Flux model
    name = "flux-dev-fill"
    if local_rank == 0:
        ckpt_path = configs[name].ckpt_path
        if (ckpt_path is None and configs[name].repo_id is not None
                and configs[name].repo_flow is not None):
            ckpt_path = hf_hub_download(
                configs[name].repo_id, configs[name].repo_flow)
    else:
        ckpt_path = None

    # Broadcast ckpt_path to all ranks
    if world_size > 1:
        object_list = [ckpt_path]
        dist.broadcast_object_list(object_list, src=0)
        ckpt_path = object_list[0]

    # Initialize Flux model
    model = Flux(params=configs[name].params).to(torch.bfloat16)

    def remap_sd_for_blocks(sd):
        new_sd = dict()

        for k, v in sd.items():
            if k.startswith("double_blocks.") or k.startswith("single_blocks."):
                parts = k.split(".")
                new_key = f"{parts[0]}.blocks.{'.'.join(parts[1:])}"
                new_sd[new_key] = v
            else:
                new_sd[k] = v
        return new_sd

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device="cpu")
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(
            remap_sd_for_blocks(sd), strict=False, assign=True)
        if local_rank == 0:
            print_load_warning(missing, unexpected)

    # Apply LoRA modifications
    lora_rank = 16
    lora_scale = 1.0
    replace_linear_with_lora(
        model, lora_rank, lora_scale, recursive=False,
        keys_override=["single_blocks", "final_layer"],
        keys_ignore=["img_in", "txt_in"]
    )

    # Wrap model with FSDP
    model = wrap_flux_model(model, local_rank+1)
    print(model)

    # Setup dataset and dataloader with DistributedSampler
    dataset = FluxFillDataset(root="cube-dataset")
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=local_rank)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # Initialize optimizer
    optimizer = bitsandbytes.optim.AdamW8bit(
        model.parameters(), lr=config.learning_rate
    )

    # Training loop
    print_gpu_memory(local_rank)
    for epoch in range(config.num_epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        model.train()
        epoch_loss = 0

        if epoch == 0:
            print_gpu_memory(local_rank)
            print("Memory usage before first batch") if local_rank == 0 else None

        for batch in dataloader:
            optimizer.zero_grad()

            img, mask, coords = batch
            img = img.to(flux_device)
            mask = mask.to(flux_device)

            inputs = None
            # Prepare inputs on encoder_device (GPU 0)
            if local_rank == 0:
                inputs = prepare_fill(
                    t5=t5, clip=clip, ae=ae,
                    img=img, mask=mask,
                    prompt="a photo of sks"
                )
                # Move inputs to flux_device after preparation
                inputs = {k: v.to(flux_device) for k, v in inputs.items()}

            # Broadcast inputs to all ranks
            if world_size > 1:
                object_list = [inputs]
                dist.broadcast_object_list(object_list, src=0)
                inputs = object_list[0]

            # Convert inputs to bfloat16
            inputs = {k: v.to(torch.bfloat16) for k, v in inputs.items()}
            guids = torch.ones(inputs["img"].shape[0], dtype=torch.bfloat16,
                               device=flux_device)

            # Forward pass
            pred = model(
                img=torch.cat((inputs["img"], inputs["img_cond"]), dim=-1),
                img_ids=inputs["img_ids"],
                txt=inputs["txt"],
                txt_ids=inputs["txt_ids"],
                y=inputs["vec"],
                timesteps=inputs["t"],
                guidance=guids,
                homo_pos_map=None
            )

            pred = unpack(pred, 512, 512)
            loss = torch.pow(pred - inputs["vt"], 2).mean()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if local_rank == 0:
                print(f"Step loss: {loss.item():.4f}")
            epoch_loss += loss.item()

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs}, "
                  f"Loss: {epoch_loss/len(dataloader):.4f}")

        # Save checkpoint on rank 0
        if local_rank == 0 and (epoch + 1) % config.save_every == 0:
            save_file(
                model.state_dict(),
                f"lora_checkpoint_epoch_{epoch+1}.safetensors"
            )


if __name__ == "__main__":
    main()
