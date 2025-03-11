from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path


from safetensors.torch import load_file as load_sft, save_file
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from einops import rearrange, repeat
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import random
import base64
import torch
import math
import sys
import io

from flux.util import (
    load_t5, load_clip, load_ae,
    configs, optionally_expand_state_dict,
    print_load_warning
)

from flux.gpu_split import split_flux_model_to_gpus, GPUSplitConfig
from flux.sampling import denoise, get_noise, unpack, get_schedule
from flux.modules.autoencoder import AutoEncoder
from flux.modules.conditioner import HFEmbedder
from flux.modules.layers import SelfAttention
from flux.modules.lora import LinearLora
from flux.model import PosEncProcessor, FluxLoraWrapper


def t_to_p(img):
    return transforms.ToPILImage()(img)


def display_tensor(tensor, title=None):
    """Display a tensor as an image using kitty graphics protocol"""

    # Convert tensor to PIL Image
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor + 1) / 2  # Convert from [-1,1] to [0,1]
    pil_img = transforms.ToPILImage()(tensor)

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
class TrainingConfig:
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_epochs: int = 5000
    save_every: int = 1000


class FluxFillDataset(Dataset):
    def __init__(self,
                 root: Path | str,
                 prompt: str = "a photo of sks",
                 height: int = 512,
                 width: int = 512):
        root = root if isinstance(root, Path) else Path(root)
        self.image_paths = [f for f in root.iterdir()]
        # self.mask_paths = [f for f in root.iterdir()
        #                    if str(f).endswith(('mask.jpg', 'mask.png',
        #                                        'mask.jpeg'))]
        # assert len(self.image_paths) == len(self.mask_paths)
        self.max_rectangle_scale = 0.2
        self.num_mask_rect = 5
        self.prompt = prompt
        self.height = height
        self.width = width
        self.pos_enc = create_sinusoidal_pos_enc(
            2048, 2048, 8, 0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[0]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((2048, 2028))
        w, h = img.size

        x = random.randint(0, 2048 - 512)
        y = random.randint(0, 2048 - 512)

        image_crop = img.crop((x, y, x + 512, y + 512))
        # REVISIT:?
        image_tensor = transforms.ToTensor()(image_crop) * 2.0 - 1.0

        sphere_coords = self.pos_enc[:, y:y+512, x:x+512]

        mask = torch.ones_like(image_tensor)
        if random.random() < 0.5:
            mask.zero_()
        else:
            erase_transform = transforms.RandomErasing(
                p=1., scale=(0.05, self.max_rectangle_scale),
                ratio=(0.3, 3.3), value=0, inplace=True)
            for _ in range(self.num_mask_rect):
                erase_transform(mask)
            if random.random() < 0.5:
                mask = 1 - mask

        return image_tensor, mask, sphere_coords


class OptimalTransportPath:
    def __init__(self, sig_min: float = 1e-5, num_timesteps: int = 1000) -> None:
        self.sig_min = sig_min
        self.num_timesteps = num_timesteps
        self.timesteps = torch.linspace(1, 0, num_timesteps + 1)

        m = (1.15 - 0.5) / (4096 - 256)
        b = 0.5 - m * 256
        def mu(x): return m * x + b
        mu = mu((1024 // 8) * (1024 // 8) // 4)
        # self.timesteps = math.exp(mu) / math.exp(mu) + (
        #       1 / self.timesteps - 1) ** 1.

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
    xt, vt = path.sample(img, x0, t)
    # NOTE: actually get the image based on noise and path
    return_dict = prepare(t5, clip, original_img, prompt)
    return_dict["img"] = xt
    return_dict["img_cond"] = img_cond
    return_dict["t"] = t
    return_dict["weights"] = weights
    return_dict["vt"] = vt
    return return_dict


def plot_loss_history(losses, max_epochs, current_epoch):
    """Plot loss history and display using kitty graphics protocol"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f'Training Loss (Epoch {current_epoch}/{max_epochs})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Use log scale for y-axis
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.grid(True, which="minor", ls=":", alpha=0.2)

    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Convert to PIL Image and display
    img = Image.open(buf)
    display_tensor(transforms.ToTensor()(img), "Loss History")


def main(args):
    config = TrainingConfig()
    args.outdir = Path(args.outdir)
    args.outdir.mkdir(exist_ok=True, parents=True)

    # Initialize loss history
    loss_history = []
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.save_every = args.save_every
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

    with torch.device("meta" if ckpt_path is not None else device):
        model = FluxLoraWrapper(lora_rank=128,
                                params=configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # First move from meta to empty on the target device
        model = model.to_empty(device=device)
        # Then load the checkpoint
        sd = load_sft(ckpt_path, device="cpu")
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(
            sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    else:
        # If no checkpoint, just move directly to device
        model = model.to(device)

    model.requires_grad_(False)

    def init_and_set_requires_grad_recursive(module, name=''):
        if isinstance(module, LinearLora):
            module.requires_grad_(False)
            # Initialize weights while they're in float32
            with torch.cuda.device(device):
                if module.lora_A.weight.is_meta:
                    module.lora_A.weight = torch.nn.Parameter(
                        torch.empty_like(module.lora_A.weight, device=device))
                if module.lora_B.weight.is_meta:
                    module.lora_B.weight = torch.nn.Parameter(
                        torch.empty_like(module.lora_B.weight, device=device))

                torch.nn.init.kaiming_uniform_(module.lora_A.weight.data)
                torch.nn.init.zeros_(module.lora_B.weight.data)

                # Convert to bfloat16
                module.lora_A.weight.data = module.lora_A.weight.data.to(
                    torch.bfloat16)
                module.lora_B.weight.data = module.lora_B.weight.data.to(
                    torch.bfloat16)

                if hasattr(module.lora_B, 'bias') and module.lora_B.bias is not None:
                    if module.lora_B.bias.is_meta:
                        module.lora_B.bias = torch.nn.Parameter(
                            torch.zeros_like(module.lora_B.bias, device=device))
                    torch.nn.init.zeros_(module.lora_B.bias)

            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
        elif isinstance(module, PosEncProcessor):
            module.requires_grad_(True)
        elif False and isinstance(module, SelfAttention):
            module.requires_grad_(True)

        for child_name, child in module.named_children():
            child_full_name = f"{name}.{child_name}" if name else child_name
            init_and_set_requires_grad_recursive(child, child_full_name)

    # Then initialize LoRA weights
    model = model.to(device)  # Move to device before initialization
    init_and_set_requires_grad_recursive(model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"grad=True: {name}")
            if torch.isnan(param).any():
                print(f"WARNING: NaN detected in {name}")

    gpu_config = GPUSplitConfig(
        gpu_ids=[0],  # List of GPU IDs to use
        max_params_per_gpu=5e9,  # Maximum parameters per GPU
        base_gpu=0  # GPU to place non-distributed components
    )
    model = split_flux_model_to_gpus(model, gpu_config)
    model = model.to()

    dataset = FluxFillDataset(root=args.dataset)

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
        optimizer, step_size=3000, gamma=0.75)

    # Training loop
    grad_accum_steps = max(1, len(dataset))
    print("Setting grad accum step:", grad_accum_steps)
    with open("train.log", "w") as f:
        f.write("epoch,step,loss\n")
    debug_img = args.debug_img > 0
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
            if debug_img:
                if epoch >= debug_img:
                    debug_img = 0
                display_tensor(img, "Input Image")
                display_tensor(mask, "Mask")
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

            if epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    height = width = 512
                    x = get_noise(
                        1,
                        height,
                        width,
                        device=device,
                        dtype=torch.bfloat16,
                        seed=420
                    )
                    inputs["sphere_coords"] = sphere_coords if args.sphere_coords == 1 else None
                    print("Validation...")
                    timesteps = get_schedule(
                        50, inputs["img"].shape[1], shift=(name != "flux-schnell"))
                    torch.cuda.empty_cache()
                    x = denoise(model, **inputs, timesteps=timesteps, guidance=1)
                    x = unpack(x.float(), height, width)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        x = ae.decode(x)
                display_tensor(img,
                               f"Validation Target {epoch}/{config.num_epochs}")
                display_tensor(mask,
                               f"Validation Target {epoch}/{config.num_epochs}")
                x = x.clamp(-1, 1)
                display_tensor(x.to(torch.float32),
                               f"Validation Pred {epoch}/{config.num_epochs}")
                model.train()
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
            loss.backward()
            # print(f"Step loss: {loss.item():.4f}")
            with open("train.log", "a+") as f:
                f.write(f"{epoch},{step},{loss.item()}\n")
            if ((step + 1) % grad_accum_steps) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()

            if offload:
                model.cpu()
                torch.cuda.empty_cache()
        avg_epoch_loss = epoch_loss/len(dataloader)
        loss_history.append(avg_epoch_loss)

        # Plot loss every 100 epochs
        if epoch % 100 == 0:
            plot_loss_history(loss_history, config.num_epochs, epoch + 1)
        print(
            f"Epoch {epoch+1}/{config.num_epochs}, " +
            f"Loss: {avg_epoch_loss:.4f}")

        if config.save_every > 0 and (epoch + 1) % config.save_every == 0:
            # torch.save(
            #     model.state_dict(),
            #     f"lora_checkpoint_epoch_{epoch+1}.safetensors"
            # )
            sd = model.state_dict()
            lora_dict = {k: sd[k]
                         for k in sd.keys() if "lora_" in k or "panorama" in k}
            # save_file(model.state_dict(), args.outdir /
            print(f"Saved checkpoint lora_checkpoint_epoch_{epoch+1}.safetensors.")
            save_file(lora_dict, args.outdir /
                      f"lora_checkpoint_epoch_{epoch+1}.safetensors")
    sd = model.state_dict()
    lora_dict = {k: sd[k]
                 for k in sd.keys() if "lora_" in k or "panorama" in k}
    save_file(lora_dict, args.outdir / "lora_last.safetensors")


parser = ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--outdir", required=True)

parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--save-every", type=int, default=1000)
parser.add_argument("--debug-img", type=int, default=0)
parser.add_argument("--sphere-coords", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
