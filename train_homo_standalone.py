from pathlib import Path
from argparse import ArgumentParser

from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from hydra import compose, initialize
from PIL.ImageOps import exif_transpose
from huggingface_hub import hf_hub_download
from flux.model import Flux, FluxLoraWrapper
import torchvision.transforms.v2 as transforms_v2
from safetensors.torch import load_file as load_sft
from flux.util import (
    configs, load_ae, load_clip, load_t5,
    print_load_warning, optionally_expand_state_dict
)

import torch
import random
import torchvision
import numpy as np

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="./config.yaml")


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


def collate_fn(examples):
    prompts = [example["prompt"] for example in examples]
    images = [example["images"] for example in examples]

    masks = [example["masks"] for example in examples]
    weightings = [example["weightings"] for example in examples]
    conditioning_images = [
        example["conditioning_images"] for example in examples
    ]

    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()

    masks = torch.stack(masks)
    masks = masks.to(memory_format=torch.contiguous_format).float()

    weightings = torch.stack(weightings)
    weightings = weightings.to(memory_format=torch.contiguous_format).float()

    conditioning_images = torch.stack(conditioning_images)
    conditioning_images = conditioning_images.to(
        memory_format=torch.contiguous_format
    ).float()

    batch = {
        "prompts": prompts,
        "images": images,
        "masks": masks,
        "weightings": weightings,
        "conditioning_images": conditioning_images,
    }
    return batch


class InfillDataset(Dataset):
    def __init__(self, root: str, cfg):
        self.epoch = 0
        size = self.size = cfg.dataset.resolution
        self.cfg = cfg.dataset

        self.ref_data_root = Path(root) / "ref"
        self.gt_image = Path(root) / "target" / "gt.png"
        self.target_image = Path(root) / "target" / "target.png"
        self.target_mask = Path(root) / "target" / "mask.png"
        if not (
            self.ref_data_root.exists()
            and self.target_image.exists()
            and self.target_mask.exists()
        ):
            raise ValueError("Train images root doesn't exist.")

        self.train_images_path = [
            p
            for p in self.ref_data_root.iterdir()
            if p.is_file() and p.suffix == ".png"
        ] + [self.target_image]

        self.num_train_images = len(self.train_images_path)
        self.train_prompt = cfg.training.prompt

        self.transform = transforms_v2.Compose(
            [
                transforms_v2.RandomResize(size, int(1.125 * size)),
                transforms_v2.RandomCrop(size),
                transforms_v2.ToImage(),
                transforms_v2.ConvertImageDtype(),
                transforms_v2.Normalize([0.5], [0.5]),
            ]
        )

        self.train_images = list()
        for fname in self.train_images_path:
            image = Image.open(fname)
            image = exif_transpose(image)

            if not image.mode == "RGB":
                image = image.convert("RGB")
            self.train_images.append(image)

        if cfg.dataset.normalize_colors:
            avg_tgt = (
                torchvision.transforms.functional.pil_to_tensor(
                    self.train_images[-1]
                )
                .to(torch.float32)
                .mean(dim=(0, 1), keepdim=True)
            )
            topil = torchvision.transforms.ToPILImage()
            for i, image in enumerate(self.train_images[:-1]):
                image = torchvision.transforms.functional.pil_to_tensor(
                    image
                ).to(torch.float32)
                scale = avg_tgt / image.mean(dim=(0, 1), keepdim=True)
                self.train_images[i] = topil((image * scale).to(torch.uint8))

    def __len__(self):
        return self.num_train_images

    def __getitem__(self, index):
        example = dict()

        image = self.train_images[index]

        if index < len(self) - 1:
            weighting = Image.new("L", image.size)
        else:
            weighting = Image.open(self.target_mask)
            weighting = exif_transpose(weighting)

        image, weighting = self.transform(image, weighting)

        example["images"], example["weightings"] = image, weighting[0:1] < 0
        ignore_mask_prob = 1 - self.cfg.ignore_mask_prob

        rnd = random.random()
        if index == len(self) - 1:
            example["masks"] = 1 - (example["weightings"]).float()
        elif rnd > ignore_mask_prob:
            example["masks"] = torch.ones_like(example["images"][0:1])
        else:
            example["masks"] = make_rec_mask(
                example["images"], self.size, self.cfg.max_rec_mask
            )

        train_prompt = (
            ""
            if random.random() < self.cfg.drop_prompt_prob
            else self.train_prompt
        )
        example["prompt"] = train_prompt
        return example


def main(cfg: OmegaConf):
    # LOAD MODELS -----------------
    name = "flux-dev-fill"
    torch_device = torch.device("cpu")
    t5 = load_t5(torch_device, max_length=128)
    clip = load_clip(torch_device)
    ae = load_ae(name, device=torch_device)
    ckpt_path = configs[name].ckpt_path
    ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)
    with torch.device("meta" if ckpt_path is not None else str(torch_device)):
        model = FluxLoraWrapper(params=configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=torch_device)
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(
            sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)

    if configs[name].lora_path is not None:
        print("Loading LoRA")
        lora_sd = load_sft(configs[name].lora_path, device=torch_device)
        # loading the lora params + overwriting scale values in the norms
        missing, unexpected = model.load_state_dict(
            lora_sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    ae.requires_grad_(False)
    model.train()
    # -----------------------------

    # TODO: FREEZE ALL BUT LORA

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 100
    train_dataset = InfillDataset(root=cfg.dataset.train_data_dir, cfg=cfg)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataset.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.dataset.num_workers,
    )
    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_dataloader:
            # Forward pass
            outputs = model(batch["input_ids"])
            loss = loss_fn(outputs, batch["labels"])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")


if __name__ == "__main__":
    args = parser.parse_args()
    initialize(version_base=None, config_path="", job_name="None")
    cfg = compose(config_name=args.config)
    OmegaConf.resolve(cfg)
    main(cfg)
