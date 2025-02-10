import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Iterator, Tuple
import math
import os
import torchvision
import tempfile
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
from flux.train_utils import PanoramaSphereMapper, SphereConfig, make_rec_mask


@dataclass
class CropInfo:
    position: Tuple[int, int]      # (y, x) position in original image
    conditioning: Image.Image       # 512x512 crop from previous generation
    mask: Image.Image              # Mask where 1 is area to generate
    sphere_coords: torch.Tensor    # Spherical coordinates for this crop


class OrderedGridWindowCrop:
    def __init__(
        self,
        crop_size: tuple[int, int],
        stride: tuple[int, int] | None = None,  # If None, use crop_size//2
        padding: int | tuple[int, int] = 0
    ):
        self.crop_size = crop_size
        self.stride = stride or (crop_size[0]//2, crop_size[1]//2)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Initialize position tracking
        self.current_row = 0
        self.current_col = 0
        self.y_positions = []
        self.x_positions = []
        self.current_image_size = None

    def _init_grid(self, img_size: tuple[int, int]):
        """Initialize grid positions for a given image size."""
        H, W = img_size
        crop_h, crop_w = self.crop_size
        stride_h, stride_w = self.stride
        pad_y, pad_x = self.padding

        # Calculate valid start positions
        self.y_positions = list(range(
            pad_y,
            H - crop_h - pad_y,
            stride_h
        ))
        self.x_positions = list(range(
            pad_x,
            W - crop_w - pad_x,
            stride_w
        ))

        self.current_row = 0
        self.current_col = 0
        self.current_image_size = img_size

        print(
            f"Grid initialized with {len(self.y_positions)} rows and {len(self.x_positions)} columns")

    def get_next_window(self, img_size: tuple[int, int]) -> tuple[int, int] | None:
        """Get next window position from the grid."""
        if self.current_image_size != img_size:
            self._init_grid(img_size)

        if self.current_row >= len(self.y_positions):
            return None

        y = self.y_positions[self.current_row]
        x = self.x_positions[self.current_col]

        return (y, x)

    def get_position_info(self) -> tuple[int, int, bool, bool]:
        """
        Get information about current position.
        Returns:
            tuple containing:
            - current row
            - current column
            - whether this is first crop in first row
            - whether this is first row
        """
        is_first_crop = self.current_row == 0 and self.current_col == 0
        is_first_row = self.current_row == 0
        return self.current_row, self.current_col, is_first_crop, is_first_row

    def advance_position(self):
        """Advance to next position."""
        self.current_col += 1
        if self.current_col >= len(self.x_positions):
            self.current_col = 0
            self.current_row += 1

    def __call__(self, img: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], dict] | None:
        """
        Extract the next window from the grid.

        Returns:
            tuple of:
                - Cropped image tensor
                - Window position
                - Position info dictionary
            or None if done
        """
        next_pos = self.get_next_window(img.shape[-2:])
        if next_pos is None:
            return None

        # Extract window
        crop_y, crop_x = next_pos
        crop_h, crop_w = self.crop_size
        window = img[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        # Get position info
        row, col, is_first_crop, is_first_row = self.get_position_info()
        position_info = {
            'row': row,
            'col': col,
            'is_first_crop': is_first_crop,
            'is_first_row': is_first_row
        }

        # Advance position for next call
        self.advance_position()

        return window, next_pos, position_info


def create_progressive_mask(crop_size: tuple[int, int], is_first_crop: bool, is_first_row: bool) -> torch.Tensor:
    """
    Create appropriate mask based on position:
    - First crop: No mask (all zeros)
    - First row: Right 50% masked
    - Other rows: Bottom 50% masked

    Returns:
        Tensor of shape (1, H, W) where 1 indicates areas to generate
    """
    H, W = crop_size
    mask = torch.zeros((1, H, W))

    if is_first_crop:
        return mask

    if is_first_row:
        # Mask right half
        mask[..., :, W//2:] = 1
    else:
        # Mask bottom half
        mask[..., H//2:, :] = 1

    return mask


class PanoramaInfiller:
    def __init__(self,
                 panorama_path: str,
                 output_dir: str,
                 crop_size: int = 512):
        """Initialize panorama infiller."""
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])

        # Load and transform panorama
        self.panorama_pil = Image.open(panorama_path).convert("RGB")
        self.width, self.height = self.panorama_pil.size
        self.panorama = self.transform(self.panorama_pil)

        # Setup output directory structure
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.temp_dir = os.path.join(output_dir, "temp")
        self.crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.crops_dir, exist_ok=True)

        # Initialize components
        self.window_crop = OrderedGridWindowCrop(
            crop_size=(crop_size, crop_size),
            stride=(crop_size//2, crop_size//2),
            padding=64
        )

        self.sphere_mapper = PanoramaSphereMapper(
            SphereConfig(patch_size=2, target_size=(512, 512))
        )

        # Initialize result buffer
        self.result = Image.new('RGB', (self.width, self.height))

        # Save paths for temporary files
        self.temp_cond_path = os.path.join(self.temp_dir, "temp_cond.png")
        self.temp_mask_path = os.path.join(self.temp_dir, "temp_mask.png")

    def get_next_crop(self) -> Tuple[CropInfo, str, str] | None:
        """Get next window to process."""
        next_window = self.window_crop(self.panorama)
        if next_window is None:
            return None

        cropped_img, crop_pos, position_info = next_window
        rotation = 0  # No rotation

        # Get sphere coordinates
        sphere_coords = self.sphere_mapper.get_coords_for_crop(
            panorama_size=(self.height, self.width),
            crop_size=(512, 512),
            crop_position=crop_pos,
            rotation=rotation
        )
        sphere_coords = self.sphere_mapper.get_nearest_sphere_patches(
            sphere_coords)

        # Convert tensor to PIL for saving
        conditioning = torchvision.transforms.ToPILImage()(cropped_img)
        conditioning.save(self.temp_cond_path)

        # Create appropriate mask based on position
        mask = create_progressive_mask(
            (512, 512),
            position_info['is_first_crop'],
            position_info['is_first_row']
        )

        # Convert mask to PIL and save
        mask_pil = Image.fromarray(
            (mask.squeeze(0).numpy() * 255).astype(np.uint8))
        mask_pil.save(self.temp_mask_path)

        crop_info = CropInfo(
            position=crop_pos,
            conditioning=conditioning,
            mask=mask_pil,
            sphere_coords=sphere_coords
        )

        print(
            f"Processing window at row {position_info['row']}, col {position_info['col']}")
        print(
            f"First crop: {position_info['is_first_crop']}, First row: {position_info['is_first_row']}")

        return crop_info, self.temp_cond_path, self.temp_mask_path

    def update_result(self, crop_info: CropInfo, generated_image: Image.Image):
        """Update result buffer with newly generated image."""
        import pdb; pdb.set_trace()
        y, x = crop_info.position
        mask = np.array(crop_info.mask) / 255.0

        # Update full panorama
        crop_size = (512, 512)
        existing = np.array(self.result.crop(
            (x, y, x + crop_size[0], y + crop_size[1])))
        new = np.array(generated_image)

        mask = mask[..., np.newaxis]
        blended = existing * (1 - mask) + new * mask

        self.result.paste(Image.fromarray(blended.astype(np.uint8)), (x, y))

        # Save individual crop
        row = y // (crop_size[0]//2)
        col = x // (crop_size[1]//2)
        crop_path = os.path.join(self.crops_dir, f"crop_row{row}_col{col}.png")
        generated_image.save(crop_path)

        # Save intermediate panorama
        self.save_result(os.path.join(
            self.output_dir, f"panorama_row{row}_col{col}.png"))

    def save_result(self, path: str):
        """Save current result."""
        self.result.save(path)

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_cond_path):
            os.remove(self.temp_cond_path)
        if os.path.exists(self.temp_mask_path):
            os.remove(self.temp_mask_path)
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass


@torch.inference_mode()
def main(
    panorama_path: str,
    output_path: str,
    prompt: str = "a photo of sks",
    device: str = "cuda",
    num_steps: int = 50,
    guidance: float = 30.0,
    seed: int | None = 420,
):
    torch_device = torch.device(device)
    name = "flux-dev-fill"
    # Initialize your model components
    t5 = load_t5(torch_device, max_length=128)
    clip = load_clip(torch_device)
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
    ckpt_path = "shapes-without-sphere-pe/lora_last.safetensors"
    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(
            ckpt_path, device="cpu")
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(
            sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)

    gpu_config = GPUSplitConfig(
        gpu_ids=[1, 2, 3, 4, 5],  # List of GPU IDs to use
        max_params_per_gpu=5e9,  # Maximum parameters per GPU
        base_gpu=1  # GPU to place non-distributed components
    )
    model = split_flux_model_to_gpus(model, gpu_config)
    model = model.to()
    # ----------
    ae = load_ae(name, device=torch_device)

    # Initialize infiller
    output_dir = os.path.dirname(output_path)
    infiller = PanoramaInfiller(panorama_path, output_dir)

    try:
        while (next_crop := infiller.get_next_crop()) is not None:
            crop_info, cond_path, mask_path = next_crop
            print(f"Processing window at position {crop_info.position}")

            # Prepare inputs
            x = get_noise(
                1,
                512,
                512,
                device=torch_device,
                dtype=torch.bfloat16,
                seed=seed,
            )

            # Use the saved files in prepare_fill
            inp = prepare_fill(
                t5,
                clip,
                x,
                prompt=prompt,
                ae=ae,
                img_cond_path=cond_path,
                mask_path=mask_path,
            )

            # Add sphere coordinates
            inp["sphere_coords"] = None  # crop_info.sphere_coords.unsqueeze(0).to(x.device)

            # Generate
            timesteps = get_schedule(num_steps, inp["img"].shape[1])
            x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)
            x = unpack(x.float(), 512, 512)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                generated = ae.decode(x)

            # Convert and update
            generated_image = torchvision.transforms.ToPILImage()(
                generated.squeeze(0).cpu().float())
            infiller.update_result(crop_info, generated_image)

        # Save final result
        infiller.save_result(output_path)

    finally:
        # Clean up temporary files
        infiller.cleanup()


if __name__ == "__main__":
    main(
        panorama_path="datasets/shapes/image.png",
        output_path="output/final_result.jpg",
        prompt="a photo of sks",
        num_steps=50,
        guidance=30.0
    )
