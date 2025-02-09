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


@dataclass
class CropInfo:
    position: Tuple[int, int]      # (y, x) position in original image
    conditioning: Image.Image       # 512x512 crop from previous generation
    mask: Image.Image              # Mask where 1 is area to generate
    sphere_coords: torch.Tensor    # Spherical coordinates for this crop


class PanoramaInfiller:
    def __init__(self, 
                 panorama_path: str, 
                 output_dir: str,
                 crop_size: int = 512, 
                 overlap: float = 0.5):
        """
        Initialize panorama infiller.
        """
        # Setup transform to match dataloader
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])
        
        # Load and transform panorama
        self.panorama_pil = Image.open(panorama_path).convert("RGB")
        self.width, self.height = self.panorama_pil.size
        self.panorama = self.transform(self.panorama_pil)
        
        self.crop_size = crop_size
        self.overlap = overlap
        self.stride = int(crop_size * (1 - overlap))
        
        # Setup output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Calculate number of crops
        self.n_crops_w = math.ceil((self.width - crop_size) / self.stride) + 1
        self.n_crops_h = math.ceil((self.height - crop_size) / self.stride) + 1
        
        # Initialize result buffer
        self.result = Image.new('RGB', (self.width, self.height))
        self.current_row = 0
        self.current_col = 0
        
        # Initialize components to match dataloader
        self.random_rotate_crop = RandomRotatedCrop(
            crop_size=(512, 512),
            max_rotation=0,  # Set to 0 as in your dataloader
            padding=64
        )
        
        self.sphere_mapper = PanoramaSphereMapper(
            SphereConfig(patch_size=2, target_size=(512, 512))
        )
        
        # Save paths for temporary files
        self.temp_cond_path = os.path.join(self.temp_dir, "temp_cond.png")
        self.temp_mask_path = os.path.join(self.temp_dir, "temp_mask.png")

    def get_next_crop(self) -> Tuple[CropInfo, str, str] | None:
        """
        Get next crop to process.
        Returns (CropInfo, conditioning_path, mask_path) or None when done.
        """
        if self.current_row >= self.n_crops_h:
            return None
            
        # Get base crop position for this iteration
        y, x = self._get_crop_position(self.current_row, self.current_col)
        
        # Get crop and rotate using same approach as dataloader
        crop_tensor = self.panorama[:, y:y+self.crop_size, x:x+self.crop_size]
        cropped_img, rotation, crop_pos = self.random_rotate_crop(crop_tensor)
        
        # Get sphere coordinates using exact same process as dataloader
        sphere_coords = self.sphere_mapper.get_coords_for_crop(
            panorama_size=(self.height, self.width),
            crop_size=(512, 512),
            crop_position=crop_pos,
            rotation=rotation
        )
        
        # Get nearest patches exactly as in dataloader
        sphere_coords = self.sphere_mapper.get_nearest_sphere_patches(sphere_coords)
        
        # Convert tensor to PIL for saving
        conditioning = torchvision.transforms.ToPILImage()(cropped_img)
        conditioning.save(self.temp_cond_path)
        
        # Create and save mask
        mask = self._create_overlap_mask(self.current_row, self.current_col)
        mask.save(self.temp_mask_path)
        
        crop_info = CropInfo(
            position=(y, x),
            conditioning=conditioning,
            mask=mask,
            sphere_coords=sphere_coords
        )
        
        # Update position
        self.current_col += 1
        if self.current_col >= self.n_crops_w:
            self.current_col = 0
            self.current_row += 1
            
        return crop_info, self.temp_cond_path, self.temp_mask_path

    def _get_crop_position(self, row: int, col: int) -> Tuple[int, int]:
        """Get top-left position of crop in original image."""
        x = min(col * self.stride, self.width - self.crop_size)
        y = min(row * self.stride, self.height - self.crop_size)
        return (y, x)

    def _create_overlap_mask(self, row: int, col: int) -> Image.Image:
        """Create mask where 1 indicates areas to generate."""
        mask = Image.new('L', (self.crop_size, self.crop_size), 255)
        
        if col > 0:
            mask_array = np.array(mask)
            mask_array[:, :self.stride] = 0
            mask = Image.fromarray(mask_array)
        
        if row > 0:
            mask_array = np.array(mask)
            mask_array[:self.stride, :] = 0
            mask = Image.fromarray(mask_array)
            
        return mask

    def update_result(self, crop_info: CropInfo, generated_image: Image.Image):
        """Update result buffer with newly generated image."""
        y, x = crop_info.position
        mask = np.array(crop_info.mask) / 255.0
        
        existing = np.array(self.result.crop((x, y, x + self.crop_size, y + self.crop_size)))
        new = np.array(generated_image)
        
        mask = mask[..., np.newaxis]
        blended = existing * (1 - mask) + new * mask
        
        self.result.paste(Image.fromarray(blended.astype(np.uint8)), (x, y))
        self.save_result(os.path.join(self.output_dir, f"result_{self.current_row}_{self.current_col}.png"))

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

# Example usage in your main script:
@torch.inference_mode()
def main(
    panorama_path: str,
    output_path: str,
    prompt: str = "a photo of sks",
    device: str = "cuda",
    num_steps: int = 50,
    guidance: float = 30.0,
    seed: int | None = None,
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
    ckpt_path = "shapes-with-sphere-pe/lora_last.safetensors"
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
    infiller = PanoramaInfiller(panorama_path, output_dir, patch_size=2)

    try:
        while (next_crop := infiller.get_next_crop()) is not None:
            if next_crop is None:
                break

            crop_info, cond_path, mask_path = next_crop
            print(f"Processing crop at position {crop_info.position}")

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
            inp["coords"] = crop_info.sphere_coords.unsqueeze(0).to(x.device)

            # Generate
            timesteps = get_schedule(num_steps, inp["img"].shape[1])
            x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)
            x = unpack(x.float(), 512, 512)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                generated = ae.decode(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Convert and update
            generated_image = torchvision.transforms.ToPILImage()(generated.squeeze(0).cpu())
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
