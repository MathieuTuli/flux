from dataclasses import dataclass
from pathlib import Path

from flux.modules.autoencoder import AutoEncoder
from flux.modules.conditioner import HFEmbedder
from torch.utils.data import Dataset
from einops import rearrange, repeat
from PIL import Image

import numpy as np
import torchvision
import random
import torch
import math


class OptimalTransportPath:
    def __init__(self,
                 sig_min: float = 1e-5,
                 num_timesteps: int = 1000) -> None:
        self.sig_min = sig_min
        self.num_timesteps = num_timesteps
        self.timesteps = torch.linspace(1, 0, num_timesteps + 1)

        # DEPRECATE:
        # m = (1.15 - 0.5) / (4096 - 256)
        # b = 0.5 - m * 256
        # def mu(x): return m * x + b
        # mu = mu((1024 // 8) * (1024 // 8) // 4)
        # self.timesteps = math.exp(mu) / math.exp(mu) + (1 / self.timesteps - 1) ** 1.

    def get_timesteps(self, batch_size: int):
        indices = torch.randint(
            low=0, high=self.num_timesteps, size=(batch_size,))
        t = self.timesteps[indices]
        t = t.to("cuda:0")
        return t

    def sample(self,
               x1: torch.Tensor,
               x0: torch.Tensor,
               t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # REVISIT:xlabs says to flip these
        t = t.view(-1, 1, 1)
        xt = x0 * t + (1 - (1 - self.sig_min) * t) * x1
        vt = x0 - (1 - self.sig_min) * x1
        return xt, vt


class FluxFillDataset(Dataset):
    def __init__(self,
                 root: Path | str,
                 prompt: str = "a photo of sks",
                 height: int = 512,
                 width: int = 512):
        root = root if isinstance(root, Path) else Path(root)
        self.image_paths = [f for f in root.iterdir()]
        self.prompt = prompt
        self.height = height
        self.width = width
        # REVISIT:
        # - setting rotation to 0 for now
        # - padding? what's that

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float32),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )
        if True:
            self.random_rotate_crop = RandomRotatedCrop(
                crop_size=(512, 512),
                max_rotation=0,
                padding=64
            )
        else:
            self.random_rotate_crop = GridRotatedCrop(
                crop_size=(512, 512),
                stride=(256, 256),
                max_rotation=0,
                padding=64
            )
        self.sphere_mapper = PanoramaSphereMapper(
            SphereConfig(patch_size=2, target_size=(512, 512))
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # REVISIT:
        # - basically, the panorama should already be "noramlized"?
        # - it should be a sparse reconstruction on the unit sphere
        # - from there, you can get a mask if it exceeds limits
        # - but it will also wrap and then the crop should properly wrap too
        panorama = Image.open(img_path).convert("RGB")
        w, h = panorama.size
        assert w > self.width and h > self.height
        panorama = self.transform(panorama)
        print(panorama.shape)
        # panorama = rearrange(panorama, "h w c -> c h w")

        cropped_img, rotation, crop_pos = self.random_rotate_crop(panorama)

        sphere_coords = self.sphere_mapper.get_coords_for_crop(
            panorama_size=(h, w),
            crop_size=(512, 512),
            crop_position=crop_pos,
            rotation=rotation
        )

        # Compute spherical coordinates for the patches
        sphere_coords = self.sphere_mapper.get_nearest_sphere_patches(
            sphere_coords)

        mask = make_rec_mask(cropped_img.unsqueeze(0),
                             resolution=512, times=30).squeeze(0)

        return cropped_img, mask, sphere_coords


def make_rec_mask(images, resolution, times=30, drop_mask_prob: float = 0.5):
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

    return 1 - mask if random.random() < drop_mask_prob else mask


def get_rotated_crop_params(
    img_size: tuple[int, int],
    crop_size: tuple[int, int],
    rotation: float,
    crop_position: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute affine transformation matrix for rotation and cropping.

    Args:
        img_size: Original image size (H, W)
        crop_size: Size of crop (H, W)
        rotation: Rotation in degrees
        crop_position: Position of crop top-left corner (y, x)

    Returns:
        Affine transformation matrices for torch.nn.functional.affine_grid
    """
    H, W = img_size
    crop_h, crop_w = crop_size
    crop_y, crop_x = crop_position

    # Convert rotation to radians
    rot_rad = math.radians(rotation)
    cos_rot = math.cos(rot_rad)
    sin_rot = math.sin(rot_rad)

    # Center of the crop in the original image
    center_y = crop_y + crop_h / 2
    center_x = crop_x + crop_w / 2

    # Compute the affine transformation matrix
    # First translate to origin, then rotate, then translate back
    theta = torch.tensor([
        [cos_rot, -sin_rot, 0],
        [sin_rot, cos_rot, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    # Scale factors to normalize coordinates to [-1, 1]
    scale_y = crop_h / H * 2
    scale_x = crop_w / W * 2

    # Translation to center the crop
    trans_x = -(2 * center_x / W - 1)
    trans_y = -(2 * center_y / H - 1)

    # Combine transformations
    scale = torch.tensor([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])

    trans = torch.tensor([
        [1, 0, trans_x],
        [0, 1, trans_y],
        [0, 0, 1]
    ])

    # Combine transformations: scale * rotation * translation
    transform = scale @ theta @ trans

    # Convert to the format expected by affine_grid
    affine_mat = transform[:2, :].unsqueeze(0)

    return affine_mat


def apply_rotated_crop(
    img: torch.Tensor,
    crop_size: tuple[int, int],
    rotation: float,
    crop_position: tuple[int, int]
) -> torch.Tensor:
    """
    Apply rotation and cropping to an image.

    Args:
        img: Input image tensor (B, C, H, W)
        crop_size: Size of crop (H, W)
        rotation: Rotation in degrees
        crop_position: Position of crop top-left corner (y, x)

    Returns:
        Cropped and rotated image tensor (B, C, crop_H, crop_W)
    """
    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    B, C, H, W = img.shape
    crop_h, crop_w = crop_size

    # Get transformation matrix
    theta = get_rotated_crop_params(
        img_size=(H, W),
        crop_size=crop_size,
        rotation=rotation,
        crop_position=crop_position
    ).to(img.device)

    # Create sampling grid
    grid = torch.nn.functional.affine_grid(
        theta.expand(B, -1, -1),
        size=(B, C, crop_h, crop_w),
        align_corners=False
    )

    # Apply transformation
    out = torch.nn.functional.grid_sample(
        img,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    return out


class GridRotatedCrop:
    def __init__(
        self,
        crop_size: tuple[int, int],
        stride: tuple[int, int] | None = None,  # If None, use crop_size//2
        max_rotation: float = 180.0,
        padding: int | tuple[int, int] = 0
    ):
        """
        Grid-based random rotation and crop transformation.

        Args:
            crop_size: Size of crop (H, W)
            stride: Size of grid stride (H, W). If None, uses crop_size//2
            max_rotation: Maximum rotation in degrees
            padding: Padding to apply to avoid empty regions after rotation
        """
        self.crop_size = crop_size
        self.stride = stride or (crop_size[0]//2, crop_size[1]//2)
        self.max_rotation = max_rotation
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Initialize grid positions
        self.available_positions = []
        self.current_image_size = None

    def _init_grid(self, img_size: tuple[int, int]):
        """Initialize grid positions for a given image size."""
        H, W = img_size
        crop_h, crop_w = self.crop_size
        stride_h, stride_w = self.stride
        pad_y, pad_x = self.padding

        # Calculate valid start positions
        y_positions = list(range(
            pad_y,
            H - crop_h - pad_y,
            stride_h
        ))
        x_positions = list(range(
            pad_x,
            W - crop_w - pad_x,
            stride_w
        ))

        # Create all grid positions
        self.available_positions = [
            (y, x) for y in y_positions for x in x_positions
        ]

        # Shuffle positions
        torch.manual_seed(torch.randint(0, 2**32, (1,)).item())
        torch.randperm(len(self.available_positions))

        self.current_image_size = img_size

    def get_params(self, img_size: tuple[int, int]) -> tuple[float, tuple[int, int]]:
        """Get random rotation and next grid position."""
        # Initialize or reinitialize grid if needed
        if self.current_image_size != img_size or not self.available_positions:
            self._init_grid(img_size)

        # Get next position from grid
        crop_pos = self.available_positions.pop()

        # Random rotation
        rotation = torch.rand(1).item() * 2 * \
            self.max_rotation - self.max_rotation

        return rotation, crop_pos

    def __call__(self, img: torch.Tensor) -> tuple[torch.Tensor, float, tuple[int, int]]:
        """
        Apply random rotation and grid-based crop.

        Args:
            img: Input image tensor (C, H, W)

        Returns:
            tuple of:
                - Transformed image tensor (C, crop_H, crop_W)
                - Rotation angle
                - Crop position
        """
        H, W = img.shape[-2:]
        rotation, crop_pos = self.get_params((H, W))

        transformed = apply_rotated_crop(
            img,
            self.crop_size,
            rotation,
            crop_pos
        )

        return transformed.squeeze(0), rotation, crop_pos


class RandomRotatedCrop:
    def __init__(
        self,
        crop_size: tuple[int, int],
        max_rotation: float = 180.0,
        padding: int | tuple[int, int] = 0
    ):
        """
        Random rotation and crop transformation.

        Args:
            crop_size: Size of crop (H, W)
            max_rotation: Maximum rotation in degrees
            padding: Padding to apply to avoid empty regions after rotation
        """
        self.crop_size = crop_size
        self.max_rotation = max_rotation
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def get_params(self, img_size: tuple[int, int]) -> tuple[float, tuple[int, int]]:
        """Get random rotation and crop position"""
        H, W = img_size
        crop_h, crop_w = self.crop_size
        pad_y, pad_x = self.padding

        # Random rotation
        rotation = torch.rand(1).item() * 2 * \
            self.max_rotation - self.max_rotation

        # Random position (accounting for padding)
        max_y = H - crop_h - 2 * pad_y
        max_x = W - crop_w - 2 * pad_x
        crop_y = torch.randint(pad_y, max_y + 1, (1,)).item()
        crop_x = torch.randint(pad_x, max_x + 1, (1,)).item()

        return rotation, (crop_y, crop_x)

    def __call__(self, img: torch.Tensor) -> tuple[torch.Tensor, float, tuple[int, int]]:
        """
        Apply random rotation and crop.

        Args:
            img: Input image tensor (C, H, W)

        Returns:
            tuple of:
                - Transformed image tensor (C, crop_H, crop_W)
                - Rotation angle
                - Crop position
        """
        H, W = img.shape[-2:]
        rotation, crop_pos = self.get_params((H, W))

        transformed = apply_rotated_crop(
            img,
            self.crop_size,
            rotation,
            crop_pos
        )

        return transformed.squeeze(0), rotation, crop_pos


@dataclass
class SphereConfig:
    """Configuration for sphere discretization"""
    patch_size: int  # Size of each patch
    target_size: tuple[int, int]  # Size expected by model (e.g., 512x512)


class PanoramaSphereMapper:
    def __init__(self, sphere_config: SphereConfig):
        """
        Initialize sphere mapper for panorama images.

        Args:
            sphere_config: Configuration specifying patch size and target dimensions
        """
        self.config = sphere_config

        # Compute grid for the entire sphere based on patch size
        # This establishes our "fixed" sphere discretization
        self.lat_grid, self.lon_grid = self._create_sphere_grid()

    def _create_sphere_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create evenly spaced grid of latitude and longitude for the entire sphere"""
        # Calculate number of patches needed for target size
        n_patches_h = self.config.target_size[0] // self.config.patch_size
        n_patches_w = self.config.target_size[1] // self.config.patch_size

        # Create latitude/longitude grids
        # Latitude: -π/2 to π/2 (south pole to north pole)
        # Longitude: -π to π
        lats = torch.linspace(-math.pi/2, math.pi/2, n_patches_h)
        lons = torch.linspace(-math.pi, math.pi, n_patches_w)

        # Create grid
        lat_grid, lon_grid = torch.meshgrid(lats, lons, indexing='ij')
        return lat_grid, lon_grid

    def get_coords_for_crop(
        self,
        panorama_size: tuple[int, int],
        crop_size: tuple[int, int],
        crop_position: tuple[int, int],
        rotation: float
    ) -> torch.Tensor:
        """
        Get sphere coordinates for a rotated crop of the panorama.

        Args:
            panorama_size: Size of original panorama (H, W)
            crop_size: Size of crop (H, W)
            crop_position: Position of crop top-left corner (y, x)
            rotation: Rotation in degrees

        Returns:
            Tensor of shape (n_patches_h, n_patches_w, 2) containing (lat, lon) in radians
        """
        H, W = panorama_size
        crop_h, crop_w = crop_size
        crop_y, crop_x = crop_position
        patch_size = self.config.patch_size

        # Number of patches in the crop
        n_patches_h = crop_h // patch_size
        n_patches_w = crop_w // patch_size

        # Get center points of each patch in crop coordinates
        patch_centers_y = torch.arange(
            n_patches_h) * patch_size + patch_size / 2
        patch_centers_x = torch.arange(
            n_patches_w) * patch_size + patch_size / 2
        patch_grid_y, patch_grid_x = torch.meshgrid(
            patch_centers_y, patch_centers_x, indexing='ij')

        # Apply rotation around crop center
        rot_rad = math.radians(rotation)
        cos_rot = math.cos(rot_rad)
        sin_rot = math.sin(rot_rad)

        # Center of rotation is the center of the crop
        center_y = crop_h / 2
        center_x = crop_w / 2

        # Translate to origin, rotate, translate back
        y_rot = (patch_grid_y - center_y) * cos_rot - \
            (patch_grid_x - center_x) * sin_rot + center_y
        x_rot = (patch_grid_y - center_y) * sin_rot + \
            (patch_grid_x - center_x) * cos_rot + center_x

        # Add crop offset to get panorama coordinates
        y_global = y_rot + crop_y
        x_global = x_rot + crop_x

        # Convert to normalized coordinates (0 to 1)
        y_norm = y_global / H
        x_norm = x_global / W

        # Convert to spherical coordinates
        # Map y from [0, 1] to [-π/2, π/2] (latitude)
        # flip y since image coords are top-down
        lat = (1 - y_norm) * math.pi - math.pi/2
        # Map x from [0, 1] to [-π, π] (longitude)
        lon = x_norm * 2 * math.pi - math.pi

        # Stack coordinates
        coords = torch.stack([lat, lon], dim=-1)

        return coords

    def get_nearest_sphere_patches(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Find nearest pre-computed sphere patches for given coordinates.

        Args:
            coords: Tensor of shape (..., 2) containing (lat, lon) in radians

        Returns:
            Tensor of same shape with coordinates snapped to nearest sphere patches
        """
        lat, lon = coords[..., 0], coords[..., 1]

        # Find nearest latitude and longitude indices
        lat_idx = torch.searchsorted(
            self.lat_grid[:, 0].contiguous(), lat[..., None].contiguous())
        lat_idx = torch.clamp(lat_idx, 0, self.lat_grid.shape[0] - 1)

        lon_idx = torch.searchsorted(
            self.lon_grid[0, :].contiguous(), lon[..., None].contiguous())
        lon_idx = torch.clamp(lon_idx, 0, self.lon_grid.shape[1] - 1)

        # Get corresponding coordinates
        nearest_lat = self.lat_grid[lat_idx, 0]
        nearest_lon = self.lon_grid[0, lon_idx]

        return torch.stack([nearest_lat, nearest_lon], dim=-1)


def prepare_fill_for_train(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: torch.Tensor,
    mask: torch.Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    path,
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
        mask = rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w",
                         ph=8, pw=8,)
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                         ph=2, pw=2)

    B, C, H, W = img.shape
    img = img.to(torch.bfloat16)
    mask_img = mask_img.to(torch.bfloat16)
    img = rearrange(
        img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    mask_img = rearrange(
        mask_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    # NOTE: this is now the actual img cond
    img_cond = torch.cat((mask_img, mask), dim=-1)

    x0 = torch.randn_like(img, device=img.device)
    t = path.get_timesteps(B)
    xt, vt = path.sample(img, x0, t)

    # -- basic prepares
    return_dict = dict()

    img_ids = torch.zeros(H // 2, W // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(H // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(W // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=B)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and B > 1:
        txt = repeat(txt, "1 ... -> B ...", B=B)
    txt_ids = torch.zeros(B, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and B > 1:
        vec = repeat(vec, "1 ... -> B ...", B=B)

    return_dict["vec"] = vec.to(img.device)
    return_dict["t"] = t
    return_dict["vt"] = vt
    return_dict["img"] = xt
    return_dict["img_cond"] = img_cond
    return_dict["txt"] = txt.to(img.device)
    return_dict["img_ids"] = img_ids.to(img.device)
    return_dict["txt_ids"] = txt_ids.to(img.device)
    return return_dict
