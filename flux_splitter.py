from functools import partial
import torch
from torch import nn
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class GPUSplitConfig:
    gpu_ids: List[int]
    max_params_per_gpu: int
    base_gpu: int = 0  # GPU to place non-distributed components


def wrap_forward_to_device(module: nn.Module, forward_fn, target_device: torch.device):
    """Wraps a module's forward function to ensure inputs are on the correct device."""
    def wrapped_forward(*args, **kwargs):
        # Move all tensor inputs to target device
        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(target_device)
            elif isinstance(x, tuple):
                return tuple(to_device(t) for t in x)
            elif isinstance(x, list):
                return [to_device(t) for t in x]
            return x

        new_args = tuple(to_device(arg) for arg in args)
        new_kwargs = {k: to_device(v) for k, v in kwargs.items()}

        output = forward_fn(*new_args, **new_kwargs)
        return output

    return wrapped_forward


def wrap_flux_forward(model: nn.Module, base_device: torch.device):
    """Wraps the main Flux model forward to handle initial device placement."""
    orig_forward = model.forward

    def wrapped_forward(img, img_ids, txt, txt_ids, timesteps, y, guidance=None, homo_pos_map=None):
        # Ensure inputs are on base device first
        img = img.to(base_device)
        img_ids = img_ids.to(base_device)
        txt = txt.to(base_device)
        txt_ids = txt_ids.to(base_device)
        timesteps = timesteps.to(base_device)
        y = y.to(base_device)
        if guidance is not None:
            guidance = guidance.to(base_device)
        if homo_pos_map is not None:
            homo_pos_map = homo_pos_map.to(base_device)

        return orig_forward(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            y=y,
            guidance=guidance,
            homo_pos_map=homo_pos_map
        )

    return wrapped_forward


def wrap_double_block_forward(block: nn.Module, target_device: torch.device):
    """Wraps DoubleStreamBlock forward to handle device mapping."""
    orig_forward = block.forward

    def wrapped_forward(img, txt, vec, pe):
        # Move inputs to target device
        img = img.to(target_device)
        txt = txt.to(target_device)
        vec = vec.to(target_device)
        pe = pe.to(target_device)

        output_img, output_txt = orig_forward(img=img, txt=txt, vec=vec, pe=pe)
        return output_img, output_txt

    return wrapped_forward


def wrap_single_block_forward(block: nn.Module, target_device: torch.device, is_last=False, output_device=None):
    """Wraps SingleStreamBlock forward to handle device mapping."""
    orig_forward = block.forward

    def wrapped_forward(x, vec, pe):
        # Move inputs to target device
        x = x.to(target_device)
        vec = vec.to(target_device)
        pe = pe.to(target_device)

        output = orig_forward(x, vec=vec, pe=pe)
        if is_last and output_device is not None:
            output = output.to(output_device)
        return output

    return wrapped_forward


def wrap_base_components(model: nn.Module, base_device: torch.device):
    """Wraps the forward methods of base components to ensure proper device placement."""
    if hasattr(model, 'img_in'):
        model.img_in.forward = wrap_forward_to_device(
            model.img_in, model.img_in.forward, base_device)
    if hasattr(model, 'txt_in'):
        model.txt_in.forward = wrap_forward_to_device(
            model.txt_in, model.txt_in.forward, base_device)
    if hasattr(model, 'time_in'):
        model.time_in.forward = wrap_forward_to_device(
            model.time_in, model.time_in.forward, base_device)
    if hasattr(model, 'vector_in'):
        model.vector_in.forward = wrap_forward_to_device(
            model.vector_in, model.vector_in.forward, base_device)
    if hasattr(model, 'guidance_in') and not isinstance(model.guidance_in, nn.Identity):
        model.guidance_in.forward = wrap_forward_to_device(
            model.guidance_in, model.guidance_in.forward, base_device)
    if hasattr(model, 'final_layer'):
        model.final_layer.forward = wrap_forward_to_device(
            model.final_layer, model.final_layer.forward, base_device)


def new_to_method(self, *args, **kwargs):
    """Replacement for the model's to() method to handle device distribution."""
    device_in_kwargs = 'device' in kwargs
    device_in_args = any(isinstance(arg, (str, torch.device)) for arg in args)

    device = None
    if device_in_kwargs:
        device = kwargs.pop('device')
    if device_in_args:
        args = list(args)
        for idx, arg in enumerate(args):
            if isinstance(arg, (str, torch.device)):
                device = arg
                del args[idx]
                break

    # Move base components to primary GPU
    base_device = torch.device(f"cuda:{self._base_gpu}")
    self.pe_embedder = self.pe_embedder.to(base_device, *args, **kwargs)
    self.homo_embed_h = nn.Parameter(
        self.homo_embed_h.to(base_device, *args, **kwargs))
    self.homo_embed_w = nn.Parameter(
        self.homo_embed_w.to(base_device, *args, **kwargs))
    self.img_in = self.img_in.to(base_device, *args, **kwargs)
    self.time_in = self.time_in.to(base_device, *args, **kwargs)
    self.vector_in = self.vector_in.to(base_device, *args, **kwargs)
    self.txt_in = self.txt_in.to(base_device, *args, **kwargs)

    if hasattr(self, 'guidance_in') and not isinstance(self.guidance_in, nn.Identity):
        self.guidance_in = self.guidance_in.to(base_device, *args, **kwargs)

    # Move distributed blocks to their assigned devices
    for block in self.double_blocks:
        block.to(block._target_device, *args, **kwargs)

    for block in self.single_blocks:
        block.to(block._target_device, *args, **kwargs)

    self.final_layer = self.final_layer.to(base_device, *args, **kwargs)

    return self


def split_flux_model_to_gpus(model: nn.Module, config: GPUSplitConfig):
    """
    Splits a Flux model across multiple GPUs based on parameter count.

    Args:
        model: The Flux model to split
        config: GPU split configuration
    """
    if not config.gpu_ids:
        raise ValueError("Must provide at least one GPU ID")

    # Set base GPU for non-distributed components
    model._base_gpu = config.base_gpu
    base_device = torch.device(f"cuda:{config.base_gpu}")

    # Wrap the main model's forward to ensure proper initial device placement
    model.forward = wrap_flux_forward(model, base_device)

    # Wrap base components
    wrap_base_components(model, base_device)

    # Calculate total parameters for distributable components
    total_params = 0
    for block in model.double_blocks:
        total_params += sum(p.numel() for p in block.parameters())
    for block in model.single_blocks:
        total_params += sum(p.numel() for p in block.parameters())

    params_per_gpu = min(config.max_params_per_gpu,
                         total_params // len(config.gpu_ids) + 1)

    current_gpu_idx = 0
    current_gpu_params = 0

    # Distribute double blocks
    for block in model.double_blocks:
        device = torch.device(f"cuda:{config.gpu_ids[current_gpu_idx]}")
        block._target_device = device
        block.forward = wrap_double_block_forward(block, device)

        block_params = sum(p.numel() for p in block.parameters())
        current_gpu_params += block_params

        if current_gpu_params >= params_per_gpu and current_gpu_idx < len(config.gpu_ids) - 1:
            current_gpu_idx += 1
            current_gpu_params = 0

    # Distribute single blocks
    for i, block in enumerate(model.single_blocks):
        device = torch.device(f"cuda:{config.gpu_ids[current_gpu_idx]}")
        block._target_device = device
        is_last = (i == len(model.single_blocks) - 1)
        block.forward = wrap_single_block_forward(
            block, device,
            is_last=is_last,
            output_device=base_device if is_last else None
        )

        block_params = sum(p.numel() for p in block.parameters())
        current_gpu_params += block_params

        if current_gpu_params >= params_per_gpu and current_gpu_idx < len(config.gpu_ids) - 1:
            current_gpu_idx += 1
            current_gpu_params = 0

    # Replace the model's to() method
    model._original_to = model.to
    model.to = partial(new_to_method, model)

    return model


# Usage example:
"""
config = GPUSplitConfig(
    gpu_ids=[0, 1, 2, 3],  # List of GPU IDs to use
    max_params_per_gpu=5e9,  # Maximum parameters per GPU
    base_gpu=0  # GPU to place non-distributed components
)

model = Flux(params)  # Your Flux model
model = split_flux_model_to_gpus(model, config)
"""
