from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    SphericalEmbed, QuaternionEmbed, compute_sphere_coords_32x32,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from flux.modules.lora import LinearLora, replace_linear_with_lora
import numpy as np


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class DoubleStreamSequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]

    def forward(self, img, txt, vec, pe):
        for block in self.blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        return img, txt


class SingleStreamSequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]

    def forward(self, img, vec, pe):
        for block in self.blocks:
            img = block(img=img, vec=vec, pe=pe)
        return img


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        if True:  # use_spherical_rope:
            self.sphere_embedder = QuaternionEmbed(dim=128, theta=params.theta)
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(
                in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )
        # self.double_blocks = DoubleStreamSequential(double_blocks)

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size,
                                  self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )
        # self.single_blocks = SingleStreamSequential(single_blocks)

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        # num_patches = 32 * 32 * 4
        # self.pos_embed = nn.Parameter(torch.zeros(
        #     1, num_patches, self.hidden_size), requires_grad=False)
        # pos_embed = get_2d_sincos_pos_embed(
        #     self.pos_embed.shape[-1], int(num_patches ** 0.5))
        # self.pos_embed.data.copy_(
        #     torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        sphere_coords: Tensor | None = None
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError(
                "Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        img_ids[:, :, 1:] = img_ids[:, :, 1:] + 0.1 * sphere_coords
        # img = img + self.pos_embed.squeeze(0).chunk(4)[sphere_coords.item()].unsqueeze(0).to(img.device).to(torch.bfloat16)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        sphere_pe = None
        sphere_coords = None
        if False and sphere_coords is not None:
            sphere_coords = compute_sphere_coords_32x32(sphere_coords)
            sphere_pe = self.sphere_embedder(sphere_coords,
                                             seq_len=ids.shape[1],
                                             txt_len=txt_ids.shape[1])

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec,
                             pe=pe, sphere_pe=sphere_pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe, sphere_pe=sphere_pe)
        img = img[:, txt.shape[1]:, ...]

        # (N, T, patch_size ** 2 * out_channels)
        img = self.final_layer(img, vec)
        return img


class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
