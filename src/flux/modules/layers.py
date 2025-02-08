import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from flux.math import attention, rope


class QuaternionEmbed(nn.Module):
    def __init__(self, dim: int = 128, theta: int = 10000):
        """
        Initialize quaternion embedding for rotations.
        Uses 4 components (w, x, y, z) for quaternion representation.

        Args:
            dim: Embedding dimension (should be multiple of 4)
            theta: Base for frequency computation
        """
        super().__init__()
        assert dim % 4 == 0, "Dimension must be multiple of 4 for quaternion encoding"
        self.dim = dim
        self.theta = theta

    def forward(self, sphere_coords: Tensor) -> Tensor:
        """
        Compute quaternion embeddings from spherical coordinates.

        Args:
            sphere_coords: Tensor of shape (..., 2) containing (lat, lon) in radians

        Returns:
            Tensor of shape (..., 4) containing quaternion components (w, x, y, z)
        """
        lat, lon = sphere_coords[..., 0], sphere_coords[..., 1]
        freq_dim = self.dim // 4

        # Scale dimensions for frequency bands
        scale = torch.arange(0, freq_dim, dtype=torch.float64,
                             device=lat.device) / freq_dim
        omega = 1.0 / (self.theta**scale)

        # Compute scaled positions
        lat_scaled = torch.einsum('...n,d->...nd', lat, omega)
        lon_scaled = torch.einsum('...n,d->...nd', lon, omega)

        # Convert spherical rotations to quaternions
        # First, create quaternions for latitude rotation around X axis
        lat_half = lat_scaled / 2.0
        sin_lat = torch.sin(lat_half)
        cos_lat = torch.cos(lat_half)
        q_lat_w = cos_lat
        q_lat_x = sin_lat
        q_lat_y = torch.zeros_like(sin_lat)
        q_lat_z = torch.zeros_like(sin_lat)

        # Then quaternions for longitude rotation around Y axis
        lon_half = lon_scaled / 2.0
        sin_lon = torch.sin(lon_half)
        cos_lon = torch.cos(lon_half)
        q_lon_w = cos_lon
        q_lon_x = torch.zeros_like(sin_lon)
        q_lon_y = sin_lon
        q_lon_z = torch.zeros_like(sin_lon)

        # Multiply quaternions to combine rotations
        # (w1, x1, y1, z1) * (w2, x2, y2, z2)
        w = q_lon_w * q_lat_w - q_lon_x * q_lat_x - \
            q_lon_y * q_lat_y - q_lon_z * q_lat_z
        x = q_lon_w * q_lat_x + q_lon_x * q_lat_w + \
            q_lon_y * q_lat_z - q_lon_z * q_lat_y
        y = q_lon_w * q_lat_y - q_lon_x * q_lat_z + \
            q_lon_y * q_lat_w + q_lon_z * q_lat_x
        z = q_lon_w * q_lat_z + q_lon_x * q_lat_y - \
            q_lon_y * q_lat_x + q_lon_z * q_lat_w

        # Stack quaternion components
        return torch.stack([w, x, y, z], dim=-1).float()


class SphericalEmbed(nn.Module):
    def __init__(self, dim: int = 3, theta: int = 10000):
        """
        Spherical coordinate embedding layer.

        Args:
            dim: Output dimension (must be multiple of 3)
            theta: Base for frequency computation
        """
        super().__init__()
        assert dim % 3 == 0, "Dimension must be multiple of 3 for spherical coordinates"
        self.dim = dim
        self.theta = theta

    def forward(self, sphere_coords: Tensor) -> Tensor:
        """
        Compute spherical embeddings.

        Args:
            sphere_coords: Tensor of shape (..., 2) containing (lat, lon) in radians

        Returns:
            Tensor of shape (..., dim//3, 3) containing rotation components
        """
        lat, lon = sphere_coords[..., 0], sphere_coords[..., 1]
        freq_dim = self.dim // 3

        # Scale dimensions
        scale = torch.arange(0, freq_dim, dtype=torch.float64,
                             device=lat.device) / freq_dim
        omega = 1.0 / (self.theta**scale)

        # Compute scaled positions
        lat_scaled = torch.einsum('...n,d->...nd', lat, omega)
        lon_scaled = torch.einsum('...n,d->...nd', lon, omega)

        # Compute trig functions
        sin_lat = torch.sin(lat_scaled)
        cos_lat = torch.cos(lat_scaled)
        sin_lon = torch.sin(lon_scaled)
        cos_lon = torch.cos(lon_scaled)

        # Stack components for multiplication
        return torch.stack([
            cos_lon,            # Component 0
            -cos_lat * sin_lon,  # Component 1
            sin_lat * sin_lon   # Component 2
        ], dim=-1).float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i],
                  self.axes_dim[i],
                  self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D",
                            K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[
            :, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor,
                pe: Tensor, sphere_pe: Tensor | None) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe, sphere_pe=sphere_pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * \
            self.img_mlp((1 + img_mod2.scale) *
                         self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * \
            self.txt_mlp((1 + txt_mod2.scale) *
                         self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor,
                sphere_pe: Tensor | None) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(
            x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D",
                            K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, sphere_pe=sphere_pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
