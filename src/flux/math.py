import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor,
              sphere_pe: Tensor | None) -> Tensor:
    q, k = apply_rope(q, k, pe)

    if sphere_pe is not None:
        q, k = apply_quaternion_rope(q, k, sphere_pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64,
                         device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out),
                      torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def apply_spherical_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """
    Apply spherical rotation similar to regular RoPE application.

    Args:
        xq: Query tensor
        xk: Key tensor
        freqs_cis: Frequency tensor from SphericalEmbed
    """
    # Reshape inputs to match the 3 components
    *_, dim = xq.shape
    dim_per_component = dim // 3
    xq_ = xq.float().reshape(*xq.shape[:-1], 3, dim_per_component)
    xk_ = xk.float().reshape(*xk.shape[:-1], 3, dim_per_component)

    # Apply components similar to RoPE
    xq_out = (freqs_cis[..., 0] * xq_[..., 0] +
              freqs_cis[..., 1] * xq_[..., 1] +
              freqs_cis[..., 2] * xq_[..., 2])
    xq_out = sum([freqs_cis[..., i:i + 1] * xq_[..., i:i+1, :]
                 for i in range(3)])
    xk_out = sum([freqs_cis[..., i:i + 1] * xk_[..., i:i+1, :]
                 for i in range(3)])

    return (xq_out.reshape(*xq.shape).type_as(xq),
            xk_out.reshape(*xk.shape).type_as(xk))


def apply_quaternion_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """
    Apply quaternion-based rope with correct broadcasting.

    Args:
        xq: Query tensor of shape (B, H, L, D)
        xk: Key tensor of shape (B, H, L, D)
        freqs_cis: Quaternion components (B, L, 4) or (1, L, 4)
    """
    B, H, L, D = xq.shape
    assert D % 4 == 0, f"Dimension {D} must be multiple of 4"

    # Print shapes for debugging
    print(f"xq shape: {xq.shape}")
    print(f"freqs_cis shape: {freqs_cis.shape}")

    # Ensure freqs_cis has batch dimension
    if freqs_cis.shape[0] == 1:
        freqs_cis = freqs_cis.expand(B, -1, -1)

    # Reshape inputs to group dimensions into quaternion components
    xq_4d = xq.reshape(B, H, L, -1, 4)  # (B, H, L, D//4, 4)
    xk_4d = xk.reshape(B, H, L, -1, 4)  # (B, H, L, D//4, 4)

    # Expand freqs_cis for broadcasting
    # From (B, L, 4) to (B, 1, L, 1, 4)
    freqs_cis = freqs_cis.unsqueeze(1).unsqueeze(3)

    # Verify shapes before multiplication
    print(f"xq_4d shape: {xq_4d.shape}")
    print(f"freqs_cis expanded shape: {freqs_cis.shape}")

    # Apply rotation components with broadcasting
    xq_out = (xq_4d * freqs_cis).reshape(B, H, L, D)
    xk_out = (xk_4d * freqs_cis).reshape(B, H, L, D)

    return xq_out.type_as(xq), xk_out.type_as(xk)
