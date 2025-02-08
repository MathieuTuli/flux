import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor,
              sphere_pe: Tensor | None) -> Tensor:
    q, k = apply_rope(q, k, pe)

    if sphere_pe is not None:
        q, k = apply_spherical_rope(q, k, sphere_pe)

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
    import pdb; pdb.set_trace()
    *_, dim = xq.shape
    dim_per_component = dim // 3
    xq_ = xq.float().reshape(*xq.shape[:-1], 3, dim_per_component)
    xk_ = xk.float().reshape(*xk.shape[:-1], 3, dim_per_component)

    # Apply components similar to RoPE
    xq_out = (freqs_cis[..., 0] * xq_[..., 0] +
              freqs_cis[..., 1] * xq_[..., 1] +
              freqs_cis[..., 2] * xq_[..., 2])
    xq_out = sum([freqs_cis[..., i:i + 1] * xq_[..., i:i+1, :] for i in range(3)])
    xk_out = sum([freqs_cis[..., i:i + 1] * xk_[..., i:i+1, :] for i in range(3)])

    return (xq_out.reshape(*xq.shape).type_as(xq),
            xk_out.reshape(*xk.shape).type_as(xk))
