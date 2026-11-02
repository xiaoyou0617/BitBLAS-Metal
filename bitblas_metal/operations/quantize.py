from typing import Tuple

import torch


@torch.no_grad()
def quantize_int4(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize floating weights W[K, N] with one symmetric scale per column.

    Returns unpacked int8 codes qW[K, N] in [-7, 7] and float32 scales[N],
    both on W.device. Reconstruct weights with qW.float() * scales.
    FP32 arithmetic and round-to-nearest, ties-to-even are used throughout.
    W always uses logical [K,N], independently of MatmulConfig.layout.
    For nt matmul, pack qW.T so its physical storage is [N,ceil(K/2)].
    All-zero columns use scale 1. A nonzero scale must be representable
    as a positive FP32 value. Gradients are not tracked during preparation.
    """
    if not isinstance(W, torch.Tensor):
        raise TypeError(f"W must be a torch.Tensor; got {type(W).__name__}.")
    if W.dtype not in (torch.float16, torch.float32):
        raise TypeError(
            f"W must have dtype torch.float16 or torch.float32; got {W.dtype}."
        )
    if W.ndim != 2:
        raise ValueError(f"W must be a 2-D tensor; got shape {tuple(W.shape)}.")
    if W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError(f"W must have nonzero rows and columns; got {tuple(W.shape)}.")

    W_fp32 = W.float()
    if not torch.isfinite(W_fp32).all().item():
        raise ValueError("W must contain only finite values; NaN and infinity are unsupported.")

    amax = W_fp32.abs().amax(dim=0)
    scales = torch.where(amax == 0, torch.ones_like(amax), amax / 7.0)
    if not (scales > 0).all().item():
        raise ValueError("A nonzero column's scale underflows to zero in float32.")

    scaled = W_fp32 / scales
    # Torch 2.8 MPS rounds exact halves away from zero. Handle ties explicitly
    # so CPU and MPS both use the nearest even integer. scaled is in [-7, 7].
    lower = torch.floor(scaled)
    halfway = (scaled - lower) == 0.5
    nearest_even = lower + (lower.to(torch.int32) & 1).to(torch.float32)
    rounded = torch.where(halfway, nearest_even, torch.round(scaled))
    # Cast only after rounding; a direct float-to-int cast would truncate.
    qW = rounded.clamp(-7, 7).to(torch.int8)
    return qW, scales


@torch.no_grad()
def quantize_int4_with_grouping(
    W: torch.Tensor, group_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize W[K, N] with one FP32 scale per K-group and output column.

    group_size must be a positive integer. Returns int8 qW[K, N] in [-7, 7]
    and float32 scales[ceil(K / group_size), N] on W.device. The last group
    is zero-padded for reduction; padded rows are removed from qW.
    All-zero group/column slices use scale 1. Rounding is nearest-even.
    Input and qW always use logical [K,N]. For nt matmul, pack qW.T;
    scales remain [groups,N] and grouping still follows logical K.
    """
    if not isinstance(W, torch.Tensor):
        raise TypeError(f"W must be a torch.Tensor; got {type(W).__name__}.")
    if W.dtype not in (torch.float16, torch.float32):
        raise TypeError(
            f"W must have dtype torch.float16 or torch.float32; got {W.dtype}."
        )
    if W.ndim != 2:
        raise ValueError(f"W must be a 2-D tensor; got shape {tuple(W.shape)}.")
    if W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError(f"W must have nonzero rows and columns; got {tuple(W.shape)}.")
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError(f"group_size must be an integer; got {group_size!r}.")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive; got {group_size}.")

    W_fp32 = W.float()
    if not torch.isfinite(W_fp32).all().item():
        raise ValueError("W must contain only finite values; NaN and infinity are unsupported.")

    K, N = W_fp32.shape
    pad_len = (-K) % group_size
    if pad_len:
        padding = torch.zeros((pad_len, N), dtype=torch.float32, device=W.device)
        W_pad = torch.cat((W_fp32, padding), dim=0)
    else:
        W_pad = W_fp32
    num_groups = W_pad.shape[0] // group_size
    W_grouped = W_pad.reshape(num_groups, group_size, N)

    abs_max_val = W_grouped.abs().amax(dim=1)
    scales = torch.where(abs_max_val == 0, torch.ones_like(abs_max_val), abs_max_val / 7.0)
    if not (scales > 0).all().item():
        raise ValueError("A nonzero group/column scale underflows to zero in float32.")

    # [groups, group_size, N] / [groups, 1, N].
    scaled = W_grouped / scales[:, None, :]
    # Preserve the same explicit nearest-even rule on CPU and Torch 2.8 MPS.
    lower = torch.floor(scaled)
    halfway = (scaled - lower) == 0.5
    nearest_even = lower + (lower.to(torch.int32) & 1).to(torch.float32)
    rounded = torch.where(halfway, nearest_even, torch.round(scaled))
    qW_grouped = rounded.clamp(-7, 7).to(torch.int8)
    qW = qW_grouped.reshape(-1, N)[:K]
    return qW, scales
