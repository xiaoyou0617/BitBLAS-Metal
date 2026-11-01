import torch


def pack_int4(W: torch.Tensor) -> torch.Tensor:
    """Pack a 2-D tensor of signed INT4 codes into uint8 bytes.

    W must have dtype int8 and values in [-8, 7]. Packing is along columns:
    the first value occupies the low four bits, the second the high four.
    An odd final column is padded with zero in the high four bits.
    The output has shape (rows, ceil(cols / 2)) and stays on W.device.
    Batched Torch operations run on the input device. Range validation
    reads one boolean on the host, synchronizing once for MPS inputs.
    """
    if not isinstance(W, torch.Tensor):
        raise TypeError(f"W must be a torch.Tensor; got {type(W).__name__}.")
    if W.dtype != torch.int8:
        raise TypeError(f"W must have dtype torch.int8; got {W.dtype}.")
    if W.ndim != 2:
        raise ValueError(f"W must be a 2-D tensor; got shape {tuple(W.shape)}.")
    if torch.any((W < -8) | (W > 7)).item():
        raise ValueError("W values must be in the signed INT4 range [-8, 7].")

    cols = W.shape[1]
    # Widen before shifting so all byte values fit without signed overflow.
    lo = W[:, 0::2].to(torch.int16) & 0xF
    hi = W[:, 1::2].to(torch.int16) & 0xF
    # For odd cols, the final low code already has a zero high nibble.
    lo[:, :cols // 2] |= hi << 4
    return lo.to(torch.uint8)


def unpack_int4(packed_W: torch.Tensor, original_cols: int) -> torch.Tensor:
    """Restore signed INT4 codes to a 2-D int8 tensor on the same device.

    original_cols is the number of columns before packing, not the total
    number of elements. It distinguishes padding from a real trailing zero.
    Bit extraction and signed restoration run in batches on the input device.
    """
    if not isinstance(packed_W, torch.Tensor):
        raise TypeError(
            f"packed_W must be a torch.Tensor; got {type(packed_W).__name__}."
        )
    if packed_W.dtype != torch.uint8:
        raise TypeError(
            f"packed_W must have dtype torch.uint8; got {packed_W.dtype}."
        )
    if packed_W.ndim != 2:
        raise ValueError(
            f"packed_W must be a 2-D tensor; got shape {tuple(packed_W.shape)}."
        )
    if isinstance(original_cols, bool) or not isinstance(original_cols, int):
        raise TypeError(
            f"original_cols must be an integer; got {original_cols!r}."
        )
    if original_cols < 0:
        raise ValueError(f"original_cols must be nonnegative; got {original_cols}.")
    expected_packed_cols = (original_cols + 1) // 2
    if packed_W.shape[1] != expected_packed_cols:
        raise ValueError(
            f"packed_W must have {expected_packed_cols} columns for "
            f"original_cols={original_cols}; got {packed_W.shape[1]}."
        )

    W = torch.empty(
        (packed_W.shape[0], original_cols),
        dtype=torch.int8,
        device=packed_W.device,
    )
    codes = packed_W.to(torch.int16)
    lo = codes & 0xF
    hi = (codes >> 4)[:, :original_cols // 2]
    # Codes 8..15 represent signed values -8..-1.
    W[:, 0::2] = torch.where(lo >= 8, lo - 16, lo)
    W[:, 1::2] = torch.where(hi >= 8, hi - 16, hi)

    return W
