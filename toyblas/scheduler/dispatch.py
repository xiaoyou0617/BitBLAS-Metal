from typing import Optional, Union

import torch

from ..operations.config import MatmulConfig
from .base import MatmulScheduler


def select_scheduler(
    config: MatmulConfig,
    *,
    device: Optional[Union[str, torch.device]] = None,
    backend: str = "auto",
) -> MatmulScheduler:
    """Choose an implementation without profiling or moving input tensors.

    auto: use tiled Metal when MPS is available, otherwise use PyTorch CPU.
    torch: use PyTorch on the selected device, including the MPS baseline.
    tilelang: require MPS and use the tiled Metal implementation.
    """
    if backend not in ("auto", "torch", "tilelang"):
        raise ValueError(
            f"backend must be 'auto', 'torch', or 'tilelang'; got {backend!r}."
        )
    if device is None:
        target = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        target = torch.device(device)
    if target.type not in ("cpu", "mps") or target.index not in (None, 0):
        raise ValueError(f"Only the default CPU and MPS devices are supported; got {target}.")

    if target.type == "cpu":
        if backend == "tilelang":
            raise ValueError("The TileLang implementation requires device='mps'.")
        from .torch_cpu import TorchCPUMatmulScheduler

        return TorchCPUMatmulScheduler(config)

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable. Use device='cpu', "
            "or omit device to enable automatic CPU fallback."
        )
    if backend == "torch":
        from .torch_mps import TorchMPSMatmulScheduler

        return TorchMPSMatmulScheduler(config)

    if config.requires_dequant:
        from .weigh_dequant import WeightDequantMatmulScheduler

        return WeightDequantMatmulScheduler(config)

    from .tiled import TiledMatmulScheduler

    return TiledMatmulScheduler(config)
