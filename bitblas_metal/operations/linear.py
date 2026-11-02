"""Inference linear layers with a bound Weight, optional bias, and reusable kernel."""

from typing import Optional

import torch

from .config import MatmulConfig
from .matmul import Matmul, _validate_bias, _validate_weight_config
from .weight import Weight


class Linear:
    """Bind prepared weights and bias to one fixed-shape Matmul implementation.

    The device comes from weight.device. Construction validates the bound inputs
    and builds once; calls accept A[M,K] and optionally an output buffer. Weight
    and bias are retained by reference, without transfer, copying, or preparation.
    Calls run without gradient tracking and reuse the existing fused-bias kernel.
    """

    def __init__(
        self,
        config: MatmulConfig,
        weight: Weight,
        *,
        bias: Optional[torch.Tensor] = None,
        backend: str = "auto",
    ):
        if not isinstance(config, MatmulConfig):
            raise TypeError("config must be a MatmulConfig.")
        if not isinstance(weight, Weight):
            raise TypeError(f"weight must be a Weight; got {type(weight).__name__}.")
        # Reject invalid bound inputs before paying for kernel construction.
        _validate_weight_config(config, weight)
        _validate_bias(config, bias, weight.device)
        self.weight = weight
        self.bias = bias
        self.matmul = Matmul(config, device=weight.device, backend=backend)

    @property
    def config(self) -> MatmulConfig:
        return self.matmul.config

    @property
    def device(self) -> torch.device:
        return self.matmul.device

    @torch.no_grad()
    def __call__(
        self, A: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute with bound weights/bias; return out when an output is supplied."""
        return self.matmul(A, self.weight, out=out, bias=self.bias)
