from abc import ABC, abstractmethod
from typing import Any, Callable

import torch

from ..operations.config import MatmulConfig


MatmulKernel = Callable[..., Any]


class MatmulScheduler(ABC):
    """Build (A, W, out), or (A, W, scales, out) when scaling is enabled."""

    device_type: str

    def __init__(self, config: MatmulConfig):
        self.config = config
        self.device = torch.device(self.device_type)

    @abstractmethod
    def build(self) -> MatmulKernel:
        """Build once; the callable must overwrite the supplied output buffer."""
        raise NotImplementedError
