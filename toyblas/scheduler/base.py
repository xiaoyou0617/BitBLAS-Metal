from abc import ABC, abstractmethod
from typing import Any, Callable

import torch

from ..operations.config import MatmulConfig


MatmulKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any]


class MatmulScheduler(ABC):
    """Build a reusable callable with the argument order (A, W, out)."""

    device_type: str

    def __init__(self, config: MatmulConfig):
        self.config = config
        self.device = torch.device(self.device_type)

    @abstractmethod
    def build(self) -> MatmulKernel:
        """Build once; the callable must overwrite the supplied output buffer."""
        raise NotImplementedError
