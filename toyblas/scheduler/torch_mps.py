import torch

from .base import MatmulKernel, MatmulScheduler


class TorchMPSMatmulScheduler(MatmulScheduler):
    """Use PyTorch's MPS implementation with the configured compute dtype.

    Input dtype conversions happen on each call and count toward runtime.
    """

    device_type = "mps"

    def build(self) -> MatmulKernel:
        compute_dtype = getattr(torch, self.config.accum_dtype)

        def matmul(A: torch.Tensor, W: torch.Tensor, out: torch.Tensor):
            A_compute = A.to(dtype=compute_dtype)
            W_compute = W.to(dtype=compute_dtype)
            return torch.mm(A_compute, W_compute, out=out)

        return matmul
