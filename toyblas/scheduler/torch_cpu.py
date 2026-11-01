import torch

from ..operations.pack import unpack_int4
from .base import MatmulKernel, MatmulScheduler


class TorchCPUMatmulScheduler(MatmulScheduler):
    """Use PyTorch's CPU implementation with the configured compute dtype.

    Input dtype conversions happen on each call and count toward runtime.
    Packed INT4 weights are fully unpacked here as a reference implementation.
    """

    device_type = "cpu"

    def build(self) -> MatmulKernel:
        compute_dtype = getattr(torch, self.config.accum_dtype)

        def matmul(A: torch.Tensor, W: torch.Tensor, out: torch.Tensor):
            if self.config.requires_dequant:
                W = unpack_int4(W, original_cols=self.config.N)
            A_compute = A.to(dtype=compute_dtype)
            W_compute = W.to(dtype=compute_dtype)
            return torch.mm(A_compute, W_compute, out=out)

        if self.config.with_scaling and self.config.group_size > 0:
            def grouped_matmul(A, W, scales, out):
                W_compute = unpack_int4(W, original_cols=self.config.N).to(compute_dtype)
                group_indices = (
                    torch.arange(self.config.K, device=W.device) // self.config.group_size
                )
                W_compute = W_compute * scales[group_indices]
                return torch.mm(A.to(compute_dtype), W_compute, out=out)

            return grouped_matmul

        if self.config.with_scaling:
            def scaled_matmul(A, W, scales, out):
                matmul(A, W, out)
                out.mul_(scales)
                return out

            return scaled_matmul
        return matmul
