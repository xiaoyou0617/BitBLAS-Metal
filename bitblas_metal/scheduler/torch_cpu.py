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

        def output_accumulator(out):
            # FP16 output must not round intermediate sums or unscaled results.
            return out if out.dtype == compute_dtype else torch.empty_like(out, dtype=compute_dtype)

        def matmul(A: torch.Tensor, W: torch.Tensor, out: torch.Tensor):
            if self.config.requires_dequant:
                W = unpack_int4(W, original_cols=self.config.N)
            A_compute = A.to(dtype=compute_dtype)
            W_compute = W.to(dtype=compute_dtype)
            accumulator = output_accumulator(out)
            torch.mm(A_compute, W_compute, out=accumulator)
            if accumulator is not out:
                out.copy_(accumulator)
            return out

        if self.config.with_scaling and self.config.group_size > 0:
            def grouped_matmul(A, W, scales, out):
                W_compute = unpack_int4(W, original_cols=self.config.N).to(compute_dtype)
                group_indices = (
                    torch.arange(self.config.K, device=W.device) // self.config.group_size
                )
                W_compute = W_compute * scales[group_indices]
                accumulator = output_accumulator(out)
                torch.mm(A.to(compute_dtype), W_compute, out=accumulator)
                if accumulator is not out:
                    out.copy_(accumulator)
                return out

            return grouped_matmul

        if self.config.with_scaling:
            def scaled_matmul(A, W, scales, out):
                accumulator = output_accumulator(out)
                matmul(A, W, accumulator)
                accumulator.mul_(scales)
                if accumulator is not out:
                    out.copy_(accumulator)
                return out

            return scaled_matmul
        return matmul
