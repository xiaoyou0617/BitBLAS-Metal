import torch

from ..operations.pack import unpack_int4
from .base import MatmulKernel, MatmulScheduler


class TorchMPSMatmulScheduler(MatmulScheduler):
    """Torch reference with FP32 computation, scaling, and optional bias.

    Input conversions and full INT4 unpacking happen on each call.
    FP16 output is converted only after scaling and bias have been applied.
    """

    device_type = "mps"

    def build(self) -> MatmulKernel:
        compute_dtype = getattr(torch, self.config.accum_dtype)

        def compute(A, W, out, *, scales=None, bias=None):
            if self.config.requires_dequant:
                cols = self.config.N if self.config.layout == "nn" else self.config.K
                W = unpack_int4(W, original_cols=cols)
            W_compute = (W if self.config.layout == "nn" else W.T).to(compute_dtype)
            if self.config.with_scaling and self.config.group_size > 0:
                group_indices = torch.arange(self.config.K, device=W.device) // self.config.group_size
                W_compute = W_compute * scales[group_indices]
            accumulator = out if out.dtype == compute_dtype else torch.empty_like(out, dtype=compute_dtype)
            torch.mm(A.to(compute_dtype), W_compute, out=accumulator)
            if self.config.with_scaling and self.config.group_size == 0:
                accumulator.mul_(scales)
            if self.config.with_bias:
                accumulator.add_(bias)
            if accumulator is not out:
                out.copy_(accumulator)
            return out

        if self.config.with_scaling:
            if self.config.with_bias:
                def scaled_bias_matmul(A, W, scales, bias, out):
                    return compute(A, W, out, scales=scales, bias=bias)
                return scaled_bias_matmul

            def scaled_matmul(A, W, scales, out):
                return compute(A, W, out, scales=scales)
            return scaled_matmul

        if self.config.with_bias:
            def bias_matmul(A, W, bias, out):
                return compute(A, W, out, bias=bias)
            return bias_matmul

        def matmul(A, W, out):
            return compute(A, W, out)
        return matmul
