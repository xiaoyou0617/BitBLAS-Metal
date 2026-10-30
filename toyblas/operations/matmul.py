from typing import Optional

from .config import MatmulConfig
from .matrix import generate_matmul
import torch

class Matmul:

    def __init__(self, config:MatmulConfig):
        self.config = config
        self.torch_A_dtype = getattr(torch, config.A_dtype)
        self.torch_W_dtype = getattr(torch, config.W_dtype)
        self.torch_out_dtype = getattr(torch, config.out_dtype)
        self.calculas = generate_matmul(config)
        self.shape_A = (config.M, config.K)
        self.shape_W = (config.K, config.N)
        self.shape_C = (config.M, config.N)

    def __call__(
        self,
        A: torch.Tensor,
        W: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute on MPS, optionally writing into an independent output buffer."""
        for name, tensor, expected_shape, expected_dtype in (
            ("A", A, self.shape_A, self.torch_A_dtype),
            ("W", W, self.shape_W, self.torch_W_dtype),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor; got {type(tensor).__name__}."
                )
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}; got {tuple(tensor.shape)}."
                )
            if tensor.dtype != expected_dtype:
                raise TypeError(
                    f"{name} must have dtype {expected_dtype}; got {tensor.dtype}."
                )
            if tensor.device.type != "mps":
                raise ValueError(
                    f"{name} must be on MPS; got device {tensor.device}. "
                    f"Move it with {name}.to('mps') before calling Matmul."
                )

        if out is None:
            C = torch.empty(
                self.shape_C,
                dtype=self.torch_out_dtype,
                device="mps",
            )
        else:
            if not isinstance(out, torch.Tensor):
                raise TypeError(
                    f"out must be a torch.Tensor or None; got {type(out).__name__}."
                )
            if tuple(out.shape) != self.shape_C:
                raise ValueError(
                    f"out must have shape {self.shape_C}; got {tuple(out.shape)}."
                )
            if out.dtype != self.torch_out_dtype:
                raise TypeError(
                    f"out must have dtype {self.torch_out_dtype}; got {out.dtype}."
                )
            if out.device.type != "mps":
                raise ValueError(
                    f"out must be on MPS; got device {out.device}. "
                    "Allocate the output buffer with device='mps'."
                )
            if not out.is_contiguous():
                raise ValueError(
                    f"out must be contiguous; got strides {out.stride()}. "
                    "Provide a contiguous MPS output buffer."
                )
            for name, tensor in (("A", A), ("W", W)):
                if out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr():
                    raise ValueError(
                        f"out must not share storage with {name}; "
                        "in-place matrix multiplication is not supported."
                    )
            C = out

        A = A.contiguous()
        W = W.contiguous()
        self.calculas(A, W, C)
        return C
