from typing import Optional, Union

from .config import MatmulConfig
from ..scheduler.dispatch import select_scheduler
import torch

class Matmul:

    def __init__(
        self,
        config: MatmulConfig,
        *,
        device: Optional[Union[str, torch.device]] = None,
        backend: str = "auto",
    ):
        """Select a scheduler and build once; calls require inputs on self.device."""
        self.config = config
        self.torch_A_dtype = getattr(torch, config.A_dtype)
        packed_int4 = config.W_dtype == "int4"
        self.torch_W_dtype = torch.uint8 if packed_int4 else getattr(torch, config.W_dtype)
        self.torch_out_dtype = getattr(torch, config.out_dtype)
        self.scheduler = select_scheduler(config, device=device, backend=backend)
        self.device = self.scheduler.device
        self.calculas = self.scheduler.build()
        self.shape_A = (config.M, config.K)
        # Layout describes physical weights; INT4 packs their last dimension.
        rows, cols = (config.K, config.N) if config.layout == "nn" else (config.N, config.K)
        self.shape_W = (rows, (cols + 1) // 2 if packed_int4 else cols)
        self.shape_C = (config.M, config.N)
        if config.group_size > 0:
            self.shape_scales = (
                (config.K + config.group_size - 1) // config.group_size, config.N
            )
        else:
            self.shape_scales = (config.N,)

    def __call__(
        self,
        A: torch.Tensor,
        W: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute A @ W (nn) or A @ W.T (nt), optionally using FP32 scales.

        scales is required when config.with_scaling is True and must otherwise
        be omitted. Calls validate its metadata without reading device values.
        Grouping always follows logical K; scales keep the same shape in both layouts.
        With with_bias=True, bias must be FP32 [N] on the selected device.
        It is added after scaling and before the final output conversion.
        """
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
            if tensor.device.type != self.device.type:
                raise ValueError(
                    f"{name} must be on {self.device}; got device {tensor.device}. "
                    f"Move it with {name}.to('{self.device}') before calling Matmul."
                )

        if self.config.with_scaling:
            if scales is None:
                raise ValueError("scales is required when with_scaling=True.")
            if not isinstance(scales, torch.Tensor):
                raise TypeError(
                    f"scales must be a torch.Tensor; got {type(scales).__name__}."
                )
            if tuple(scales.shape) != self.shape_scales:
                raise ValueError(
                    f"scales must have shape {self.shape_scales}; "
                    f"got {tuple(scales.shape)}."
                )
            if scales.dtype != torch.float32:
                raise TypeError(f"scales must have dtype torch.float32; got {scales.dtype}.")
            if scales.device.type != self.device.type:
                raise ValueError(
                    f"scales must be on {self.device}; got device {scales.device}. "
                    f"Move it with scales.to('{self.device}') before calling Matmul."
                )
        elif scales is not None:
            raise ValueError("scales must be omitted when with_scaling=False.")

        if self.config.with_bias:
            if bias is None:
                raise ValueError("bias is required when with_bias=True.")
            if not isinstance(bias, torch.Tensor):
                raise TypeError(f"bias must be a torch.Tensor; got {type(bias).__name__}.")
            if tuple(bias.shape) != (self.config.N,):
                raise ValueError(f"bias must have shape {(self.config.N,)}; got {tuple(bias.shape)}.")
            if bias.dtype != torch.float32:
                raise TypeError(f"bias must have dtype torch.float32; got {bias.dtype}.")
            if bias.device.type != self.device.type:
                raise ValueError(
                    f"bias must be on {self.device}; got device {bias.device}. "
                    f"Move it with bias.to('{self.device}') before calling Matmul."
                )
        elif bias is not None:
            raise ValueError("bias must be omitted when with_bias=False.")

        if out is None:
            C = torch.empty(
                self.shape_C,
                dtype=self.torch_out_dtype,
                device=self.device,
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
            if out.device.type != self.device.type:
                raise ValueError(
                    f"out must be on {self.device}; got device {out.device}. "
                    f"Allocate the output buffer with device='{self.device}'."
                )
            if not out.is_contiguous():
                raise ValueError(
                    f"out must be contiguous; got strides {out.stride()}. "
                    "Provide a contiguous output buffer on the selected device."
                )
            inputs = (("A", A), ("W", W))
            if scales is not None:
                inputs += (("scales", scales),)
            if bias is not None:
                inputs += (("bias", bias),)
            for name, tensor in inputs:
                if out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr():
                    raise ValueError(
                        f"out must not share storage with {name}; "
                        "in-place matrix multiplication is not supported."
                    )
            C = out

        A = A.contiguous()
        W = W.contiguous()
        arguments = [A, W]
        if self.config.with_scaling:
            arguments.append(scales.contiguous())
        if self.config.with_bias:
            arguments.append(bias.contiguous())
        arguments.append(C)
        self.calculas(*arguments)
        return C
