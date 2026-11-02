"""Prepare and keep weight data together with its interpretation metadata."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch

from .config import MatmulConfig
from .pack import pack_int4, unpack_int4
from .quantize import quantize_int4, quantize_int4_with_grouping


@dataclass(frozen=True, init=False, eq=False)
class Weight:
    """Reusable dense or packed weights, with optional matching FP32 scales.

    The constructor accepts floating weights in logical [K,N] order, regardless
    of config.layout. Preparation stays on the source device and runs once.
    Only weight-format metadata is retained; M, tiles, and output types do not
    restrict reuse. Metadata and tensor attributes are read-only, but the
    underlying tensors are still mutable and may share storage with inputs.
    """

    data: torch.Tensor = field(repr=False)
    scales: Optional[torch.Tensor] = field(repr=False)
    logical_shape: Tuple[int, int]
    logical_dtype: str
    layout: str
    group_size: int

    @torch.no_grad()
    def __init__(self, W_float: torch.Tensor, config: MatmulConfig):
        if not isinstance(config, MatmulConfig):
            raise TypeError("config must be a MatmulConfig.")
        if not isinstance(W_float, torch.Tensor):
            raise TypeError(
                f"W_float must be a torch.Tensor; got {type(W_float).__name__}."
            )
        if W_float.dtype not in (torch.float16, torch.float32):
            raise TypeError(
                "W_float must have dtype torch.float16 or torch.float32; "
                f"got {W_float.dtype}."
            )
        if tuple(W_float.shape) != (config.K, config.N):
            raise ValueError(
                f"W_float must have logical shape {(config.K, config.N)}; "
                f"got {tuple(W_float.shape)}."
            )

        scales = None
        if config.W_dtype == "int4":
            if not config.with_scaling:
                raise ValueError(
                    "Preparing floating weights as INT4 requires with_scaling=True. "
                    "For raw INT4 codes, use pack_int4 and Weight.from_prepared."
                )
            if config.group_size > 0:
                codes, scales = quantize_int4_with_grouping(W_float, config.group_size)
            else:
                codes, scales = quantize_int4(W_float)
            stored_codes = codes if config.layout == "nn" else codes.T
            data = pack_int4(stored_codes)
        else:
            data = W_float.to(getattr(torch, config.W_dtype))
            if config.layout == "nt":
                data = data.T
        self._initialize(data, config, scales)

    @classmethod
    def from_prepared(
        cls,
        data: torch.Tensor,
        config: MatmulConfig,
        *,
        scales: Optional[torch.Tensor] = None,
    ) -> "Weight":
        """Wrap physical data without quantizing, transposing, or packing it.

        Data must already follow config.layout. INT4 data must be packed uint8;
        scales must be FP32 [N] or [ceil(K/group_size),N] on the same device.
        Metadata is checked and noncontiguous tensors are made contiguous.
        Tensor values are not inspected; callers supply matching data/scales.
        Contiguous inputs may share storage with the returned Weight.
        """
        weight = cls.__new__(cls)
        weight._initialize(data, config, scales)
        return weight

    def _initialize(self, data, config, scales):
        if not isinstance(config, MatmulConfig):
            raise TypeError("config must be a MatmulConfig.")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be a torch.Tensor; got {type(data).__name__}.")
        if data.device.type not in ("cpu", "mps"):
            raise ValueError(f"Weight supports CPU and MPS data; got {data.device}.")
        rows, cols = ((config.K, config.N) if config.layout == "nn"
                      else (config.N, config.K))
        packed = config.W_dtype == "int4"
        expected_shape = (rows, (cols + 1) // 2 if packed else cols)
        expected_dtype = torch.uint8 if packed else getattr(torch, config.W_dtype)
        if tuple(data.shape) != expected_shape:
            raise ValueError(
                f"data must have physical shape {expected_shape} for "
                f"layout={config.layout!r}; got {tuple(data.shape)}."
            )
        if data.dtype != expected_dtype:
            raise TypeError(f"data must have dtype {expected_dtype}; got {data.dtype}.")

        if config.with_scaling:
            if scales is None:
                raise ValueError("scales is required when with_scaling=True.")
            if not isinstance(scales, torch.Tensor):
                raise TypeError(
                    f"scales must be a torch.Tensor; got {type(scales).__name__}."
                )
            expected_scales = (
                ((config.K + config.group_size - 1) // config.group_size, config.N)
                if config.group_size > 0 else (config.N,)
            )
            if tuple(scales.shape) != expected_scales:
                raise ValueError(
                    f"scales must have shape {expected_scales}; got {tuple(scales.shape)}."
                )
            if scales.dtype != torch.float32:
                raise TypeError(f"scales must have dtype torch.float32; got {scales.dtype}.")
            if scales.device != data.device:
                raise ValueError(
                    f"scales must be on the same device as data ({data.device}); "
                    f"got {scales.device}."
                )
        elif scales is not None:
            raise ValueError("scales must be omitted when with_scaling=False.")

        for name, value in (
            ("data", data.detach().contiguous()),
            ("scales", scales.detach().contiguous() if scales is not None else None),
            ("logical_shape", (config.K, config.N)),
            ("logical_dtype", config.W_dtype),
            ("layout", config.layout),
            ("group_size", config.group_size),
        ):
            object.__setattr__(self, name, value)

    @property
    def with_scaling(self) -> bool:
        return self.scales is not None

    @property
    def with_grouping(self) -> bool:
        return self.with_scaling and self.group_size > 0

    @property
    def device(self) -> torch.device:
        return self.data.device

    @torch.no_grad()
    def to(self, device: Union[str, torch.device]) -> "Weight":
        """Return a Weight with data and scales moved together; no preparation.

        The original object is unchanged. Like Tensor.to, tensor storage may be
        shared when the requested device is already the current device.
        This method changes only the device, never dtype or weight format.
        """
        if not isinstance(device, (str, torch.device)):
            raise TypeError("device must be a string or torch.device.")
        device = torch.device(device)
        if device.type not in ("cpu", "mps") or device.index not in (None, 0):
            raise ValueError(f"Weight.to supports CPU and MPS device 0; got {device}.")
        data = self.data.to(device)
        scales = self.scales.to(data.device) if self.scales is not None else None
        weight = type(self).__new__(type(self))
        for name, value in (
            ("data", data), ("scales", scales),
            ("logical_shape", self.logical_shape),
            ("logical_dtype", self.logical_dtype),
            ("layout", self.layout), ("group_size", self.group_size),
        ):
            object.__setattr__(weight, name, value)
        return weight

    @torch.no_grad()
    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Materialize logical [K,N] weights on the current device.

        INT4 is decoded, then scales are applied in FP32 before conversion to
        the requested FP16/FP32 dtype. Quantization loss cannot be recovered.
        Dense weights are returned in logical order as an independent tensor.
        """
        if not isinstance(dtype, torch.dtype) or dtype not in (torch.float16, torch.float32):
            raise TypeError("dtype must be torch.float16 or torch.float32.")
        if self.logical_dtype == "int4":
            K, N = self.logical_shape
            data = unpack_int4(self.data, N if self.layout == "nn" else K)
        else:
            data = self.data
        logical = data if self.layout == "nn" else data.T
        if self.logical_dtype != "int4":
            return logical.to(dtype).clone(memory_format=torch.contiguous_format)
        result = logical.float()
        if self.with_grouping:
            group_ids = torch.arange(self.logical_shape[0], device=self.device) // self.group_size
            result.mul_(self.scales[group_ids])
        elif self.with_scaling:
            result.mul_(self.scales)
        return result.to(dtype).contiguous()
