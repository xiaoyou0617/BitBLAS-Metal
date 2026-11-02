import warnings

from dataclasses import dataclass, field
from typing import Union

import torch


# Tuple order: (A_dtype, W_dtype, accum_dtype, out_dtype, requires_dequant).
SUPPORTED_DTYPE_COMBINATIONS = [
    ("float32", "float32", "float32", "float32", False),
    ("float16", "float16", "float32", "float32", False),
    ("float16", "float32", "float32", "float32", False),
    ("float32", "float16", "float32", "float32", False),
    ("float16", "int4", "float32", "float32", True),
    ("float32", "float32", "float32", "float16", False),
    ("float16", "float16", "float32", "float16", False),
    ("float16", "float32", "float32", "float16", False),
    ("float32", "float16", "float32", "float16", False),
    ("float16", "int4", "float32", "float16", True),
]


@dataclass(frozen=True)
class MatmulConfig:
    M: int
    N: int
    K: int
    A_dtype:     Union[str, torch.dtype] = "float32"
    W_dtype:     Union[str, torch.dtype] = "float32"
    accum_dtype: Union[str, torch.dtype] = "float32"
    out_dtype:   Union[str, torch.dtype] = "float32"
    # Derived from the supported combination, not from device capabilities.
    requires_dequant: bool = field(init=False)
    # A scheduler may select the tile in a later stage.
    tile: tuple = (8, 8, 8)
    with_scaling: bool = False
    # Quantization format along K; independent of the scheduler tile.
    group_size: int = 0
    # A[M,K] @ W[K,N] for nn; A[M,K] @ W[N,K].T for nt.
    layout: str = "nt"

    def __post_init__(self):
        """Validate dimensions, normalize dtypes, and derive the dequant flag."""
        if not isinstance(self.layout, str):
            raise TypeError(f"layout must be a string; got {type(self.layout).__name__}.")
        if self.layout not in ("nn", "nt"):
            raise ValueError(f"layout must be 'nn' or 'nt'; got {self.layout!r}.")
        if not isinstance(self.with_scaling, bool):
            raise TypeError(
                f"with_scaling must be a bool; got {self.with_scaling!r}."
            )
        if isinstance(self.group_size, bool) or not isinstance(self.group_size, int):
            raise TypeError(f"group_size must be an integer; got {self.group_size!r}.")
        if self.group_size < 0:
            raise ValueError(
                f"group_size must be nonnegative (0 disables grouping); got {self.group_size}."
            )
        if not isinstance(self.tile, tuple):
            raise TypeError(
                f"tile must be a tuple; got {type(self.tile).__name__}."
            )
        if len(self.tile) != 3:
            raise ValueError(
                f"tile must contain exactly three dimensions; got {len(self.tile)}."
            )

        dimensions = (
            ("M", self.M),
            ("N", self.N),
            ("K", self.K),
            ("block_M", self.tile[0]),
            ("block_N", self.tile[1]),
            ("block_K", self.tile[2]),
        )
        for name, value in dimensions:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer; got {value!r}.")
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}.")

        dtype_fields = ("A_dtype", "W_dtype", "accum_dtype", "out_dtype")
        for name in dtype_fields:
            value = getattr(self, name)
            if not isinstance(value, (str, torch.dtype)):
                raise TypeError(
                    f"{name} must be a string or torch.dtype; "
                    f"got {type(value).__name__}."
                )
            if isinstance(value, torch.dtype):
                value = str(value).removeprefix("torch.")
            object.__setattr__(self, name, value)

        dtype_combination = tuple(getattr(self, name) for name in dtype_fields)
        for combination in SUPPORTED_DTYPE_COMBINATIONS:
            if dtype_combination == combination[:4]:
                object.__setattr__(self, "requires_dequant", combination[4])
                break
        else:
            supported_dtypes = [entry[:4] for entry in SUPPORTED_DTYPE_COMBINATIONS]
            raise NotImplementedError(
                f"Unsupported dtype combination {dtype_combination!r} "
                "in (A_dtype, W_dtype, accum_dtype, out_dtype) order. "
                f"Supported combinations: {supported_dtypes!r}."
            )

        if self.with_scaling and self.W_dtype != "int4":
            raise NotImplementedError("with_scaling currently requires INT4 weights.")
        if not self.with_scaling and self.group_size > 0:
            warnings.warn(
                "group_size is ignored when with_scaling=False; using group_size=0.",
                UserWarning,
                stacklevel=3,
            )
            object.__setattr__(self, "group_size", 0)
