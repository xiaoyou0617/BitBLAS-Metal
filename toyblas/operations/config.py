from dataclasses import dataclass
from typing import Union

import torch


# Tuple order: (A_dtype, W_dtype, accum_dtype, out_dtype).
SUPPORTED_DTYPE_COMBINATIONS = [
    ("float32", "float32", "float32", "float32"),
    ("float16", "float16", "float32", "float32"),
    ("float16", "float32", "float32", "float32"),
    ("float32", "float16", "float32", "float32"),
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
    # A scheduler may select the tile in a later stage.
    tile: tuple = (8, 8, 8)

    def __post_init__(self):
        """Validate dimensions and normalize dtype fields to strings."""
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
        if dtype_combination not in SUPPORTED_DTYPE_COMBINATIONS:
            raise NotImplementedError(
                f"Unsupported dtype combination {dtype_combination!r} "
                "in (A_dtype, W_dtype, accum_dtype, out_dtype) order. "
                f"Supported combinations: {SUPPORTED_DTYPE_COMBINATIONS!r}."
            )
