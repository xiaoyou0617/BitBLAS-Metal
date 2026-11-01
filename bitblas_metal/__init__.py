"""BitBLAS-Metal: low-precision matrix multiplication for Apple Silicon."""

from .operations import (
    Matmul,
    MatmulConfig,
    generate_matmul,
    pack_int4,
    quantize_int4,
    quantize_int4_with_grouping,
    unpack_int4,
)

__all__ = [
    "MatmulConfig",
    "Matmul",
    "generate_matmul",
    "pack_int4",
    "unpack_int4",
    "quantize_int4",
    "quantize_int4_with_grouping",
]
