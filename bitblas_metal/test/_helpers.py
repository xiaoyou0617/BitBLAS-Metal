"""Shared inputs and an independent CPU FP64 reference; no kernel implementation here."""

import os
import unittest

import torch

from bitblas_metal import Matmul, MatmulConfig
from bitblas_metal.operations.config import SUPPORTED_DTYPE_COMBINATIONS
from bitblas_metal.operations.pack import pack_int4
from bitblas_metal.operations.quantize import quantize_int4, quantize_int4_with_grouping


def on_device(device):
    """Class decorator: explicitly report excluded or unavailable devices as skips."""
    selected = os.environ.get("BITBLAS_METAL_TEST_DEVICE", "all")
    if selected not in ("all", device):
        return unittest.skip(f"device={selected} excludes {device}")
    if device == "mps" and not torch.backends.mps.is_available():
        return unittest.skip("MPS is unavailable")
    return lambda cls: cls


def configs_for(shape, tile, group_size):
    """Dense types: raw only. INT4: raw, column-scaled, and group-scaled."""
    M, N, K = shape
    for layout in ("nn", "nt"):
        for A_dtype, W_dtype, accum_dtype, out_dtype, _ in SUPPORTED_DTYPE_COMBINATIONS:
            modes = ((False, 0),)
            if W_dtype == "int4":
                modes += ((True, 0), (True, group_size))
            for with_scaling, group in modes:
                for with_bias in (False, True):
                    yield MatmulConfig(
                        M, N, K, A_dtype=A_dtype, W_dtype=W_dtype,
                        accum_dtype=accum_dtype, out_dtype=out_dtype, tile=tile,
                        with_scaling=with_scaling, group_size=group, layout=layout,
                        with_bias=with_bias,
                    )


def config_label(config):
    return {
        "dtypes": (config.A_dtype, config.W_dtype, config.accum_dtype, config.out_dtype),
        "shape": (config.M, config.N, config.K),
        "tile": config.tile,
        "layout": config.layout,
        "with_bias": config.with_bias,
        "scaling": config.with_scaling,
        "group_size": config.group_size,
    }


def noncontiguous_copy(tensor):
    """Keep values unchanged but put a gap between adjacent columns."""
    storage = torch.empty(
        (*tensor.shape[:-1], tensor.shape[-1] * 2), dtype=tensor.dtype, device=tensor.device
    )
    view = storage[..., ::2]
    view.copy_(tensor)
    return view


def check_matmul(test, config, device, backend, *, noncontiguous=False, reuse=False):
    generator = torch.Generator().manual_seed(2026)
    A_cpu = (torch.randn(config.M, config.K, generator=generator) * 0.25).to(
        getattr(torch, config.A_dtype)
    )
    scales_cpu = None
    if config.W_dtype == "int4":
        if config.with_scaling:
            W_float = torch.randn(config.K, config.N, generator=generator).to(torch.float16)
            W_float[:, -1] = 0  # An all-zero column must stay zero after dequantization.
            if config.group_size:
                qW, scales_cpu = quantize_int4_with_grouping(W_float, config.group_size)
            else:
                qW, scales_cpu = quantize_int4(W_float)
        else:
            qW = torch.randint(-8, 8, (config.K, config.N), generator=generator, dtype=torch.int8)
        stored_codes = qW if config.layout == "nn" else qW.T.contiguous()
        W_cpu = pack_int4(stored_codes)
        W_reference = qW.double()  # Deliberately do not use unpack_int4 as the oracle.
        if scales_cpu is not None:
            if config.group_size:
                group_ids = torch.arange(config.K) // config.group_size
                W_reference *= scales_cpu[group_ids].double()
            else:
                W_reference *= scales_cpu.double()
    else:
        W_cpu = (torch.randn(config.K, config.N, generator=generator) * 0.25).to(
            getattr(torch, config.W_dtype)
        )
        W_reference = W_cpu.double()
        if config.layout == "nt":
            W_cpu = W_cpu.T.contiguous()

    product_reference = A_cpu.double() @ W_reference
    bias_cpu = torch.randn(config.N, generator=generator) * 0.25 if config.with_bias else None
    reference = product_reference + bias_cpu.double() if bias_cpu is not None else product_reference
    reference = reference.to(getattr(torch, config.out_dtype))
    A, W = A_cpu.to(device), W_cpu.to(device)
    scales = scales_cpu.to(device) if scales_cpu is not None else None
    bias = bias_cpu.to(device) if bias_cpu is not None else None
    if noncontiguous:
        A, W = noncontiguous_copy(A), noncontiguous_copy(W)
        if scales is not None:
            scales = noncontiguous_copy(scales)
        if bias is not None:
            bias = noncontiguous_copy(bias)
    op = Matmul(config, device=device, backend=backend)

    def assert_result(result, expected):
        test.assertEqual(tuple(result.shape), (config.M, config.N))
        test.assertEqual(result.dtype, getattr(torch, config.out_dtype))
        test.assertEqual(result.device.type, device)
        # FP32 and FP64 accumulation may fall on opposite sides of an FP16
        # rounding boundary. Allow about one FP16 step in the random cases;
        # dedicated known-answer tests check early rounding with zero tolerance.
        rtol = 1e-3 if config.out_dtype == "float16" else 1e-4
        torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=1e-4)

    assert_result(op(A, W, scales=scales, bias=bias), reference)
    if reuse:
        # A poisoned output detects missing initialization and accidental accumulation.
        out = torch.full((config.M, config.N), float("nan"), device=device,
                         dtype=getattr(torch, config.out_dtype))
        test.assertIs(op(A, W, out=out, scales=scales, bias=bias), out)
        assert_result(out, reference)
        # Reuse the same built kernel with new activation and scale values.
        new_scales = scales * 2 if scales is not None else None
        new_bias = -bias if bias is not None else None
        expected = -product_reference * (2 if scales is not None else 1)
        if bias_cpu is not None:
            expected = expected - bias_cpu.double()
        expected = expected.to(getattr(torch, config.out_dtype))
        test.assertIs(op(-A, W, out=out, scales=new_scales, bias=new_bias), out)
        assert_result(out, expected)
