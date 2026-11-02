import unittest

import torch

from bitblas_metal import Matmul, MatmulConfig
from bitblas_metal.operations.pack import pack_int4
from bitblas_metal.test._helpers import check_matmul, config_label, configs_for, noncontiguous_copy, on_device


class MatmulSmokeChecks:
    def test_bias_after_scaling_known_answers(self):
        # Product per first-row output: raw=4, column-scaled=8,
        # grouped (g=2)=22, grouped (g=3)=13. Bias must be added once, unscaled.
        for layout in ("nn", "nt"):
            for out_dtype in ("float32", "float16"):
                for scaling, group, product in ((False, 0, 4), (True, 0, 8),
                                                (True, 2, 22), (True, 3, 13)):
                    with self.subTest(layout=layout, out_dtype=out_dtype, group=group,
                                      scaling=scaling):
                        config = MatmulConfig(2, 3, 4, A_dtype="float16", W_dtype="int4",
                                              out_dtype=out_dtype, layout=layout,
                                              with_scaling=scaling, group_size=group,
                                              with_bias=True, tile=(3, 3, 3))
                        A = torch.tensor([[1] * 4, [2] * 4], dtype=torch.float16,
                                         device=self.device)
                        codes = torch.ones((4, 3) if layout == "nn" else (3, 4), dtype=torch.int8)
                        W = pack_int4(codes).to(self.device)
                        bias = torch.tensor([5, -1, 0.25], dtype=torch.float32, device=self.device)
                        scales = None
                        if scaling:
                            scales = torch.tensor([[1] * 3, [10] * 3], dtype=torch.float32,
                                                  device=self.device) if group else torch.full(
                                                      (3,), 2.0, device=self.device)
                        expected = (torch.tensor([[product], [2 * product]])
                                    + torch.tensor([5, -1, 0.25])).to(getattr(torch, out_dtype))
                        op = Matmul(config, device=self.device, backend=self.backend)
                        torch.testing.assert_close(op(A, W, scales=scales, bias=bias).cpu(),
                                                   expected, rtol=0, atol=0)

    def test_bias_before_fp16_output_conversion(self):
        modes = (("float16", False, 0), ("int4", False, 0),
                 ("int4", True, 0), ("int4", True, 1))
        for W_dtype, scaling, group in modes:
            config = MatmulConfig(1, 1, 2, A_dtype="float16", W_dtype=W_dtype,
                                  out_dtype="float16", with_scaling=scaling,
                                  group_size=group, with_bias=True, tile=(2, 3, 1))
            op = Matmul(config, device=self.device, backend=self.backend)
            W = (pack_int4(torch.ones((1, 2), dtype=torch.int8)).to(self.device)
                 if W_dtype == "int4" else torch.ones((1, 2), dtype=torch.float16, device=self.device))
            scales = torch.ones((2, 1) if group else (1,), device=self.device) if scaling else None
            for values, offset, answer in (([2048, 1], 1, 2050),
                                           ([40000, 40000], -40000, 40000)):
                with self.subTest(W_dtype=W_dtype, scaling=scaling, group=group, values=values):
                    A = torch.tensor([values], dtype=torch.float16, device=self.device)
                    bias = torch.tensor([offset], dtype=torch.float32, device=self.device)
                    expected = torch.tensor([[answer]], dtype=torch.float16)
                    torch.testing.assert_close(op(A, W, scales=scales, bias=bias).cpu(),
                                               expected, rtol=0, atol=0)

    def test_bias_contract(self):
        config = MatmulConfig(1, 3, 2, with_bias=True)
        op = Matmul(config, device=self.device, backend=self.backend)
        A = torch.ones((1, 2), device=self.device)
        W = torch.ones((3, 2), device=self.device)
        bias = torch.ones(3, device=self.device)
        for bad_bias, error in ((None, ValueError), (1.0, TypeError),
                                (torch.ones((3, 1), device=self.device), ValueError),
                                (bias[:2], ValueError), (bias.half(), TypeError)):
            with self.subTest(bias_type=type(bad_bias).__name__, error=error.__name__):
                with self.assertRaises(error):
                    op(A, W, bias=bad_bias)
        with self.assertRaises(ValueError):
            op(A, W, out=bias.view(1, 3), bias=bias)
        if self.device == "mps":
            with self.assertRaises(ValueError):
                op(A, W, bias=bias.cpu())
        no_bias = Matmul(MatmulConfig(1, 3, 2), device=self.device, backend=self.backend)
        with self.assertRaises(ValueError):
            no_bias(A, W, bias=bias)

    def test_supported_dtype_combinations(self):
        # Every dimension has a tail; odd block_N exercises bytes crossing N tiles.
        for config in configs_for((5, 7, 11), (4, 3, 4), group_size=8):
            with self.subTest(**config_label(config)):
                check_matmul(self, config, self.device, self.backend,
                             noncontiguous=True, reuse=True)

    def test_grouping_boundaries(self):
        cases = (
            ((1, 1, 1), (8, 8, 8), 128),  # A group and a tile both larger than K.
            ((2, 3, 7), (3, 1, 4), 1),    # One scale per K row; odd single-column tiles.
            ((3, 5, 7), (3, 3, 4), 3),    # A K tile straddles two quantization groups.
            ((4, 7, 11), (3, 5, 3), 5),   # Odd BK splits packed K bytes in nt layout.
        )
        for layout in ("nn", "nt"):
            for out_dtype in ("float32", "float16"):
                for (M, N, K), tile, group in cases:
                    config = MatmulConfig(M, N, K, A_dtype="float16", W_dtype="int4",
                                          out_dtype=out_dtype, with_scaling=True,
                                          group_size=group, tile=tile, layout=layout)
                    with self.subTest(**config_label(config)):
                        check_matmul(self, config, self.device, self.backend)

    def test_layout_known_answers_and_shape_checks(self):
        A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=self.device)
        W_math = torch.arange(1, 16, dtype=torch.float32, device=self.device).reshape(3, 5)
        expected = torch.tensor([[46, 52, 58, 64, 70], [100, 115, 130, 145, 160]],
                                dtype=torch.float32)
        for layout in ("nn", "nt"):
            with self.subTest(layout=layout):
                config = MatmulConfig(2, 5, 3, layout=layout, tile=(3, 3, 2))
                W = W_math if layout == "nn" else W_math.T.contiguous()
                op = Matmul(config, device=self.device, backend=self.backend)
                torch.testing.assert_close(op(A, W).cpu(), expected, rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    op(A, W.T)  # Non-square shapes distinguish nn from nt.

    def test_nt_int4_byte_order_and_padding(self):
        # Independent bytes for physical [N,ceil(K/2)], with poisoned padding.
        packed = torch.tensor([[184, 30, 244], [201, 47, 245], [218, 48, 246]],
                              dtype=torch.uint8, device=self.device)
        expected = torch.tensor([[-8, -7, -6], [-5, -4, -3], [-2, -1, 0],
                                 [1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        config = MatmulConfig(5, 3, 5, A_dtype="float16", W_dtype="int4",
                              layout="nt", tile=(3, 3, 3))
        A = torch.eye(5, dtype=torch.float16, device=self.device)
        op = Matmul(config, device=self.device, backend=self.backend)
        torch.testing.assert_close(op(A, packed).cpu(), expected, rtol=0, atol=0)

    def test_fp16_keeps_full_fp32_sum(self):
        # FP16 rounding after each K tile would incorrectly produce 2048.
        for config in configs_for((1, 1, 3), (1, 1, 1), group_size=1):
            if config.out_dtype != "float16":
                continue
            with self.subTest(**config_label(config)):
                A = torch.tensor([[2048.0, 1.0, 1.0]], device=self.device,
                                 dtype=getattr(torch, config.A_dtype))
                if config.W_dtype == "int4":
                    codes = torch.ones((3, 1), dtype=torch.int8)
                    if config.layout == "nt":
                        codes = codes.T.contiguous()
                    W = pack_int4(codes).to(self.device)
                else:
                    shape = (3, 1) if config.layout == "nn" else (1, 3)
                    W = torch.ones(shape, device=self.device,
                                   dtype=getattr(torch, config.W_dtype))
                scales = None
                if config.with_scaling:
                    shape = (3, 1) if config.group_size else (1,)
                    scales = torch.ones(shape, device=self.device)
                op = Matmul(config, device=self.device, backend=self.backend)
                expected = torch.tensor([[2050.0]], dtype=torch.float16)
                bias = torch.zeros(1, device=self.device) if config.with_bias else None
                torch.testing.assert_close(op(A, W, scales=scales, bias=bias).cpu(), expected,
                                           rtol=0, atol=0)

    def test_fp16_scaling_precedes_output_conversion(self):
        # The unscaled sum 80000 exceeds FP16 range; the scaled result is 625.
        # This catches casting before column scaling and before grouped scaling.
        for layout in ("nn", "nt"):
            for group_size in (0, 1, 2):
                with self.subTest(layout=layout, group_size=group_size):
                    config = MatmulConfig(1, 1, 2, A_dtype="float16", W_dtype="int4",
                                          out_dtype="float16", with_scaling=True,
                                          group_size=group_size, tile=(2, 3, 2), layout=layout)
                    A = torch.full((1, 2), 40000.0, dtype=torch.float16, device=self.device)
                    codes = torch.ones((2, 1) if layout == "nn" else (1, 2), dtype=torch.int8)
                    W = pack_int4(codes).to(self.device)
                    shape = ((2 + group_size - 1) // group_size, 1) if group_size else (1,)
                    scales = torch.full(shape, 1 / 128, device=self.device)
                    op = Matmul(config, device=self.device, backend=self.backend)
                    expected = torch.tensor([[625.0]], dtype=torch.float16)
                    torch.testing.assert_close(op(A, W, scales=scales).cpu(), expected,
                                               rtol=0, atol=0)

    def test_grouped_known_answer(self):
        # 1 + 1 + 10 + 10 = 22; applying a single scale to all K would be wrong.
        config = MatmulConfig(1, 1, 4, A_dtype="float16", W_dtype="int4",
                              with_scaling=True, group_size=2, tile=(2, 3, 3))
        A = torch.ones((1, 4), dtype=torch.float16, device=self.device)
        W = pack_int4(torch.ones((1, 4), dtype=torch.int8)).to(self.device)
        scales = torch.tensor([[1.0], [10.0]], device=self.device)
        op = Matmul(config, device=self.device, backend=self.backend)
        torch.testing.assert_close(op(A, W, scales=scales).cpu(), torch.tensor([[22.0]]),
                                   rtol=0, atol=0)

    def test_input_and_output_contract(self):
        config = MatmulConfig(2, 3, 3)
        op = Matmul(config, device=self.device, backend=self.backend)
        A = torch.ones((2, 3), device=self.device)
        W = torch.ones((3, 3), device=self.device)
        cases = (
            (TypeError, lambda: op(None, W)),
            (ValueError, lambda: op(A[:1], W)),
            (TypeError, lambda: op(A, W.half())),
            (ValueError, lambda: op(A, W, out=A)),  # out aliases an input.
            (ValueError, lambda: op(A, W, out=noncontiguous_copy(A))),
            (TypeError, lambda: op(A, W, out=A.half())),
            (ValueError, lambda: op(A, W, scales=torch.ones(3, device=self.device))),
        )
        for index, (expected_error, call) in enumerate(cases):
            with self.subTest(case=index, expected_error=expected_error.__name__):
                with self.assertRaises(expected_error):
                    call()
        if self.device == "mps":
            with self.assertRaises(ValueError):
                op(A.cpu(), W)

    def test_scale_contract(self):
        config = MatmulConfig(2, 3, 7, A_dtype="float16", W_dtype="int4",
                              with_scaling=True, group_size=3)
        op = Matmul(config, device=self.device, backend=self.backend)
        A = torch.ones((2, 7), dtype=torch.float16, device=self.device)
        W = pack_int4(torch.ones((3, 7), dtype=torch.int8)).to(self.device)
        scales = torch.ones((3, 3), device=self.device)
        for bad_scales, error in ((None, ValueError), (1.0, TypeError),
                                  (scales[0], ValueError), (scales.half(), TypeError)):
            with self.subTest(scales_type=type(bad_scales).__name__, error=error.__name__):
                with self.assertRaises(error):
                    op(A, W, scales=bad_scales)
        if self.device == "mps":
            with self.assertRaises(ValueError):
                op(A, W, scales=scales.cpu())


@on_device("cpu")
class TestCPUMatmul(MatmulSmokeChecks, unittest.TestCase):
    device, backend = "cpu", "torch"


@on_device("mps")
class TestTorchMPSMatmul(MatmulSmokeChecks, unittest.TestCase):
    device, backend = "mps", "torch"


@on_device("mps")
class TestTileLangMatmul(MatmulSmokeChecks, unittest.TestCase):
    device, backend = "mps", "tilelang"
