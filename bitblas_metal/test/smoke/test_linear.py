"""Bound-input validation, inference semantics, and numerical Linear checks."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from bitblas_metal import Linear, MatmulConfig, Weight, pack_int4
from bitblas_metal.test._helpers import noncontiguous_copy, on_device


class LinearChecks:
    def test_dense_known_answers_and_output_reuse(self):
        logical = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        for layout in ("nn", "nt"):
            for dtype, bias_flag in (("float32", True), ("float16", False)):
                with self.subTest(layout=layout, dtype=dtype, bias=bias_flag):
                    config = MatmulConfig(2, 3, 3, A_dtype=dtype, W_dtype=dtype,
                                          out_dtype=dtype, with_bias=bias_flag,
                                          layout=layout, tile=(3, 3, 2))
                    weight = Weight(logical, config).to(self.device)
                    bias = (noncontiguous_copy(torch.tensor([1, -2, 0.5], device=self.device))
                            if bias_flag else None)
                    layer = Linear(config, weight, bias=bias, backend=self.backend)
                    self.assertIs(layer.weight, weight)
                    self.assertIs(layer.bias, bias)
                    self.assertIs(layer.config, config)
                    self.assertEqual(layer.device.type, self.device)
                    A = torch.tensor([[1, 2, 3], [4, 5, 6]], device=self.device,
                                     dtype=getattr(torch, dtype))
                    expected = torch.tensor([[30, 36, 42], [66, 81, 96]], dtype=torch.float32)
                    if bias_flag:
                        expected += torch.tensor([1, -2, 0.5])
                    expected = expected.to(getattr(torch, dtype))
                    torch.testing.assert_close(layer(A).cpu(), expected, rtol=0, atol=0)
                    out = torch.full((2, 3), float("nan"), dtype=getattr(torch, dtype),
                                     device=self.device)
                    self.assertIs(layer(A, out=out), out)
                    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                    # A changes, while the prepared weights and compiled callable are reused.
                    kernel = layer.matmul.calculas
                    expected_next = (expected.float() - bias.cpu() if bias_flag else expected.float()) * 2
                    if bias_flag:
                        expected_next += bias.cpu()
                    torch.testing.assert_close(layer(A * 2, out).cpu(),
                                               expected_next.to(out.dtype), rtol=0, atol=0)
                    self.assertIs(layer.matmul.calculas, kernel)

    def test_int4_scaling_and_bias_known_answers(self):
        # Unit codes: raw=4, column scale=2 gives 8, g=2 gives 22, g=3 gives 13.
        for layout in ("nn", "nt"):
            for out_dtype in ("float32", "float16"):
                for scaling, group, product in ((False, 0, 4), (True, 0, 8),
                                                (True, 2, 22), (True, 3, 13)):
                    with self.subTest(layout=layout, out=out_dtype, group=group, scaling=scaling):
                        config = MatmulConfig(2, 3, 4, A_dtype="float16", W_dtype="int4",
                                              out_dtype=out_dtype, with_scaling=scaling,
                                              group_size=group, with_bias=True,
                                              layout=layout, tile=(3, 3, 3))
                        codes = torch.ones((4, 3) if layout == "nn" else (3, 4), dtype=torch.int8)
                        scales = None
                        if scaling:
                            scales = (torch.tensor([[1, 1, 1], [10, 10, 10]], dtype=torch.float32)
                                      if group else torch.full((3,), 2.0))
                        weight = Weight.from_prepared(pack_int4(codes), config, scales=scales).to(self.device)
                        bias = torch.tensor([5, -1, 0.25], device=self.device)
                        layer = Linear(config, weight, bias=bias, backend=self.backend)
                        A = torch.tensor([[1] * 4, [2] * 4], dtype=torch.float16, device=self.device)
                        expected = (torch.tensor([[product], [2 * product]])
                                    + torch.tensor([5, -1, 0.25])).to(getattr(torch, out_dtype))
                        torch.testing.assert_close(layer(A).cpu(), expected, rtol=0, atol=0)

    def test_share_weight_across_layers_and_inference_without_gradients(self):
        config = MatmulConfig(2, 3, 4, with_bias=True, tile=(3, 3, 3))
        weight = Weight(torch.ones((4, 3)), config).to(self.device)
        bias = torch.ones(3, device=self.device, requires_grad=True)
        layer = Linear(config, weight, bias=bias, backend=self.backend)
        other = Linear(replace(config, M=1, out_dtype="float16", tile=(2, 2, 2)),
                       weight, bias=bias, backend=self.backend)
        self.assertIs(layer.weight, other.weight)
        A = torch.ones((2, 4), device=self.device, requires_grad=True)
        result = layer(A)
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad_fn)
        torch.testing.assert_close(result.cpu(), torch.full((2, 3), 5.0), rtol=0, atol=0)
        torch.testing.assert_close(other(A[:1]).cpu(), torch.full((1, 3), 5.0).half(),
                                   rtol=0, atol=0)
        # Binding retains tensors by reference; in-place value updates are visible.
        with torch.no_grad():
            bias.add_(1)
        torch.testing.assert_close(layer(A).cpu(), torch.full((2, 3), 6.0), rtol=0, atol=0)

    def test_invalid_bound_inputs_fail_before_kernel_build(self):
        config = MatmulConfig(1, 5, 5, A_dtype="float16", W_dtype="int4",
                              with_scaling=True, group_size=3, with_bias=True)
        weight = Weight(torch.ones((5, 5)), config).to(self.device)
        bias = torch.ones(5, device=self.device)
        cases = (
            (None, weight, bias, TypeError, "config"),
            (config, weight.data, bias, TypeError, "weight must be a Weight"),
            (replace(config, K=6), weight, bias, ValueError, "logical_shape"),
            (replace(config, layout="nn"), weight, bias, ValueError, "layout"),
            (replace(config, group_size=4), weight, bias, ValueError, "group_size"),
            (replace(config, with_scaling=False, group_size=0), weight, bias, ValueError, "with_scaling"),
            (config, weight, None, ValueError, "bias is required"),
            (config, weight, 1.0, TypeError, "bias must be a torch.Tensor"),
            (config, weight, bias.view(1, 5), ValueError, "bias must have shape"),
            (config, weight, bias.half(), TypeError, "bias must have dtype"),
            (replace(config, with_bias=False), weight, bias, ValueError, "bias must be omitted"),
        )
        for index, (cfg, bound_weight, bound_bias, error, message) in enumerate(cases):
            with self.subTest(case=index):
                with patch("bitblas_metal.operations.linear.Matmul") as build:
                    with self.assertRaisesRegex(error, message):
                        Linear(cfg, bound_weight, bias=bound_bias, backend=self.backend)
                    build.assert_not_called()
        if self.device == "mps":
            with patch("bitblas_metal.operations.linear.Matmul") as build:
                with self.assertRaisesRegex(ValueError, "bias must be on"):
                    Linear(config, weight, bias=bias.cpu(), backend=self.backend)
                build.assert_not_called()

    def test_runtime_inputs_and_hidden_bias_alias(self):
        config = MatmulConfig(1, 3, 3, with_bias=True, tile=(2, 2, 2))
        weight = Weight(torch.ones((3, 3)), config).to(self.device)
        bias = torch.ones(3, device=self.device)
        layer = Linear(config, weight, bias=bias, backend=self.backend)
        A = torch.ones((1, 3), device=self.device)
        with self.assertRaisesRegex(ValueError, "A must have shape"):
            layer(A.T)
        with self.assertRaisesRegex(TypeError, "A must have dtype"):
            layer(A.half())
        with self.assertRaisesRegex(ValueError, "share storage with bias"):
            layer(A, out=bias.view(1, 3))
        if self.device == "mps":
            with self.assertRaisesRegex(ValueError, "A must be on"):
                layer(A.cpu())


@on_device("cpu")
class TestCPULinear(LinearChecks, unittest.TestCase):
    device, backend = "cpu", "torch"


@on_device("mps")
class TestTorchMPSLinear(LinearChecks, unittest.TestCase):
    device, backend = "mps", "torch"


@on_device("mps")
class TestTileLangLinear(LinearChecks, unittest.TestCase):
    device, backend = "mps", "tilelang"
