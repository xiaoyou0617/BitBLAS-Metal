import unittest

import torch

from bitblas_metal.operations.pack import pack_int4, unpack_int4
from bitblas_metal.operations.quantize import quantize_int4, quantize_int4_with_grouping
from bitblas_metal.test._helpers import noncontiguous_copy, on_device


class QuantizationChecks:
    def test_all_packed_bytes(self):
        # Independent byte-format fixture: each byte has two signed 4-bit codes.
        def signed(code):
            return code if code < 8 else code - 16

        expected_codes = torch.tensor(
            [[signed(byte % 16), signed(byte // 16)] for byte in range(256)], dtype=torch.int8
        )
        expected_bytes = torch.arange(256).to(torch.uint8).reshape(256, 1)
        codes = expected_codes.to(self.device)
        packed = pack_int4(codes)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.device.type, self.device)
        torch.testing.assert_close(packed.cpu(), expected_bytes, rtol=0, atol=0)
        # Test unpack independently of pack, so two matching bugs cannot cancel.
        decoded = unpack_int4(expected_bytes.to(self.device), original_cols=2)
        torch.testing.assert_close(decoded.cpu(), expected_codes, rtol=0, atol=0)

    def test_odd_and_empty_packing(self):
        codes = torch.tensor([[-8, -1, 0, 7, 2]], dtype=torch.int8, device=self.device)
        packed = pack_int4(noncontiguous_copy(codes))
        torch.testing.assert_close(packed.cpu(), torch.tensor([[248, 112, 2]], dtype=torch.uint8),
                                   rtol=0, atol=0)
        # The unused high nibble must not become a logical extra column.
        packed[:, -1] = 162
        torch.testing.assert_close(unpack_int4(packed, 5), codes, rtol=0, atol=0)
        for shape in ((0, 3), (2, 0)):
            with self.subTest(shape=shape):
                empty = torch.empty(shape, dtype=torch.int8, device=self.device)
                decoded = unpack_int4(pack_int4(empty), shape[1])
                self.assertEqual(tuple(decoded.shape), shape)

    def test_nearest_even_and_zero_column(self):
        values = [-7, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 7]
        expected = torch.tensor([[-7, -2, -2, 0, 0, 2, 2, 7], [0] * 8], dtype=torch.int8).T
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                W = torch.tensor([values, [0] * 8], dtype=dtype, device=self.device).T
                qW, scales = quantize_int4(W)
                self.assertEqual(qW.dtype, torch.int8)
                self.assertEqual(scales.dtype, torch.float32)
                self.assertEqual(qW.device.type, self.device)
                self.assertEqual(scales.device.type, self.device)
                torch.testing.assert_close(qW.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(scales.cpu(), torch.ones(2), rtol=0, atol=0)
                grouped_qW, grouped_scales = quantize_int4_with_grouping(W, 8)
                torch.testing.assert_close(grouped_qW.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(grouped_scales.cpu(), torch.ones(1, 2), rtol=0, atol=0)

    def test_grouped_quantization_reference(self):
        generator = torch.Generator().manual_seed(19)
        for dtype in (torch.float16, torch.float32):
            W_cpu = torch.randn(7, 5, generator=generator).to(dtype)
            W_cpu[:, -1] = 0
            for group_size in (1, 3, 128):
                with self.subTest(dtype=dtype, group_size=group_size):
                    # A small scalar group loop is the oracle; no padding or grouped reshape.
                    expected_codes, expected_scales = [], []
                    for start in range(0, W_cpu.shape[0], group_size):
                        group = W_cpu[start:start + group_size].float()
                        scale = group.abs().amax(dim=0) / 7
                        scale[scale == 0] = 1
                        expected_scales.append(scale)
                        expected_codes.append(torch.round(group / scale).clamp(-7, 7).to(torch.int8))
                    qW, scales = quantize_int4_with_grouping(
                        noncontiguous_copy(W_cpu.to(self.device)), group_size
                    )
                    torch.testing.assert_close(qW.cpu(), torch.cat(expected_codes), rtol=0, atol=0)
                    torch.testing.assert_close(scales.cpu(), torch.stack(expected_scales),
                                               rtol=1e-6, atol=0)

    def test_invalid_preparation_inputs(self):
        for values in ([8], [-9]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    pack_int4(torch.tensor([values], dtype=torch.int8, device=self.device))
        with self.assertRaises(TypeError):
            pack_int4(torch.zeros((2, 3), device=self.device))
        with self.assertRaises(ValueError):
            unpack_int4(torch.zeros((2, 2), dtype=torch.uint8, device=self.device), 5)
        for values in (float("nan"), float("inf")):
            for quantizer in (quantize_int4, lambda W: quantize_int4_with_grouping(W, 3)):
                with self.subTest(value=values, quantizer=quantizer):
                    with self.assertRaises(ValueError):
                        quantizer(torch.tensor([[values]], device=self.device))
        for group_size in (0, -1, True, 1.5):
            with self.subTest(group_size=group_size):
                with self.assertRaises((TypeError, ValueError)):
                    quantize_int4_with_grouping(torch.ones((2, 3), device=self.device), group_size)


@on_device("cpu")
class TestCPUQuantization(QuantizationChecks, unittest.TestCase):
    device = "cpu"


@on_device("mps")
class TestMPSQuantization(QuantizationChecks, unittest.TestCase):
    device = "mps"
