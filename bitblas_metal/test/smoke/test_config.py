import unittest

import torch

from bitblas_metal import MatmulConfig
from bitblas_metal.operations.config import SUPPORTED_DTYPE_COMBINATIONS


class TestMatmulConfig(unittest.TestCase):
    def test_layout_defaults_to_nt(self):
        self.assertEqual(MatmulConfig(2, 3, 4).layout, "nt")
        for layout in ("nn", "nt"):
            with self.subTest(layout=layout):
                self.assertEqual(MatmulConfig(2, 3, 4, layout=layout).layout, layout)

    def test_supported_combinations_and_scaling_policy(self):
        for A, W, accum, out, dequant in SUPPORTED_DTYPE_COMBINATIONS:
            with self.subTest(dtypes=(A, W, accum, out)):
                config = MatmulConfig(2, 3, 4, A_dtype=A, W_dtype=W,
                                      accum_dtype=accum, out_dtype=out)
                self.assertEqual(config.requires_dequant, dequant)
                if W != "int4":
                    with self.assertRaises(NotImplementedError):
                        MatmulConfig(2, 3, 4, A_dtype=A, W_dtype=W,
                                     accum_dtype=accum, out_dtype=out, with_scaling=True)

    def test_dtype_normalization_and_ignored_group(self):
        config = MatmulConfig(2, 3, 4, A_dtype=torch.float16, W_dtype=torch.float32)
        self.assertEqual(config.A_dtype, "float16")
        self.assertEqual(config.W_dtype, "float32")
        with self.assertWarns(UserWarning):
            config = MatmulConfig(2, 3, 4, group_size=128)
        self.assertEqual(config.group_size, 0)

    def test_invalid_configuration(self):
        cases = (
            ({"M": 0}, ValueError), ({"K": -1}, ValueError),
            ({"N": True}, TypeError), ({"M": 2.0}, TypeError),
            ({"tile": (8, 8)}, ValueError), ({"tile": [8, 8, 8]}, TypeError),
            ({"tile": (8, 0, 8)}, ValueError), ({"tile": (8, True, 8)}, TypeError),
            ({"group_size": -1}, ValueError), ({"group_size": True}, TypeError),
            ({"group_size": 1.5}, TypeError), ({"with_scaling": 1}, TypeError),
            ({"A_dtype": "int8"}, NotImplementedError), ({"A_dtype": 16}, TypeError),
            ({"layout": "tn"}, ValueError), ({"layout": "tt"}, ValueError),
            ({"layout": "NT"}, ValueError), ({"layout": ""}, ValueError),
            ({"layout": None}, TypeError), ({"layout": ["nn"]}, TypeError),
        )
        for changes, error in cases:
            with self.subTest(changes=changes):
                kwargs = dict(M=2, N=3, K=4)
                kwargs.update(changes)
                with self.assertRaises(error):
                    MatmulConfig(**kwargs)
