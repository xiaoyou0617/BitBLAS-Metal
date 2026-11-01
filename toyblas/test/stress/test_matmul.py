import unittest

from toyblas.test._helpers import check_matmul, config_label, configs_for, on_device


class MatmulStressChecks:
    def test_large_aligned_matrices(self):
        for config in configs_for((256, 512, 1024), (8, 16, 16), group_size=128):
            with self.subTest(**config_label(config)):
                check_matmul(self, config, self.device, self.backend)

    def test_large_tail_and_cross_group_matrices(self):
        # All dimensions have tails; group_size=127 deliberately crosses K tiles.
        for config in configs_for((129, 257, 513), (8, 16, 16), group_size=127):
            with self.subTest(**config_label(config)):
                check_matmul(self, config, self.device, self.backend)


@on_device("cpu")
class TestCPUStress(MatmulStressChecks, unittest.TestCase):
    device, backend = "cpu", "torch"


@on_device("mps")
class TestTorchMPSStress(MatmulStressChecks, unittest.TestCase):
    device, backend = "mps", "torch"


@on_device("mps")
class TestTileLangStress(MatmulStressChecks, unittest.TestCase):
    device, backend = "mps", "tilelang"
