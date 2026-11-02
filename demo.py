"""Run grouped A16W4 matmul on MPS and check it against reconstructed weights."""

import torch

from bitblas_metal import (
    Matmul,
    MatmulConfig,
    Weight,
)


def main():
    torch.manual_seed(0)
    A = torch.randn((32, 1024), dtype=torch.float16)
    W = torch.randn((1024, 1024), dtype=torch.float16)

    config = MatmulConfig(
        M=A.shape[0],
        N=W.shape[1],
        K=A.shape[1],
        A_dtype="float16",
        W_dtype="int4",
        accum_dtype="float32",
        out_dtype="float32",
        with_scaling=True,
        group_size=128,
        layout="nt",
        with_bias=True,
    )
    weight = Weight(W, config)

    bias = torch.randn(config.N, dtype=torch.float32)

    # Build the CPU reference before moving the packed data to MPS.
    W_reconstructed = weight.dequantize()
    reference = A.float() @ W_reconstructed + bias

    # Move the packed data and matching scales together; no repeated preparation.
    weight_mps = weight.to("mps")
    A_mps = A.to("mps")
    op = Matmul(config, device="mps", backend="tilelang")
    C = op(A_mps, weight_mps, bias=bias.to("mps")).cpu()

    # Kernel correctness and quantization error are separate questions.
    torch.testing.assert_close(C, reference, rtol=1e-4, atol=1e-4)
    kernel_max_error = (C - reference).abs().max().item()
    original_reference = A.float() @ W.float() + bias
    quantization_rmse = (reference - original_reference).square().mean().sqrt().item()

    print(f"Kernel correctness: PASS; shape={tuple(C.shape)}, dtype={C.dtype}")
    print(f"Kernel max absolute error: {kernel_max_error:.6g}")
    print(f"Quantization output RMSE: {quantization_rmse:.6g}")


if __name__ == "__main__":
    main()
