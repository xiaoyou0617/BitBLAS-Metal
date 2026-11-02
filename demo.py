"""Run grouped A16W4 matmul on MPS and check it against reconstructed weights."""

import torch

from bitblas_metal import (
    Matmul,
    MatmulConfig,
    pack_int4,
    quantize_int4_with_grouping,
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
    qW, scales = quantize_int4_with_grouping(W, config.group_size)

    bias = torch.randn(config.N, dtype=torch.float32)

    # Build the CPU reference before moving the packed data to MPS.
    # Every K row uses the scale of its quantization group.
    group_ids = torch.arange(config.K) // config.group_size
    W_reconstructed = qW.float() * scales[group_ids]
    reference = A.float() @ W_reconstructed + bias

    # Quantization uses logical [K,N]; prepare the configured storage once.
    stored_codes = qW if config.layout == "nn" else qW.T.contiguous()
    W_packed = pack_int4(stored_codes).to("mps")
    A_mps = A.to("mps")
    scales_mps = scales.to("mps")
    op = Matmul(config, device="mps", backend="tilelang")
    C = op(A_mps, W_packed, scales=scales_mps, bias=bias.to("mps")).cpu()

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
