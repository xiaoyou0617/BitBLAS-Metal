from ..operations.config import MatmulConfig
from .base import MatmulKernel, MatmulScheduler


class TiledMatmulScheduler(MatmulScheduler):
    """Build a Metal kernel with FP32 accumulation and FP16 or FP32 output."""

    device_type = "mps"

    def __init__(self, config: MatmulConfig):
        super().__init__(config)
        self.threads = 64

    def build(self) -> MatmulKernel:
        # CPU fallback and the Torch baseline do not need to import TileLang.
        import tilelang
        import tilelang.language as T

        @tilelang.jit(
            target="metal",
            execution_backend="torch",
            out_idx=[2]
        )
        def generate_matmul(config: MatmulConfig, threads: int):
            weight_shape = (config.K, config.N) if config.layout == "nn" else (config.N, config.K)

            # Keep A/B/C device-buffer order for the current positional Metal adapter.
            @T.prim_func
            def tiled_matmul(
                A: T.Tensor((config.M, config.K), dtype=config.A_dtype),
                B: T.Tensor(weight_shape, dtype=config.W_dtype),
                C: T.Tensor((config.M, config.N), dtype=config.out_dtype)
            ):
                block_M = config.tile[0]
                block_N = config.tile[1]
                block_K = config.tile[2]
                M = config.M
                N = config.N
                K = config.K

                with T.Kernel(
                    T.ceildiv(M, block_M),
                    T.ceildiv(N, block_N),
                    threads=threads
                ) as (bx, by):
                    tile_A = T.alloc_shared((block_M, block_K), config.A_dtype, scope="shared")
                    tile_W = T.alloc_shared((block_K, block_N), config.W_dtype, scope="shared")
                    tile_Out = T.alloc_fragment((block_M, block_N), config.accum_dtype)

                    # Thread-private accumulators retain FP32 values across all K tiles.
                    for i, j in T.Parallel(block_M, block_N):
                        tile_Out[i, j] = 0.0

                    for bz in range(T.ceildiv(K, block_K)):
                        # filling the shared tiles
                        for i, k in T.Parallel(block_M, block_K):
                            x_idx = i + bx * block_M
                            k_idx = k + bz * block_K
                            if x_idx < M and k_idx < K:
                                tile_A[i, k] = A[x_idx, k_idx]
                            else:
                                tile_A[i, k] = 0.0

                        if config.layout == "nn":
                            for k, j in T.Parallel(block_K, block_N):
                                y_idx = j + by * block_N
                                k_idx = k + bz * block_K
                                if y_idx < N and k_idx < K:
                                    tile_W[k, j] = B[k_idx, y_idx]
                                else:
                                    tile_W[k, j] = 0.0
                        else:
                            # Read physical W[N,K] along K; shared storage stays [BK,BN].
                            for j, k in T.Parallel(block_N, block_K):
                                y_idx = j + by * block_N
                                k_idx = k + bz * block_K
                                if y_idx < N and k_idx < K:
                                    tile_W[k, j] = B[y_idx, k_idx]
                                else:
                                    tile_W[k, j] = 0.0

                        T.sync_threads()

                        acc = T.alloc_local((1,), config.accum_dtype)
                        for i, j in T.Parallel(block_M, block_N):
                            acc[0] = 0.0
                            for k in T.serial(block_K):
                                acc[0] += (
                                    T.cast(tile_A[i, k], config.accum_dtype)
                                    * T.cast(tile_W[k, j], config.accum_dtype)
                                )
                            tile_Out[i, j] += acc[0]
                        # All shared-memory reads must finish before the next load.
                        T.sync_threads()

                    # Convert exactly once, when the complete output is stored.
                    for i, j in T.Parallel(block_M, block_N):
                        x_idx = i + bx * block_M
                        y_idx = j + by * block_N
                        if x_idx < M and y_idx < N:
                            C[x_idx, y_idx] = T.cast(tile_Out[i, j], config.out_dtype)

            return tiled_matmul

        return generate_matmul(self.config, self.threads)
