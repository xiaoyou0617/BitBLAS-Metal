from ..operations.config import MatmulConfig
from .base import MatmulKernel, MatmulScheduler


class WeightDequantMatmulScheduler(MatmulScheduler):
    """Build FP16 x packed signed INT4 with FP32 accumulation and optional scaling.

    Integer codes in [-8, 7] are decoded during tile loads. With scaling,
    C[m, n] = sum_k(A[m, k] * qW[k, n]) * scales[n].
    Positive group_size selects scales[ceil(K / group_size), N].
    Output is converted to FP16 or FP32 only after accumulation and scaling.
    Zero points are not part of this implementation.
    """

    device_type = "mps"

    def __init__(self, config: MatmulConfig):
        super().__init__(config)
        combination = (
            config.A_dtype, config.W_dtype, config.accum_dtype, config.out_dtype
        )
        if (combination[:3] != ("float16", "int4", "float32")
                or config.out_dtype not in ("float16", "float32")):
            raise NotImplementedError(
                "WeightDequantMatmulScheduler requires "
                "(A_dtype, W_dtype, accum_dtype, out_dtype) = "
                "('float16', 'int4', 'float32', 'float16' or 'float32')."
            )
        self.threads = 64

    def build(self) -> MatmulKernel:
        import tilelang
        import tilelang.language as T

        @tilelang.jit(
            target="metal",
            execution_backend="torch",
            out_idx=[3] if self.config.with_scaling else [2],
        )
        def generate_matmul(config: MatmulConfig, threads: int):
            block_M = config.tile[0]
            block_N = config.tile[1]
            block_K = config.tile[2]
            M = config.M
            N = config.N
            K = config.K
            group_size = config.group_size
            packed_weight_shape = (
                (K, (N + 1) // 2) if config.layout == "nn" else (N, (K + 1) // 2)
            )

            @T.macro
            def load_nn_weight_tile(B, tile_W, by, bz):
                # Physical B[K,ceil(N/2)]: each byte contains two N columns.
                for k, j in T.Parallel(block_K, T.ceildiv(block_N, 2)):
                    k_idx = bz * block_K + k
                    # Start at the byte containing this tile's first column.
                    # For odd block_N, a byte may straddle two output tiles.
                    packed_col = (by * block_N) // 2 + j
                    low_col = 2 * packed_col - by * block_N
                    high_col = low_col + 1
                    packed_byte = T.alloc_local((1,), "int32")
                    packed_byte[0] = 0
                    if k_idx < K and packed_col < T.ceildiv(N, 2):
                        packed_byte[0] = T.cast(B[k_idx, packed_col], "int32")

                    low_code = packed_byte[0] & 0xF
                    high_code = (packed_byte[0] >> 4) & 0xF
                    low_signed = T.if_then_else(low_code >= 8, low_code - 16, low_code)
                    high_signed = T.if_then_else(high_code >= 8, high_code - 16, high_code)

                    if 0 <= low_col and low_col < block_N:
                        if 2 * packed_col < N:
                            tile_W[k, low_col] = T.cast(low_signed, "float16")
                        else:
                            tile_W[k, low_col] = 0.0
                    if 0 <= high_col and high_col < block_N:
                        if 2 * packed_col + 1 < N:
                            tile_W[k, high_col] = T.cast(high_signed, "float16")
                        else:
                            tile_W[k, high_col] = 0.0

            @T.macro
            def load_nt_weight_tile(B, tile_W, by, bz):
                # Physical B[N,ceil(K/2)]: each byte contains two K positions.
                for j, k in T.Parallel(block_N, T.ceildiv(block_K, 2)):
                    y_idx = by * block_N + j
                    packed_k = (bz * block_K) // 2 + k
                    low_k = 2 * packed_k - bz * block_K
                    high_k = low_k + 1
                    packed_byte = T.alloc_local((1,), "int32")
                    packed_byte[0] = 0
                    if y_idx < N and packed_k < T.ceildiv(K, 2):
                        packed_byte[0] = T.cast(B[y_idx, packed_k], "int32")

                    low_code = packed_byte[0] & 0xF
                    high_code = (packed_byte[0] >> 4) & 0xF
                    low_signed = T.if_then_else(low_code >= 8, low_code - 16, low_code)
                    high_signed = T.if_then_else(high_code >= 8, high_code - 16, high_code)

                    # An odd block_K can split a byte between adjacent K tiles.
                    if 0 <= low_k and low_k < block_K:
                        if 2 * packed_k < K:
                            tile_W[low_k, j] = T.cast(low_signed, "float16")
                        else:
                            tile_W[low_k, j] = 0.0
                    if 0 <= high_k and high_k < block_K:
                        if 2 * packed_k + 1 < K:
                            tile_W[high_k, j] = T.cast(high_signed, "float16")
                        else:
                            tile_W[high_k, j] = 0.0

            @T.macro
            def load_tiles(A, B, tile_A, tile_W, bx, by, bz):
                for i, k in T.Parallel(block_M, block_K):
                    x_idx = bx * block_M + i
                    k_idx = bz * block_K + k
                    if x_idx < M and k_idx < K:
                        tile_A[i, k] = A[x_idx, k_idx]
                    else:
                        tile_A[i, k] = 0.0
                # Compile-time choice; shared tile_W stays [block_K,block_N].
                if config.layout == "nn":
                    load_nn_weight_tile(B, tile_W, by, bz)
                else:
                    load_nt_weight_tile(B, tile_W, by, bz)

            @T.macro
            def matmul_tile(A, B, tile_Out, bx, by):
                tile_A = T.alloc_shared(
                    (block_M, block_K), config.A_dtype, scope="shared"
                )
                # This tile holds decoded signed values, not packed bytes.
                tile_W = T.alloc_shared(
                    (block_K, block_N), "float16", scope="shared"
                )

                for i, j in T.Parallel(block_M, block_N):
                    tile_Out[i, j] = 0.0

                for bz in range(T.ceildiv(K, block_K)):
                    load_tiles(A, B, tile_A, tile_W, bx, by, bz)

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
                    # Finish all reads before the next K tile overwrites shared memory.
                    T.sync_threads()

            if config.with_scaling and group_size > 0:
                num_groups = (K + group_size - 1) // group_size

                @T.prim_func
                def grouped_matmul_int4(
                    A: T.Tensor((M, K), dtype=config.A_dtype),
                    B: T.Tensor(packed_weight_shape, dtype="uint8"),
                    C: T.Tensor((num_groups, N), dtype="float32"),
                    D: T.Tensor((M, N), dtype=config.out_dtype),
                ):
                    with T.Kernel(
                        T.ceildiv(M, block_M),
                        T.ceildiv(N, block_N),
                        threads=threads,
                    ) as (bx, by):
                        tile_A = T.alloc_shared(
                            (block_M, block_K), config.A_dtype, scope="shared"
                        )
                        tile_W = T.alloc_shared(
                            (block_K, block_N), "float16", scope="shared"
                        )
                        tile_Out = T.alloc_fragment(
                            (block_M, block_N), config.accum_dtype
                        )

                        for i, j in T.Parallel(block_M, block_N):
                            tile_Out[i, j] = 0.0

                        for bz in range(T.ceildiv(K, block_K)):
                            load_tiles(A, B, tile_A, tile_W, bx, by, bz)
                            T.sync_threads()
                            acc = T.alloc_local((1,), config.accum_dtype)
                            for i, j in T.Parallel(block_M, block_N):
                                acc[0] = 0.0
                                x_idx = bx * block_M + i
                                y_idx = by * block_N + j
                                if group_size % block_K == 0:
                                    # Every valid k in this tile belongs to the same group.
                                    for k in T.serial(block_K):
                                        acc[0] += (
                                            T.cast(tile_A[i, k], config.accum_dtype)
                                            * T.cast(tile_W[k, j], config.accum_dtype)
                                        )
                                    if x_idx < M and y_idx < N:
                                        tile_Out[i, j] += acc[0] * C[(bz * block_K) // group_size, y_idx]
                                else:
                                    # A tile may cross groups. Select the scale for each k.
                                    for k in T.serial(block_K):
                                        k_idx = bz * block_K + k
                                        if k_idx < K and y_idx < N:
                                            acc[0] += (
                                                T.cast(tile_A[i, k], config.accum_dtype)
                                                * T.cast(tile_W[k, j], config.accum_dtype)
                                                * C[k_idx // group_size, y_idx]
                                            )
                                    if x_idx < M and y_idx < N:
                                        tile_Out[i, j] += acc[0]
                            T.sync_threads()

                        for i, j in T.Parallel(block_M, block_N):
                            x_idx = bx * block_M + i
                            y_idx = by * block_N + j
                            if x_idx < M and y_idx < N:
                                D[x_idx, y_idx] = T.cast(tile_Out[i, j], config.out_dtype)

                return grouped_matmul_int4

            if config.with_scaling:
                # A/B/C/D names keep positional Metal bindings in argument order:
                # C is the scale vector here; D is the output.
                @T.prim_func
                def scaled_matmul_int4(
                    A: T.Tensor((M, K), dtype=config.A_dtype),
                    B: T.Tensor(packed_weight_shape, dtype="uint8"),
                    C: T.Tensor((N,), dtype="float32"),
                    D: T.Tensor((M, N), dtype=config.out_dtype),
                ):
                    with T.Kernel(
                        T.ceildiv(M, block_M),
                        T.ceildiv(N, block_N),
                        threads=threads,
                    ) as (bx, by):
                        tile_Out = T.alloc_fragment(
                            (block_M, block_N), config.accum_dtype
                        )
                        matmul_tile(A, B, tile_Out, bx, by)
                        # Multiply in FP32, then convert once when writing the output.
                        for i, j in T.Parallel(block_M, block_N):
                            x_idx = bx * block_M + i
                            y_idx = by * block_N + j
                            if x_idx < M and y_idx < N:
                                D[x_idx, y_idx] = T.cast(
                                    tile_Out[i, j] * C[y_idx], config.out_dtype
                                )

                return scaled_matmul_int4

            @T.prim_func
            def tiled_matmul_int4(
                A: T.Tensor((M, K), dtype=config.A_dtype),
                B: T.Tensor(packed_weight_shape, dtype="uint8"),
                C: T.Tensor((M, N), dtype=config.out_dtype),
            ):
                with T.Kernel(
                    T.ceildiv(M, block_M),
                    T.ceildiv(N, block_N),
                    threads=threads,
                ) as (bx, by):
                    tile_Out = T.alloc_fragment(
                        (block_M, block_N), config.accum_dtype
                    )
                    matmul_tile(A, B, tile_Out, bx, by)
                    for i, j in T.Parallel(block_M, block_N):
                        x_idx = bx * block_M + i
                        y_idx = by * block_N + j
                        if x_idx < M and y_idx < N:
                            C[x_idx, y_idx] = T.cast(tile_Out[i, j], config.out_dtype)

            return tiled_matmul_int4

        return generate_matmul(self.config, self.threads)
