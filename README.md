# BitBLAS-Metal

Experimental low-precision matrix multiplication for **Apple Silicon**, built with
[PyTorch](https://pytorch.org/) and [TileLang](https://github.com/tile-ai/tilelang).

**Inspired by [BitBLAS](https://github.com/tile-ai/BitBLAS)**, this project explores
reusable operators, tiled Metal kernels, and low-bit weight formats on the Mac.
It is developed as a research and learning project under the guidance of
**[Lei Wang](https://github.com/LeiWang1999)**, a major contributor to BitBLAS.

The current focus is a clear implementation and numerical correctness. Systematic
performance evaluation and M5-specific optimization are future work. The project
has its own API and packed-weight format; compatibility with upstream BitBLAS
should be checked explicitly.

## Features

- Reusable operators: `MatmulConfig → scheduler selection → kernel build → calls`.
- Three execution paths: PyTorch CPU, PyTorch MPS, and TileLang-generated Metal.
- FP16/FP32 activation and weight inputs, including mixed FP16/FP32 inputs.
- `nt` (default) and `nn` weight layouts, including packed INT4 weights.
- A16W4: FP16 activations with signed INT4 weights packed into `uint8` storage.
- FP32 computation and accumulation with FP16 or FP32 output. Scaling is applied
  before the final output conversion.
- Optional FP32 bias, fused after scaling and before the final output conversion.
- Symmetric INT4 weight quantization with per-column or K-grouped FP32 scales.
- Prepared `Weight` objects keep data, scales, and format metadata together.
- Inference `Linear` binds a `Weight` and optional bias to a reusable operator.
- Tail handling, noncontiguous inputs, reusable output buffers, and input checks.
- Smoke and larger-matrix correctness regression suites.

## Current scope and tested environment

The Metal path has been validated on the following local setup:

| Component | Validated configuration |
| --- | --- |
| Hardware | Apple M5 |
| Operating system | macOS 26.5.2 |
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| TileLang | 0.1.9, with a working Metal backend |

Other Apple Silicon chips and dependency versions need separate validation.
The kernels currently use explicit tiled computation; an M5 TensorOps backend
has not been implemented. Shapes are fixed when an operator is constructed.
Autograd, zero points, batched matmul, and automatic tile tuning are future work.

## Installation

Prepare a Python environment with PyTorch and a **Metal-enabled TileLang build**.
For TileLang setup, consult the
[upstream project](https://github.com/tile-ai/tilelang); the table above records
the versions used for local validation. A Metal-capable build and the matching
Apple developer toolchain are required for the TileLang Metal path.

From the repository root, with that environment activated:

```bash
python -m pip install -e .
```

The distribution name is `bitblas-metal`; the Python import is `bitblas_metal`.
The dependency minimums in `pyproject.toml` are not a claim that every newer
version has been tested.

## Dense matmul

```python
import torch
from bitblas_metal import Matmul, MatmulConfig

config = MatmulConfig(
    M=32, N=64, K=128,
    A_dtype="float16",
    W_dtype="float16",
    accum_dtype="float32",
    out_dtype="float16",
    layout="nt",
    with_bias=True,
    tile=(8, 8, 8),
)
op = Matmul(config, device="mps", backend="tilelang")

A = torch.randn((config.M, config.K), dtype=torch.float16, device="mps")
W = torch.randn((config.N, config.K), dtype=torch.float16, device="mps")
bias = torch.randn(config.N, dtype=torch.float32, device="mps")
C = op(A, W, bias=bias)
assert C.shape == (config.M, config.N) and C.dtype == torch.float16

# Reuse both the compiled kernel and an output buffer.
out = torch.empty_like(C)
assert op(A, W, out=out, bias=bias) is out
```

`tile` is `(block_M, block_N, block_K)`. Operator construction selects and builds
an implementation; calling the operator performs the computation without tuning.

`layout="nt"` is the default: pass physical weights `[N,K]` to compute `A @ W.T`.
Use `layout="nn"` with physical weights `[K,N]` to compute `A @ W`.
Both paths load their weight format directly into a shared `[block_K,block_N]`
tile; the Metal path does not transpose the full weight matrix on each call.
Existing calls that pass `[K,N]` weights should explicitly set `layout="nn"`.

Set `with_bias=True` when constructing the operator, then provide `bias=` on each
call. Bias must be an FP32 `[N]` tensor on the selected device and broadcasts
over M. It is added once after all accumulation and scaling, before conversion
to the output dtype. Omit `bias` when `with_bias=False` (the default).

## Grouped A16W4 matmul

Quantization and packing are weight preparation steps. A `Weight` object prepares
them once and keeps the resulting data and matching scales together across calls.

```python
import torch
from bitblas_metal import Matmul, MatmulConfig, Weight

config = MatmulConfig(
    M=32, N=64, K=128,
    A_dtype="float16",
    W_dtype="int4",
    accum_dtype="float32",
    out_dtype="float16",
    with_scaling=True,
    group_size=32,
    layout="nt",
    with_bias=True,
    tile=(8, 8, 8),
)
# Weight always accepts logical [K,N], regardless of the matmul layout.
W_logical = torch.randn((config.K, config.N), dtype=torch.float16)
weight = Weight(W_logical, config).to("mps")
A = torch.randn((config.M, config.K), dtype=torch.float16, device="mps")
bias = torch.randn(config.N, dtype=torch.float32, device="mps")

op = Matmul(config, device="mps", backend="tilelang")
C = op(A, weight, bias=bias)
assert C.shape == (config.M, config.N) and C.dtype == torch.float16
```

### Prepared weights

`Weight(W_float, config)` accepts FP16/FP32 logical `[K,N]` inputs on CPU or MPS.
Dense weights are cast to `W_dtype` and stored in the configured layout. INT4
weights require `with_scaling=True`; the constructor selects column or grouped
quantization, arranges the layout, and packs the codes. It does not build a kernel.

The object exposes `data`, `scales`, `logical_shape`, `logical_dtype`, `layout`,
and `group_size`. `with_scaling` and `with_grouping` are derived properties.
Metadata and tensor attributes are read-only; tensor contents remain mutable.

```python
# Materialize reconstructed logical [K,N] weights, on weight.device.
# FP32 scales are applied before conversion to an optional FP16 result.
W_reconstructed = weight.dequantize()  # default torch.float32
W_half = weight.dequantize(dtype=torch.float16)

# Transfer data and scales together, without repeating preprocessing.
# Returns a new Weight; the original object stays on its original device.
weight_cpu = weight.to("cpu")

# Wrap existing physical data without quantizing, transposing, or packing it.
wrapped = Weight.from_prepared(weight.data, config, scales=weight.scales)
```

`from_prepared` validates shapes, dtypes, and matching devices, and makes tensors
contiguous. It does not inspect values or verify that the supplied scales were
computed from the supplied data. Contiguous inputs may share storage with the
object. `dequantize` reconstructs the represented weights, not the original
values lost during quantization, and returns an independent contiguous tensor.

The existing Tensor API remains available:

```python
C = op(A, weight.data, scales=weight.scales, bias=bias)
```

When passing a `Weight` object, omit `scales=`. `Matmul` checks its logical shape,
dtype, layout, scaling mode, and group size before using the tensors. M, tiles,
activation/output types, and bias are computation settings, so compatible
operators can reuse a weight despite differing in those settings. No transfer or
weight preparation happens implicitly inside `Matmul.__call__`.

For raw INT4 without scaling, pack integer codes in `[-8,7]` and use
`Weight.from_prepared(packed_data, config)` with `with_scaling=False`. Floating
inputs cannot be prepared as unscaled INT4 by dropping the quantizer's scales.

### Weight and scale conventions

| Value | Shape and type |
| --- | --- |
| Activations `A` | `[M, K]`, as specified by `A_dtype` |
| Logical weights / quantizer input | `[K, N]` |
| Dense weights, `nn` | `[K, N]` |
| Dense weights, `nt` (default) | `[N, K]` |
| Quantizer output `qW` | `[K, N]`, `torch.int8` |
| Packed INT4 weights, `nn` | `[K, ceil(N / 2)]`, `torch.uint8` |
| Packed INT4 weights, `nt` | `[N, ceil(K / 2)]`, `torch.uint8` |
| Per-column scales, `group_size=0` | `[N]`, `torch.float32` |
| Grouped scales, `group_size>0` | `[ceil(K / group_size), N]`, `torch.float32` |
| Bias, `with_bias=True` | `[N]`, `torch.float32`, on the selected device |
| Output | `[M, N]`, FP16 or FP32 as configured |

Packing runs along the physical last dimension: N for `nn`, K for `nt`.
The first value occupies the low nibble and the second
occupies the high nibble, using signed two's-complement codes in `[-8, 7]`.
The quantizers currently generate symmetric codes in `[-7, 7]`, using scale
`amax / 7` and nearest-even rounding. An all-zero column or group uses scale 1.

Use `quantize_int4(W)` for per-column scaling and
`quantize_int4_with_grouping(W, group_size)` for positive K-group sizes.
`group_size` belongs to the weight format and is independent of the execution
tile. Configure it in `MatmulConfig`; `Weight` supplies its matching scales, or
pass them explicitly when using the Tensor API.
The quantizers keep their logical `[K,N]` input and output convention. Pack `qW`
for `nn`, or pack `qW.T.contiguous()` for `nt`, during weight preparation.
Scales stay `[N]` or `[groups,N]` in both layouts; groups always follow logical K.
With scaling disabled, omit `scales`.

Move inputs and weights to the selected device before calling an operator.
For `Weight`, use `.to(device)` to move data and scales together.

## Linear

`Linear(config, weight, *, bias=None, backend="auto")` binds prepared weights and
optional bias, validates them before building, and creates one reusable `Matmul`.
Its device comes from `weight.device`; move weights and bias before construction.

```python
import torch
from bitblas_metal import Linear, MatmulConfig, Weight

config = MatmulConfig(
    M=32, N=64, K=128,
    A_dtype="float16", W_dtype="int4",
    out_dtype="float16", with_scaling=True, group_size=32,
    with_bias=True,
)
weight = Weight(torch.randn(config.K, config.N), config).to("mps")
bias = torch.zeros(config.N, dtype=torch.float32, device="mps")
layer = Linear(config, weight, bias=bias, backend="tilelang")

A = torch.randn(config.M, config.K, dtype=torch.float16, device="mps")
Y = layer(A)
out = torch.empty_like(Y)
assert layer(A * 2, out=out) is out
torch.testing.assert_close(out, Y * 2)  # bias is zero in this example
```

`with_bias=True` requires an FP32 `[N]` bias on the weight device; otherwise omit
bias. Calls accept a fixed two-dimensional `[M,K]` input and reuse the fused
kernel. The layer retains weights and bias by reference and can share a `Weight`
with other compatible layers. `layer.config` and `layer.device` expose the bound
configuration and device. Calls run without gradient tracking for inference.

## Backends

| Selection | Behavior |
| --- | --- |
| `device="mps", backend="tilelang"` | Build a dense or INT4 Metal kernel |
| `device="mps", backend="torch"` | PyTorch MPS reference implementation |
| `device="cpu", backend="torch"` | PyTorch CPU reference implementation |
| Defaults: `device=None, backend="auto"` | Use TileLang Metal when MPS is available; otherwise use CPU |

`auto` selects an available implementation without measuring performance.
The Torch INT4 reference paths unpack weights on each call, so their preparation
cost should be considered when interpreting future performance comparisons.

## Tests and demo

Run from the repository root, with the project's environment activated:

```bash
bash bitblas_metal/test/run.sh smoke
bash bitblas_metal/test/run.sh stress
bash bitblas_metal/test/run.sh all
bash bitblas_metal/test/run.sh all --device cpu
python demo.py
```

The tests use the standard-library `unittest` module. `smoke` covers small shapes,
dtype combinations, both layouts, quantization, packing, prepared-weight lifecycle,
API checks, and precision regressions.
`stress` checks larger matrices, including tails and K tiles crossing quantization
groups. These are correctness tests, not performance benchmarks.

MPS tests include both Torch and TileLang. The shell runner disables MPS CPU
fallback. Unavailable devices are reported as skipped; explicitly requesting
`--device mps` fails if MPS is unavailable.

Test status is colored automatically in terminals: green for success, red for
failures and errors, yellow for skips. Add `--color always` or `--color never`
to override this. The runner disables TileLang's disk cache to exercise current
kernel generation.

The demo checks kernel correctness against reconstructed weights and separately
reports quantization error against the original floating-point weights.

## Project layout

```text
bitblas_metal/
├── operations/    # Configuration, Matmul/Linear/Weight APIs, packing, and quantization
├── scheduler/     # Dispatch, Torch references, and TileLang Metal kernels
└── test/
    ├── smoke/    # Small correctness and interface checks
    └── stress/   # Larger-matrix correctness checks
demo.py
pyproject.toml
```

## Acknowledgements

Thanks to **[Lei Wang](https://github.com/LeiWang1999)** for guiding this research and learning project.
The operator organization and low-precision work are inspired by
**[BitBLAS, maintained under tile-ai](https://github.com/tile-ai/BitBLAS)**.
Thanks also to the TileLang and PyTorch contributors for the compiler and runtime
infrastructure used by this project.

## License

BitBLAS-Metal is licensed under the [MIT License](LICENSE).

BitBLAS has its own [MIT License](https://github.com/tile-ai/BitBLAS/blob/main/LICENSE).
Any reused upstream material retains its original copyright and license notices.
