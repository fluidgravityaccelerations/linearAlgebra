"""Custom PyCUDA solver for batched singular values of small real square matrices.

The implementation follows the pipeline used by the original notebooks:

    matrix -> bidiagonal form -> symmetric tridiagonal B^T B
           -> Sturm sequence + bisection -> singular values

The code intentionally supports square matrices only.  The original notebooks
claimed support for Nrows <= Ncols, but the rectangular path left part of the
tridiagonal data uninitialized.  Restricting the custom solver to square
matrices makes the implemented algorithm and its memory layout unambiguous.

PyCUDA is imported lazily so that this module can be imported on machines
without CUDA when only the repository documentation is being inspected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


def _dtype_info(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return dtype, "float", np.float32(1e-6)
    if dtype == np.dtype(np.float64):
        return dtype, "double", np.float64(1e-12)
    raise TypeError("The custom PyCUDA solver supports only float32 and float64.")


def _require_pycuda():
    try:
        import pycuda.driver as drv
        from pycuda.compiler import SourceModule
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime
        raise RuntimeError(
            "PyCUDA is required for the custom solver. Install PyCUDA and run "
            "this code on a CUDA-capable system."
        ) from exc
    return drv, SourceModule


def make_rearrange_kernel(dtype_str: str, batch_size: int, n: int) -> str:
    return f"""
    #define T {dtype_str}
    #define BATCH {int(batch_size)}
    #define N {int(n)}

    extern "C" {{
    __global__ void rearrangeKernel(const T * __restrict__ inputMatrices,
                                    T * __restrict__ outputMatrices) {{
        const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid >= BATCH) return;
        #pragma unroll
        for (unsigned int i = 0; i < N; ++i) {{
            #pragma unroll
            for (unsigned int j = 0; j < N; ++j) {{
                outputMatrices[tid + j * BATCH + i * (BATCH * N)] =
                    inputMatrices[tid * (N * N) + j * N + i];
            }}
        }}
    }}
    }}
    """


def make_bidiagonalize_kernel(dtype_str: str, batch_size: int, n: int) -> str:
    return f"""
    #define T {dtype_str}
    #define BATCH {int(batch_size)}
    #define N {int(n)}

    __device__ __forceinline__ void computeLeftHouseholderVectors(
        const T * __restrict__ inputMatrices,
        T * __restrict__ v,
        T &beta,
        const unsigned int tid,
        const unsigned int k)
    {{
        #pragma unroll
        for (unsigned int i = 0; i < N; ++i) v[i] = (T)0;

        T norm2 = (T)0;
        #pragma unroll
        for (unsigned int i = k + 1; i < N; ++i) {{
            const unsigned int idx = i * (BATCH * N) + k * BATCH + tid;
            const T value = inputMatrices[idx];
            v[i] = value;
            norm2 += value * value;
        }}

        const unsigned int idx0 = k * (BATCH * N) + k * BATCH + tid;
        const T x0 = inputMatrices[idx0];

        if (norm2 == (T)0) {{
            beta = (T)0;
            return;
        }}

        const T mu = sqrt(x0 * x0 + norm2);
        const T v0 = (x0 <= (T)0) ? (x0 - mu) : (-norm2 / (x0 + mu));
        const T inv_v0 = (T)1 / v0;
        beta = (T)2 * (v0 * v0) / (norm2 + v0 * v0);
        v[k] = (T)1;
        #pragma unroll
        for (unsigned int i = k + 1; i < N; ++i) v[i] *= inv_v0;
    }}

    __device__ __forceinline__ void computeRightHouseholderVectors(
        const T * __restrict__ inputVector,
        T * __restrict__ v,
        T &beta,
        const unsigned int k)
    {{
        #pragma unroll
        for (unsigned int i = 0; i < N; ++i) v[i] = (T)0;

        // There is no right reflector at or beyond the penultimate column.
        if (k + 1 >= N) {{
            beta = (T)0;
            return;
        }}

        T norm2 = (T)0;
        #pragma unroll
        for (unsigned int i = k + 2; i < N; ++i) {{
            const T value = inputVector[i];
            v[i] = value;
            norm2 += value * value;
        }}

        const T x0 = inputVector[k + 1];
        if (norm2 == (T)0) {{
            beta = (T)0;
            return;
        }}

        const T mu = sqrt(x0 * x0 + norm2);
        const T v0 = (x0 <= (T)0) ? (x0 - mu) : (-norm2 / (x0 + mu));
        const T inv_v0 = (T)1 / v0;
        beta = (T)2 * (v0 * v0) / (norm2 + v0 * v0);
        v[k + 1] = (T)1;
        #pragma unroll
        for (unsigned int i = k + 2; i < N; ++i) v[i] *= inv_v0;
    }}

    extern "C" {{
    __global__ void bidiagonalizeKernel(T *inputMatrices) {{
        const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid >= BATCH) return;

        T left[N];
        T right[N];
        T rightInput[N];
        T w[N];
        T xi[N];
        T z[N];
        T betaLeft = (T)0;
        T betaRight = (T)0;

        // For an N x N matrix, the last iteration needs only the left update.
        #pragma unroll
        for (unsigned int k = 0; k < N; ++k) {{
            computeLeftHouseholderVectors(inputMatrices, left, betaLeft, tid, k);

            if (k < N - 2) {{
                #pragma unroll
                for (unsigned int j = 0; j < N; ++j) {{
                    T sum = (T)0;
                    #pragma unroll
                    for (unsigned int i = 0; i < N; ++i) {{
                        const unsigned int idx = i * (BATCH * N) + j * BATCH + tid;
                        sum += -betaLeft * left[i] * inputMatrices[idx];
                    }}
                    const unsigned int idxDiag = k * (BATCH * N) + j * BATCH + tid;
                    rightInput[j] = sum + inputMatrices[idxDiag];
                    xi[j] = -sum;
                }}

                computeRightHouseholderVectors(rightInput, right, betaRight, k);

                #pragma unroll
                for (unsigned int i = 0; i < N; ++i) {{
                    T sum = (T)0;
                    #pragma unroll
                    for (unsigned int j = 0; j < N; ++j) {{
                        const unsigned int idx = i * (BATCH * N) + j * BATCH + tid;
                        sum += betaRight * right[j] * inputMatrices[idx];
                    }}
                    w[i] = sum;
                }}

                T dot = (T)0;
                #pragma unroll
                for (unsigned int i = 0; i < N; ++i) dot += xi[i] * right[i];
                #pragma unroll
                for (unsigned int i = 0; i < N; ++i)
                    z[i] = xi[i] - betaRight * dot * right[i];

                #pragma unroll
                for (unsigned int r = 0; r < N; ++r) {{
                    #pragma unroll
                    for (unsigned int c = 0; c < N; ++c) {{
                        const unsigned int idx = r * (BATCH * N) + c * BATCH + tid;
                        inputMatrices[idx] += -left[r] * z[c] - w[r] * right[c];
                    }}
                }}
            }} else {{
                #pragma unroll
                for (unsigned int j = 0; j < N; ++j) {{
                    T sum = (T)0;
                    #pragma unroll
                    for (unsigned int i = 0; i < N; ++i) {{
                        const unsigned int idx = i * (BATCH * N) + j * BATCH + tid;
                        sum += left[i] * inputMatrices[idx];
                    }}
                    xi[j] = sum;
                }}
                #pragma unroll
                for (unsigned int r = 0; r < N; ++r) {{
                    #pragma unroll
                    for (unsigned int c = 0; c < N; ++c) {{
                        const unsigned int idx = r * (BATCH * N) + c * BATCH + tid;
                        inputMatrices[idx] += -betaLeft * xi[c] * left[r];
                    }}
                }}
            }}
        }}
    }}
    }}
    """


def make_extract_diagonals_kernel(dtype_str: str, batch_size: int, n: int) -> str:
    return f"""
    #define T {dtype_str}
    #define BATCH {int(batch_size)}
    #define N {int(n)}

    extern "C" {{
    __global__ void extractDiagonalsKernel(
        const T * __restrict__ inputMatrices,
        T * __restrict__ d,
        T * __restrict__ e)
    {{
        const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid >= BATCH) return;
        #pragma unroll
        for (unsigned int i = 0; i < N - 1; ++i) {{
            d[i * BATCH + tid] = inputMatrices[(i * N + i) * BATCH + tid];
            e[i * BATCH + tid] = inputMatrices[(i * N + i + 1) * BATCH + tid];
        }}
        d[(N - 1) * BATCH + tid] =
            inputMatrices[((N - 1) * N + (N - 1)) * BATCH + tid];
    }}
    }}
    """


def make_tridiagonal_kernel(dtype_str: str, batch_size: int, n: int) -> str:
    return f"""
    #define T {dtype_str}
    #define BATCH {int(batch_size)}
    #define N {int(n)}

    extern "C" {{
    __global__ void tridiagFromBidiagKernel(
        const T * __restrict__ d,
        const T * __restrict__ e,
        T * __restrict__ diag,
        T * __restrict__ offdiag)
    {{
        const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= BATCH) return;

        const T d0 = d[tid];
        diag[tid] = d0 * d0;
        #pragma unroll
        for (unsigned int i = 1; i < N; ++i) {{
            const T di = d[i * BATCH + tid];
            const T ei_1 = e[(i - 1) * BATCH + tid];
            diag[i * BATCH + tid] = di * di + ei_1 * ei_1;
        }}
        #pragma unroll
        for (unsigned int i = 0; i < N - 1; ++i) {{
            offdiag[i * BATCH + tid] = d[i * BATCH + tid] * e[i * BATCH + tid];
        }}
    }}
    }}
    """


def make_pivot_kernel(dtype_str: str, batch_size: int, n: int) -> str:
    eps_str = "FLT_EPSILON" if dtype_str == "float" else "DBL_EPSILON"
    min_str = "FLT_MIN" if dtype_str == "float" else "DBL_MIN"
    return f"""
    #define T {dtype_str}
    #define BATCH {int(batch_size)}
    #define N {int(n)}
    #include <cfloat>
    #include <cmath>
    #define MACHINE_EPS {eps_str}
    #define MIN_NORMAL {min_str}

    __device__ __forceinline__ float absT(float x) {{ return fabsf(x); }}
    __device__ __forceinline__ double absT(double x) {{ return fabs(x); }}
    __device__ __forceinline__ float maxT(float x, float y) {{ return fmaxf(x, y); }}
    __device__ __forceinline__ double maxT(double x, double y) {{ return fmax(x, y); }}

    extern "C" {{
    __global__ void pivotsComputationKernel(
        const T * __restrict__ diag,
        const T * __restrict__ offdiag,
        T * __restrict__ piv)
    {{
        const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= BATCH) return;
        T scale = (T)1;
        #pragma unroll
        for (unsigned int i = 0; i < N; ++i)
            scale = maxT(scale, absT(diag[i * BATCH + tid]));
        #pragma unroll
        for (unsigned int i = 0; i < N - 1; ++i)
            scale = maxT(scale, absT(offdiag[i * BATCH + tid]));

        T p = scale * (T)16 * (T)MACHINE_EPS;
        const T floorValue = (T)MIN_NORMAL * (T)16;
        if (p < floorValue) p = floorValue;
        piv[tid] = p;
    }}
    }}
    """


def make_interval_kernel(dtype_str: str, batch_size: int, n: int) -> str:
    return f"""
    #define T {dtype_str}
    #define BATCH {int(batch_size)}
    #define N {int(n)}
    #include <cmath>

    __device__ __forceinline__ float absT(float x) {{ return fabsf(x); }}
    __device__ __forceinline__ double absT(double x) {{ return fabs(x); }}

    extern "C" {{
    __global__ void computeInitialIntervalsKernel(
        const T * __restrict__ diag,
        const T * __restrict__ offdiag,
        T * __restrict__ lower,
        T * __restrict__ upper)
    {{
        const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= BATCH) return;

        T lo = diag[tid] - absT(offdiag[tid]);
        T hi = diag[tid] + absT(offdiag[tid]);

        #pragma unroll
        for (unsigned int i = 1; i < N - 1; ++i) {{
            const T radius = absT(offdiag[(i - 1) * BATCH + tid])
                           + absT(offdiag[i * BATCH + tid]);
            const T di = diag[i * BATCH + tid];
            const T li = di - radius;
            const T ui = di + radius;
            if (li < lo) lo = li;
            if (ui > hi) hi = ui;
        }}

        const T lastRadius = absT(offdiag[(N - 2) * BATCH + tid]);
        const T lastDiag = diag[(N - 1) * BATCH + tid];
        const T lastLo = lastDiag - lastRadius;
        const T lastHi = lastDiag + lastRadius;
        if (lastLo < lo) lo = lastLo;
        if (lastHi > hi) hi = lastHi;

        // B^T B is positive semidefinite.  Keeping a slightly negative lower
        // bound is harmless for Sturm bisection and protects against roundoff.
        lower[tid] = lo;
        upper[tid] = hi;
    }}
    }}
    """


def make_roots_kernel(dtype_str: str, batch_size: int, n: int) -> str:
    return f"""
    #define T {dtype_str}
    #define BATCH {int(batch_size)}
    #define N {int(n)}
    #include <cmath>

    __device__ __forceinline__ float absT(float x) {{ return fabsf(x); }}
    __device__ __forceinline__ double absT(double x) {{ return fabs(x); }}
    __device__ __forceinline__ float maxT(float x, float y) {{ return fmaxf(x, y); }}
    __device__ __forceinline__ double maxT(double x, double y) {{ return fmax(x, y); }}

    extern "C" {{
    __global__ void sturmBisectionKernel(
        const T * __restrict__ diag,
        const T * __restrict__ offdiag,
        const T * __restrict__ pivots,
        const T * __restrict__ lower,
        const T * __restrict__ upper,
        T * __restrict__ singularVals,
        const T tol)
    {{
        const unsigned int matrixId = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int rootId = blockIdx.y * blockDim.y + threadIdx.y;
        if (matrixId >= BATCH || rootId >= N) return;

        T a = lower[matrixId];
        T b = upper[matrixId];
        const T pivot = pivots[matrixId];
        T c = (a + b) * (T)0.5;

        for (unsigned int iter = 0; iter < 256; ++iter) {{
            c = (a + b) * (T)0.5;
            unsigned int numChanges = 0;
            T q = diag[matrixId] - c;
            if (absT(q) <= pivot) q = -pivot;
            if (q < (T)0) ++numChanges;

            #pragma unroll
            for (unsigned int i = 1; i < N; ++i) {{
                const T bi = offdiag[(i - 1) * BATCH + matrixId];
                q = diag[i * BATCH + matrixId] - c - (bi * bi) / q;
                if (absT(q) <= pivot) q = -pivot;
                if (q < (T)0) ++numChanges;
            }}

            // rootId=0 corresponds to the largest eigenvalue, matching the
            // descending order returned by NumPy/PyTorch/CuPy SVD routines.
            if (numChanges > (N - (rootId + 1))) b = c;
            else                                  a = c;

            const T width = absT(b - a);
            const T scale = maxT((T)1, absT(a) + absT(b));
            if (width <= tol * scale) break;
        }}

        T eigenvalue = c;
        const T roundoffScale = maxT((T)1, absT(upper[matrixId]));
        if (eigenvalue < (T)0 && absT(eigenvalue) <= tol * roundoffScale)
            eigenvalue = (T)0;
        // A substantially negative value indicates a failure of the numerical
        // pipeline.  Leaving it negative makes sqrt return NaN instead of
        // silently turning a large error into a plausible singular value.
        singularVals[rootId + matrixId * N] = sqrt(eigenvalue);
    }}
    }}
    """


@dataclass
class TimingStats:
    """Summary of repeated wall-clock timings in seconds."""

    median: float
    minimum: float
    maximum: float
    mean: float
    std: float
    samples: tuple[float, ...]


def _timing_stats(samples):
    arr = np.asarray(samples, dtype=np.float64)
    return TimingStats(
        median=float(np.median(arr)),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        samples=tuple(float(x) for x in arr),
    )


@dataclass
class BenchmarkResult:
    singular_values: np.ndarray
    timing: TimingStats


class CustomBatchedSingularValues:
    """Compiled single-GPU solver for a fixed batch size and square matrix size.

    The solver computes singular values only. Compilation and device-buffer
    allocation occur in ``__init__`` and are therefore outside all benchmark
    timings.
    """

    def __init__(self, batch_size: int, matrix_size: int = 4, dtype=np.float64):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if matrix_size < 2:
            raise ValueError("matrix_size must be at least 2")
        self.batch_size = int(batch_size)
        self.n = int(matrix_size)
        self.dtype, self.dtype_str, self.tol = _dtype_info(dtype)
        self.drv, SourceModule = _require_pycuda()

        options = ["--std=c++11"]
        self.modules = {
            "rearrange": SourceModule(make_rearrange_kernel(self.dtype_str, self.batch_size, self.n), options=options, no_extern_c=True),
            "bidiag": SourceModule(make_bidiagonalize_kernel(self.dtype_str, self.batch_size, self.n), options=options, no_extern_c=True),
            "extract": SourceModule(make_extract_diagonals_kernel(self.dtype_str, self.batch_size, self.n), options=options, no_extern_c=True),
            "tridiag": SourceModule(make_tridiagonal_kernel(self.dtype_str, self.batch_size, self.n), options=options, no_extern_c=True),
            "pivot": SourceModule(make_pivot_kernel(self.dtype_str, self.batch_size, self.n), options=options, no_extern_c=True),
            "interval": SourceModule(make_interval_kernel(self.dtype_str, self.batch_size, self.n), options=options, no_extern_c=True),
            "roots": SourceModule(make_roots_kernel(self.dtype_str, self.batch_size, self.n), options=options, no_extern_c=True),
        }
        self.functions = {
            "rearrange": self.modules["rearrange"].get_function("rearrangeKernel"),
            "bidiag": self.modules["bidiag"].get_function("bidiagonalizeKernel"),
            "extract": self.modules["extract"].get_function("extractDiagonalsKernel"),
            "tridiag": self.modules["tridiag"].get_function("tridiagFromBidiagKernel"),
            "pivot": self.modules["pivot"].get_function("pivotsComputationKernel"),
            "interval": self.modules["interval"].get_function("computeInitialIntervalsKernel"),
            "roots": self.modules["roots"].get_function("sturmBisectionKernel"),
        }
        self._allocate()

    def _allocate(self):
        drv = self.drv
        B, N, item = self.batch_size, self.n, self.dtype.itemsize
        self.d_input = drv.mem_alloc(B * N * N * item)
        self.d_work = drv.mem_alloc(B * N * N * item)
        self.d_d = drv.mem_alloc(B * N * item)
        self.d_e = drv.mem_alloc(B * (N - 1) * item)
        self.d_diag = drv.mem_alloc(B * N * item)
        self.d_offdiag = drv.mem_alloc(B * (N - 1) * item)
        self.d_pivot = drv.mem_alloc(B * item)
        self.d_lower = drv.mem_alloc(B * item)
        self.d_upper = drv.mem_alloc(B * item)
        self.d_sv = drv.mem_alloc(B * N * item)
        self._buffers = [
            self.d_input, self.d_work, self.d_d, self.d_e, self.d_diag,
            self.d_offdiag, self.d_pivot, self.d_lower, self.d_upper, self.d_sv,
        ]

    def _prepare_host(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=self.dtype)
        expected = (self.batch_size, self.n, self.n)
        if batch.shape != expected:
            raise ValueError(f"Expected batch shape {expected}, got {batch.shape}.")
        if not np.all(np.isfinite(batch)):
            raise ValueError("Input batch contains NaN or Inf values.")
        # Flatten each matrix in column-major order, then store all matrices
        # contiguously. The first CUDA kernel converts this into a structure-of-
        # arrays layout used by the subsequent kernels.
        return np.ascontiguousarray(batch.transpose(0, 2, 1)).reshape(-1)

    def upload(self, batch: np.ndarray):
        host = self._prepare_host(batch)
        self.drv.memcpy_htod(self.d_input, host)
        return host

    def compute(self, stream=None):
        B, N = self.batch_size, self.n
        block = 128
        grid = ((B + block - 1) // block, 1, 1)
        kw = {} if stream is None else {"stream": stream}
        self.functions["rearrange"](self.d_input, self.d_work, block=(block, 1, 1), grid=grid, **kw)
        self.functions["bidiag"](self.d_work, block=(block, 1, 1), grid=grid, **kw)
        self.functions["extract"](self.d_work, self.d_d, self.d_e, block=(block, 1, 1), grid=grid, **kw)
        self.functions["tridiag"](self.d_d, self.d_e, self.d_diag, self.d_offdiag, block=(block, 1, 1), grid=grid, **kw)
        self.functions["pivot"](self.d_diag, self.d_offdiag, self.d_pivot, block=(block, 1, 1), grid=grid, **kw)
        self.functions["interval"](self.d_diag, self.d_offdiag, self.d_lower, self.d_upper, block=(block, 1, 1), grid=grid, **kw)

        bx, by = 32, 4
        gx = (B + bx - 1) // bx
        gy = (N + by - 1) // by
        self.functions["roots"](
            self.d_diag, self.d_offdiag, self.d_pivot, self.d_lower, self.d_upper,
            self.d_sv, self.tol, block=(bx, by, 1), grid=(gx, gy, 1), **kw
        )

    def synchronize(self):
        self.drv.Context.synchronize()

    def download(self) -> np.ndarray:
        out = np.empty((self.batch_size, self.n), dtype=self.dtype)
        self.drv.memcpy_dtoh(out, self.d_sv)
        return out

    def solve(self, batch: np.ndarray) -> np.ndarray:
        """Compute singular values without timing."""
        self.upload(batch)
        self.compute()
        self.synchronize()
        return self.download()

    def benchmark_device(self, batch: np.ndarray, repeats: int = 7) -> BenchmarkResult:
        """Benchmark compute only, with the input already resident on the GPU.

        Wall-clock timing plus an explicit synchronization is used deliberately;
        the same convention is used for every backend in the notebooks.
        """
        if repeats < 1:
            raise ValueError("repeats must be at least one")
        self.upload(batch)
        self.compute()  # warm-up
        self.synchronize()

        samples = []
        for _ in range(int(repeats)):
            self.synchronize()
            t0 = time.perf_counter()
            self.compute()
            self.synchronize()
            samples.append(time.perf_counter() - t0)
        values = self.download()
        return BenchmarkResult(values, _timing_stats(samples))

    def benchmark_end_to_end(self, batch: np.ndarray, repeats: int = 7) -> BenchmarkResult:
        """Benchmark H2D + compute + D2H, excluding compilation/allocation."""
        if repeats < 1:
            raise ValueError("repeats must be at least one")
        self.solve(batch)  # warm-up

        samples = []
        values = None
        for _ in range(int(repeats)):
            self.synchronize()
            t0 = time.perf_counter()
            self.upload(batch)
            self.compute()
            self.synchronize()
            values = self.download()
            samples.append(time.perf_counter() - t0)
        return BenchmarkResult(values, _timing_stats(samples))

    def close(self):
        for buf in getattr(self, "_buffers", []):
            try:
                buf.free()
            except Exception:
                pass
        self._buffers = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class MultiGPUCustomBatchedSingularValues:
    """Distribute one batch across several CUDA devices.

    One retained primary context, one stream and one compiled solver are kept
    per selected device. Work is enqueued on every GPU before synchronization,
    so kernels and transfers can overlap across devices.
    """

    def __init__(self, total_batch_size: int, matrix_size: int = 4,
                 dtype=np.float64, device_ids: Optional[Iterable[int]] = None):
        self.drv, _ = _require_pycuda()
        self.drv.init()
        available = self.drv.Device.count()
        if device_ids is None:
            device_ids = list(range(available))
        self.device_ids = [int(x) for x in device_ids]
        if not self.device_ids:
            raise RuntimeError("No CUDA devices selected.")
        if any(x < 0 or x >= available for x in self.device_ids):
            raise ValueError(f"device_ids must be between 0 and {available - 1}.")

        self.total_batch_size = int(total_batch_size)
        if self.total_batch_size <= 0:
            raise ValueError("total_batch_size must be positive")
        self.device_ids = self.device_ids[:min(len(self.device_ids), self.total_batch_size)]
        self.n = int(matrix_size)
        self.dtype, _, _ = _dtype_info(dtype)

        counts = np.full(len(self.device_ids), self.total_batch_size // len(self.device_ids), dtype=int)
        counts[:self.total_batch_size % len(self.device_ids)] += 1
        self.counts = counts.tolist()
        self.slices = []
        start = 0
        for count in self.counts:
            self.slices.append((start, start + count))
            start += count

        self.contexts = []
        self.streams = []
        self.solvers = []
        self.h_inputs = []
        self.h_outputs = []

        for dev_id, count in zip(self.device_ids, self.counts):
            ctx = self.drv.Device(dev_id).retain_primary_context()
            ctx.push()
            try:
                stream = self.drv.Stream()
                solver = CustomBatchedSingularValues(count, self.n, self.dtype)
                h_in = self.drv.pagelocked_empty(count * self.n * self.n, self.dtype)
                h_out = self.drv.pagelocked_empty((count, self.n), self.dtype)
            finally:
                ctx.pop()
            self.contexts.append(ctx)
            self.streams.append(stream)
            self.solvers.append(solver)
            self.h_inputs.append(h_in)
            self.h_outputs.append(h_out)

    def _check_batch(self, batch):
        batch = np.asarray(batch, dtype=self.dtype)
        expected = (self.total_batch_size, self.n, self.n)
        if batch.shape != expected:
            raise ValueError(f"Expected batch shape {expected}, got {batch.shape}.")
        if not np.all(np.isfinite(batch)):
            raise ValueError("Input batch contains NaN or Inf values.")
        return batch

    def _stage_inputs(self, batch):
        for h_in, solver, (start, end) in zip(self.h_inputs, self.solvers, self.slices):
            prepared = solver._prepare_host(batch[start:end])
            h_in[:] = prepared

    def _enqueue_full_pipeline(self):
        for ctx, stream, solver, h_in, h_out in zip(
            self.contexts, self.streams, self.solvers, self.h_inputs, self.h_outputs
        ):
            ctx.push()
            try:
                self.drv.memcpy_htod_async(solver.d_input, h_in, stream)
                solver.compute(stream=stream)
                self.drv.memcpy_dtoh_async(h_out, solver.d_sv, stream)
            finally:
                ctx.pop()

    def _enqueue_compute(self):
        for ctx, stream, solver in zip(self.contexts, self.streams, self.solvers):
            ctx.push()
            try:
                solver.compute(stream=stream)
            finally:
                ctx.pop()

    def _enqueue_uploads(self):
        for ctx, stream, solver, h_in in zip(self.contexts, self.streams, self.solvers, self.h_inputs):
            ctx.push()
            try:
                self.drv.memcpy_htod_async(solver.d_input, h_in, stream)
            finally:
                ctx.pop()

    def _enqueue_downloads(self):
        for ctx, stream, solver, h_out in zip(self.contexts, self.streams, self.solvers, self.h_outputs):
            ctx.push()
            try:
                self.drv.memcpy_dtoh_async(h_out, solver.d_sv, stream)
            finally:
                ctx.pop()

    def synchronize(self):
        for ctx, stream in zip(self.contexts, self.streams):
            ctx.push()
            try:
                stream.synchronize()
            finally:
                ctx.pop()

    def _collect_outputs(self):
        return np.vstack([np.asarray(x).copy() for x in self.h_outputs])

    def solve(self, batch):
        batch = self._check_batch(batch)
        self._stage_inputs(batch)
        self._enqueue_full_pipeline()
        self.synchronize()
        return self._collect_outputs()

    def benchmark_device(self, batch, repeats: int = 7) -> BenchmarkResult:
        if repeats < 1:
            raise ValueError("repeats must be at least one")
        batch = self._check_batch(batch)
        self._stage_inputs(batch)
        self._enqueue_uploads()
        self.synchronize()
        self._enqueue_compute()  # warm-up
        self.synchronize()

        samples = []
        for _ in range(int(repeats)):
            self.synchronize()
            t0 = time.perf_counter()
            self._enqueue_compute()
            self.synchronize()
            samples.append(time.perf_counter() - t0)

        self._enqueue_downloads()
        self.synchronize()
        return BenchmarkResult(self._collect_outputs(), _timing_stats(samples))

    def benchmark_end_to_end(self, batch, repeats: int = 7) -> BenchmarkResult:
        if repeats < 1:
            raise ValueError("repeats must be at least one")
        batch = self._check_batch(batch)
        self.solve(batch)  # warm-up

        samples = []
        for _ in range(int(repeats)):
            self.synchronize()
            t0 = time.perf_counter()
            # Copying from the ordinary NumPy batch into pinned host staging
            # buffers is included in end-to-end timing.
            self._stage_inputs(batch)
            self._enqueue_full_pipeline()
            self.synchronize()
            samples.append(time.perf_counter() - t0)
        return BenchmarkResult(self._collect_outputs(), _timing_stats(samples))

    def close(self):
        for ctx, solver in zip(self.contexts, self.solvers):
            ctx.push()
            try:
                solver.close()
            finally:
                ctx.pop()
        self.solvers = []
        self.streams = []
        self.h_inputs = []
        self.h_outputs = []
        for ctx in self.contexts:
            try:
                ctx.detach()
            except Exception:
                pass
        self.contexts = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
