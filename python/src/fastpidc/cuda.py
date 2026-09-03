"""GPU-accelerated PUC computation.

Host-side driver for the shared CUDA kernels in
``fastpidc/kernels/pidc_kernels.cu`` (see that file for the algorithm and
the sharing strategy with FastPIDC.jl's CUDA extension). Requires the
optional ``cupy`` dependency (install with ``pip install fastpidc[cuda]``)
and a functional GPU.

Genes are processed along the target ("z") axis in chunks, to bound device
memory use, mirroring ``FastPIDCCUDAExt.compute_puc_full_cuda`` in
FastPIDC.jl.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .types import Node

__all__ = ["compute_puc_full_cuda", "cuda_available"]

_KERNEL_SOURCE_PATH = Path(__file__).with_name("kernels") / "pidc_kernels.cu"
_DEFAULT_CHUNK_SIZE = 256
_FALLBACK_CUDA_HEADER_DIRS = ("/usr", "/usr/local/cuda")


def _ensure_cuda_path() -> None:
    """Best-effort fallback for systems (e.g. some Debian/Ubuntu CUDA
    installs) where the CUDA toolkit headers live under a system prefix
    that cupy's nvrtc backend does not probe automatically.

    Must run before cupy is first imported: cupy resolves its include-path
    search once (effectively at import time), so setting ``CUDA_PATH`` any
    later has no effect. This is why it is invoked at import time below,
    rather than lazily inside ``_load_module``.
    """
    if "CUDA_PATH" in os.environ:
        return
    for candidate in _FALLBACK_CUDA_HEADER_DIRS:
        if (Path(candidate) / "include" / "cuda_runtime.h").exists():
            os.environ["CUDA_PATH"] = candidate
            return


_ensure_cuda_path()


def cuda_available() -> bool:
    """Whether ``cupy`` is importable and reports a functional GPU."""
    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def _load_module():
    import cupy as cp

    _ensure_cuda_path()
    source = _KERNEL_SOURCE_PATH.read_text()
    return cp.RawModule(code=source, options=("--std=c++11",))


def compute_puc_full_cuda(
    nodes: list[Node], *, base: float = 2, verbose: bool = False, chunk_size: int = _DEFAULT_CHUNK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """GPU implementation of :func:`fastpidc.puc.compute_puc_full`.

    ``base`` is accepted for signature parity with the CPU path but unused:
    mutual information is always computed in base 2 on the GPU, matching
    FastPIDC.jl's CUDA extension.
    """
    del base
    if not cuda_available():
        raise RuntimeError(
            "CUDA backend requested but cupy is not installed or no functional "
            "GPU was detected. Install the 'cuda' extra and ensure a GPU is "
            "available, or use config.backend = 'cpu'."
        )

    import cupy as cp

    module = _load_module()
    joint_counts_kernel = module.get_function("joint_counts_kernel")
    mi_si_kernel = module.get_function("mi_si_kernel")
    puc_accumulation_kernel = module.get_function("puc_accumulation_kernel")

    n = len(nodes)
    m = nodes[0].binned_values.size
    k_bins = max(node.number_of_bins for node in nodes)

    data_host = np.zeros((m, n), dtype=np.int32)
    marginals_host = np.zeros((k_bins, n), dtype=np.float64)
    for i, node in enumerate(nodes):
        data_host[:, i] = node.binned_values
        marginals_host[: node.number_of_bins, i] = node.probabilities

    data_gpu = cp.asarray(data_host)
    marginals_gpu = cp.asarray(marginals_host)

    puc_scores_gpu = cp.zeros((n, n), dtype=cp.float64)
    mi_matrix_gpu = cp.zeros((n, n), dtype=cp.float64)

    counts_chunk_gpu = cp.zeros((k_bins, k_bins, n, chunk_size), dtype=cp.int32)
    si_chunk_gpu = cp.zeros((k_bins, n, chunk_size), dtype=cp.float64)

    threads = (16, 16)

    if verbose:
        n_chunks = -(-n // chunk_size)
        print(f"[fastpidc] GPU chunked PUC: processing {n} x {n} pairs...")
        print(f"[fastpidc] Using chunk size {chunk_size} ({n_chunks} iterations)")

    for z_start in range(0, n, chunk_size):
        z_chunk_size = min(chunk_size, n - z_start)

        counts_chunk_gpu.fill(0)
        si_chunk_gpu.fill(0.0)

        blocks = (-(-n // threads[0]), -(-z_chunk_size // threads[1]))

        joint_counts_kernel(
            blocks,
            threads,
            (
                data_gpu,
                counts_chunk_gpu,
                np.int32(n),
                np.int32(m),
                np.int32(k_bins),
                np.int32(z_start),
                np.int32(z_chunk_size),
            ),
        )
        mi_si_kernel(
            blocks,
            threads,
            (
                counts_chunk_gpu,
                marginals_gpu,
                mi_matrix_gpu,
                si_chunk_gpu,
                np.int32(n),
                np.int32(m),
                np.int32(k_bins),
                np.int32(z_start),
                np.int32(z_chunk_size),
            ),
        )
        puc_accumulation_kernel(
            blocks,
            threads,
            (
                si_chunk_gpu,
                mi_matrix_gpu,
                puc_scores_gpu,
                marginals_gpu,
                np.int32(n),
                np.int32(k_bins),
                np.int32(z_start),
                np.int32(z_chunk_size),
            ),
        )

    puc_scores = cp.asnumpy(puc_scores_gpu)
    mi_matrix = cp.asnumpy(mi_matrix_gpu)

    # Symmetrize: each ordered pair (x, z) only holds one of the two
    # directional contributions (see the module docstring in pidc_kernels.cu).
    puc_scores = puc_scores + puc_scores.T

    return mi_matrix, puc_scores
