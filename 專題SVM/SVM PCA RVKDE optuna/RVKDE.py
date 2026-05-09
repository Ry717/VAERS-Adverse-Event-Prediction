"""
gpu_kde.py — GPU-accelerated KDE utilities with CuPy + (cuML or FAISS-GPU)

This version keeps only the pairwise cross-group density implementation.

Provided utilities
------------------
- rvkde_sigmas(..., backend="cuml"|"faiss")  : per-sample sigma estimation
- build_nn_kernels(..., backend="cuml"|"faiss"): neighbor search builder
- cross_group_density_pairwise(...)          : use K2 neighbors for KDE
- free_gpu()                                 : free CuPy memory pool

"""

from __future__ import annotations

from typing import Tuple, Optional, Literal, Any

import numpy as np
from sklearn.mixture import GaussianMixture

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as cp_logsumexp
except Exception:
    cp = None
    cp_logsumexp = None  # type: ignore

try:
    from cuml.neighbors import NearestNeighbors as cuMLNearestNeighbors  # type: ignore
except Exception:
    cuMLNearestNeighbors = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

from scipy.special import gamma as sp_gamma


def _require_cupy():
    if cp is None:
        raise ImportError("cupy is required for this function, but is not installed.")

def _ensure_backend(backend: Literal["cuml", "faiss"]) -> Literal["cuml", "faiss"]:
    if backend not in ("cuml", "faiss"):
        raise ValueError("backend must be 'cuml' or 'faiss'")
    if backend == "cuml" and cuMLNearestNeighbors is None:
        raise ImportError("cuML is not available. Try backend='faiss'.")
    if backend == "faiss" and faiss is None:
        raise ImportError("FAISS is not available. Please install faiss-gpu.")
    return backend

def free_gpu():
    if cp is not None:
        cp._default_memory_pool.free_all_blocks()


class _KNNWrapper:
    """Unifies cuML/FAISS neighbor search APIs."""
    def __init__(
        self,
        backend: Literal["cuml", "faiss"],
        model: Any,
        X_train: np.ndarray,
        faiss_gpu_res: Any = None,
        faiss_on_gpu: bool = True,
    ):
        self.backend = backend
        self.model = model
        self.X_train = X_train
        self.faiss_gpu_res = faiss_gpu_res
        self.faiss_on_gpu = faiss_on_gpu

    def kneighbors(self, X: np.ndarray, return_distance: bool = True, n_neighbors: Optional[int] = None):
        X = np.asarray(X, dtype=np.float32)
        k = n_neighbors if n_neighbors is not None else getattr(self, "_K", None)
        if k is None:
            raise RuntimeError(f"{self.backend} wrapper needs n_neighbors or preset _K.")

        if self.backend == "cuml":
            if return_distance:
                d, ind = self.model.kneighbors(X, n_neighbors=int(k), return_distance=True)
                return d, ind
            else:
                return self.model.kneighbors(X, n_neighbors=int(k), return_distance=False)
        else:
            D, I = self.model.search(X, int(k))
            if return_distance:
                return D.astype(np.float32, copy=False), I.astype(np.int32, copy=False)
            else:
                return I.astype(np.int32, copy=False)


def build_nn_kernels(X_kernels: np.ndarray, K2: int, metric: str = "euclidean",
                     backend: Literal["cuml", "faiss"] = "cuml",
                     faiss_use_gpu: bool = True, faiss_device: int = 0) -> _KNNWrapper:
    backend = _ensure_backend(backend)
    if metric != "euclidean":
        raise NotImplementedError("Only 'euclidean' is supported.")

    X_kernels = np.ascontiguousarray(X_kernels, dtype=np.float32)
    N, d = X_kernels.shape
    K2 = max(1, min(int(K2), N))

    if backend == "cuml":
        nn = cuMLNearestNeighbors(n_neighbors=K2, metric="euclidean", output_type="numpy")
        nn.fit(X_kernels)  # 這裡用 CPU numpy 即可，cuML 內部會搬到 GPU
        wrap = _KNNWrapper("cuml", nn, X_kernels)
    else:
        cpu_index = faiss.IndexFlatL2(d)
        use_gpu = faiss_use_gpu and hasattr(faiss, "StandardGpuResources")
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, faiss_device, cpu_index)
                gpu_index.add(X_kernels)
                wrap = _KNNWrapper("faiss", gpu_index, X_kernels, faiss_gpu_res=res, faiss_on_gpu=True)
            except Exception:
                cpu_index.add(X_kernels)
                wrap = _KNNWrapper("faiss", cpu_index, X_kernels, faiss_gpu_res=None, faiss_on_gpu=False)
        else:
            cpu_index.add(X_kernels)
            wrap = _KNNWrapper("faiss", cpu_index, X_kernels, faiss_gpu_res=None, faiss_on_gpu=False)

    setattr(wrap, "_K", K2)  # 只是預設 實際查詢時可用 kneighbors(..., n_neighbors=...) 覆蓋
    return wrap



def rvkde_sigmas(
    samples: np.ndarray,
    beta: float,
    smoothing: bool = True,
    K: Optional[int] = None,
    dim: int = 8,
    batch_size: int = 2048,
    min_sigma: float = 1e-3,
    backend: Literal["cuml", "faiss"] = "cuml",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    backend = _ensure_backend(backend)
    samples = np.asarray(samples, dtype=np.float32)
    N, d = samples.shape
    assert dim == d, f"[rvkde_sigmas] dim={dim} != data dim={d}"

    if K is None:
        K = max(1, N // 100)
    K = int(max(1, min(K, N - 1)))

    R_scale = np.sqrt(np.pi) / ((K + 1) * sp_gamma(dim / 2 + 1)) ** (1.0 / dim)
    sigma_scale = R_scale * beta * (dim + 1) / dim / K
    sigmas = np.zeros(N, dtype=np.float32)

    # 第一階段：計算初始 Sigmas
    if backend == "cuml":
        _require_cupy()
        eps = 1e-18
        nn = cuMLNearestNeighbors(n_neighbors=K + 1, metric="euclidean")
        nn.fit(cp.asarray(samples))
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = cp.asarray(samples[start:end])
            
            # --- 修正處：強健地接收 cuML 回傳值 ---
            res = nn.kneighbors(batch, return_distance=True)
            dists = res[0] if isinstance(res, (tuple, list)) else res
            
            is_self = (dists[:, 0] <= eps)
            dsel = cp.where(is_self[:, None], dists[:, 1:], dists[:, :K])
            local_sum = cp.sum(dsel, axis=1)
            sigmas[start:end] = cp.asnumpy(cp.maximum(local_sum * sigma_scale, min_sigma))
            del batch, res, dists, is_self, dsel, local_sum
            free_gpu()
    else:
        eps_sq = 1e-18
        knn_index = build_nn_kernels(samples, K + 1, backend="faiss")
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            # Faiss 通常很穩定，但我們也統一處理
            res = knn_index.kneighbors(samples[start:end], return_distance=True)
            D2, _ = res if isinstance(res, (tuple, list)) else (res, None)
            
            is_self = (D2[:, 0] <= eps_sq)
            dsel = np.where(is_self[:, None], D2[:, 1:], D2[:, :K])
            dsel = np.sqrt(np.clip(dsel, 0.0, None))
            sigmas[start:end] = np.maximum(dsel.sum(axis=1) * sigma_scale, min_sigma)

    # 第二階段：平滑化 (Smoothing)
    if smoothing:
        smoothed_sigmas = np.zeros(N, dtype=np.float32)
        sigmas_cp = cp.asarray(sigmas) if backend == "cuml" else sigmas
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            if backend == "cuml":
                batch = cp.asarray(samples[start:end])
                res = nn.kneighbors(batch, return_distance=False)
                
                # --- 修正處：確保 indices 取得正確 ---
                indices = res[0] if isinstance(res, (tuple, list)) else res
                
                neighbor_sigmas = sigmas_cp[indices]
                smoothed_sigmas[start:end] = cp.asnumpy(neighbor_sigmas.mean(axis=1))
                del batch, res, indices, neighbor_sigmas
            else:
                res = knn_index.kneighbors(samples[start:end], return_distance=False)
                I = res[0] if isinstance(res, (tuple, list)) else res
                smoothed_sigmas[start:end] = sigmas[I].mean(axis=1)
                del I
        sigmas = np.maximum(smoothed_sigmas, min_sigma).astype(np.float32)

    return sigmas, None

##############################################
def kde_log_mean_from_d2(
    dist_sq_batch,
    sigmas_batch,
    dim: int = 8,
) -> np.ndarray:
    _require_cupy()
    d2 = cp.asarray(dist_sq_batch, dtype=cp.float32)
    sig = cp.asarray(sigmas_batch, dtype=cp.float32)
    log_prob = -0.5 * (dim * cp.log(2 * cp.pi) + 2.0 * dim * cp.log(sig) + d2 / (sig * sig))
    out = cp_logsumexp(log_prob, axis=1) - cp.log(log_prob.shape[1])
    res = cp.asnumpy(out)
    free_gpu()
    return res
###############################################

def cross_group_density_pairwise(
    X_query: np.ndarray,
    X_kernels,
    sigmas_k,
    nn,                      # _KNNWrapper（或具備 kneighbors 的物件）
    K2: int,
    dim: int = 8,
    batch_size: int = 2048,
    same_dataset: bool = True,
    use_gpu: Optional[bool] = None,  # None=自動(cp 存在就用 GPU)
) -> np.ndarray:
    """
    以每個 query 的 K2 個最近 kernel 做等權 Gaussian KDE（log-mean），
    其中：
      - 若 same_dataset=True，會查詢 K2+1 並丟掉第一個（視為 self）
      - 若 same_dataset=False，直接取前 K2。

    parameter
    ----
    X_query : (M, d) float32
    X_kernels : (N, d) float32（CPU 或 GPU 皆可；本函式會自動轉成對應後端）
    sigmas_k : (N,) or (N,1) float32
    nn : 需提供 kneighbors(X, return_distance: bool, [n_neighbors: int]) 介面
         *若不支援 n_neighbors 參數，會自動 fallback
    K2 : int  最終使用的鄰居數
    dim : int 資料維度 d
    batch_size : int
    same_dataset : bool
    use_gpu : Optional[bool]  # 指定 True/False；None 時以 cp 是否可用自動判斷

    return
    ----
    log_density : (M,) float32  每個 query 的 log-mean KDE 值
    """
    #包一層 kneighbors，支援無 n_neighbors 的 wrapper
    def _kneighbors(ann, X, return_distance: bool, k: int):
        try:
            return ann.kneighbors(X, return_distance=return_distance, n_neighbors=int(k))
        except TypeError:
            return ann.kneighbors(X, return_distance=return_distance)

    X_query = np.asarray(X_query, dtype=np.float32)
    K2 = int(max(1, K2))
    M, d = X_query.shape
    assert dim == d, f"[cross_group_density_pairwise] dim={dim} != data dim={d}"

    gpu_available = (cp is not None)
    if use_gpu is None:
        use_gpu = gpu_available
    use_gpu = bool(use_gpu and gpu_available)


    # GPU 路徑（CuPy）

    if use_gpu:
        _require_cupy()
        Xk_gpu = X_kernels if isinstance(X_kernels, cp.ndarray) else cp.asarray(X_kernels, dtype=cp.float32)
        sk_gpu = sigmas_k if isinstance(sigmas_k, cp.ndarray) else cp.asarray(sigmas_k, dtype=cp.float32)
        if sk_gpu.ndim == 2 and sk_gpu.shape[1] == 1:
            sk_gpu = sk_gpu.ravel()

        log2pi = cp.log(2 * cp.pi)
        out = cp.empty((M,), dtype=cp.float32)

        for s in range(0, M, batch_size):
            e = min(s + batch_size, M)
            k_query = (K2 + 1) if same_dataset else K2

            I_np = _kneighbors(nn, X_query[s:e], return_distance=False, k=k_query)
            I_gpu = cp.asarray(I_np, dtype=cp.int32)
            if same_dataset:
                inds = I_gpu[:, 1:]
            else:
                inds = I_gpu[:, :K2]
            used_K = inds.shape[1]

            centers = cp.asarray(X_query[s:e], dtype=cp.float32)
            neigh = Xk_gpu[inds]
            diff = neigh - centers[:, None, :]
            d2 = cp.sum(diff * diff, axis=2)

            sig = sk_gpu[inds]
            log_sigma = cp.log(sig)
            sigma2 = sig * sig

            log_prob = -0.5 * (dim * log2pi + 2.0 * dim * log_sigma + d2 / sigma2)
            out[s:e] = (cp_logsumexp(log_prob, axis=1) - cp.log(int(used_K))).astype(cp.float32)

            del centers, neigh, diff, d2, sig, log_sigma, sigma2, log_prob, inds
            free_gpu()

        res = cp.asnumpy(out)
        del out
        free_gpu()
        return res

    
    # CPU 路徑（NumPy/SciPy）

    from scipy.special import logsumexp
    Xk = np.asarray(X_kernels, dtype=np.float32)
    sk = np.asarray(sigmas_k, dtype=np.float32).reshape(-1)
    out = np.empty((M,), dtype=np.float32)

    for s in range(0, M, batch_size):
        e = min(s + batch_size, M)
        k_query = (K2 + 1) if same_dataset else K2

        I = _kneighbors(nn, X_query[s:e], return_distance=False, k=k_query)
        if same_dataset:
            inds = I[:, 1:]
        else:
            inds = I[:, :K2]
        used_K = inds.shape[1]

        centers = X_query[s:e].astype(np.float32, copy=False)
        neigh = Xk[inds]
        diff = neigh - centers[:, None, :]
        d2 = (diff * diff).sum(axis=2)

        sig = sk[inds]
        log_prob = -0.5 * (dim * np.log(2*np.pi) + 2.0 * dim * np.log(sig) + d2/(sig*sig))
        out[s:e] = (logsumexp(log_prob, axis=1) - np.log(int(used_K))).astype(np.float32)

    return out



__all__ = [
    "free_gpu",
    "rvkde_sigmas",
    "build_nn_kernels",
    "cross_group_density_pairwise",
]

if __name__ == "__main__":
    print("gpu_kde.py loaded with cuML/FAISS backends. cross_group_density_pairwise only.")