# trains/singleTask/model/spectral_band.py
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_path_laplacian_eigvecs(
    T: int,
    normalized: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Path graph over T time steps: node i connected to i-1 and i+1.
    Return eigenvectors U (T,T) of (normalized) Laplacian, ascending eigenvalues.
    """
    A = torch.zeros((T, T), device=device, dtype=dtype)
    if T > 1:
        idx = torch.arange(T - 1, device=device)
        A[idx, idx + 1] = 1.0
        A[idx + 1, idx] = 1.0
    deg = A.sum(dim=1)

    if normalized:
        d_inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        L = torch.eye(T, device=device, dtype=dtype) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = torch.diag(deg) - A

    L = (L + L.t()) * 0.5  # numerical symmetry
    _, U = torch.linalg.eigh(L)
    return U


def _equal_energy_edges(energy: torch.Tensor, num_bands: int) -> List[int]:
    """
    energy: [T] non-negative
    returns edges [e1, e2, ...] with len=num_bands-1, 1<=ei<=T-1, strictly increasing
    """
    T = energy.numel()
    if num_bands <= 1 or T <= 1:
        return []
    energy = energy.clamp_min(0)
    total = energy.sum().clamp_min(1e-12)
    cum = torch.cumsum(energy, dim=0)

    edges: List[int] = []
    for b in range(1, num_bands):
        target = total * (b / num_bands)
        k = int(torch.searchsorted(cum, target).item())
        k = max(1, min(k, T - 1))
        edges.append(k)

    # enforce strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(edges[i - 1] + 1, T - 1)
    return edges


def _edges_from_ratios(T: int, ratios: Sequence[float]) -> List[int]:
    edges: List[int] = []
    prev = 0
    for r in ratios:
        k = int(round(float(r) * T))
        k = max(1, min(k, T - 1))
        k = max(k, prev + 1)
        k = min(k, T - 1)
        edges.append(k)
        prev = k
    out: List[int] = []
    for k in edges:
        if not out or k > out[-1]:
            out.append(k)
    return out


class GraphSpectralBandDecomposer(nn.Module):
    """
    x: [B, T, D]
    split into M band-limited components, each [B, T, D]
    """
    def __init__(
        self,
        length: int,
        num_bands: int = 3,
        partition: str = "equal_energy",  # "equal_energy" | "ratio"
        band_edges: Optional[Sequence[int]] = None,  # fixed edges (preferred once you precompute)
        band_edge_ratios: Optional[Sequence[float]] = (1/3, 2/3),
        normalized_laplacian: bool = True,
        basis_device: str = "cpu",
    ) -> None:
        super().__init__()
        assert partition in {"equal_energy", "ratio"}
        self.length = int(length)
        self.num_bands = int(num_bands)
        self.partition = partition
        self._fixed_edges = list(band_edges) if band_edges is not None else None
        self._ratios = tuple(band_edge_ratios) if band_edge_ratios is not None else None

        # Precompute basis U on CPU (small constant), then move with module
        U = _build_path_laplacian_eigvecs(
            T=self.length,
            normalized=normalized_laplacian,
            device=torch.device(basis_device),
            dtype=torch.float32,
        )
        self.register_buffer("U", U, persistent=False)

    def _get_edges(self, x_hat: torch.Tensor) -> List[int]:
        if self.num_bands <= 1:
            return []
        if self._fixed_edges is not None:
            return [int(e) for e in self._fixed_edges]
        if self.partition == "ratio":
            ratios = self._ratios if self._ratios is not None else (1/3, 2/3)
            return _edges_from_ratios(self.length, ratios)

        # equal-energy on current batch (建议后续用离线预计算替代它，训练更稳)
        energy = x_hat.pow(2).mean(dim=(0, 2))  # [T]
        return _equal_energy_edges(energy, self.num_bands)

    def split_bands(self, x: torch.Tensor) -> List[torch.Tensor]:
        assert x.dim() == 3, f"expected [B,T,D], got {tuple(x.shape)}"
        B, T, D = x.shape
        assert T == self.length, f"expected T={self.length}, got {T}"

        U = self.U.to(device=x.device, dtype=x.dtype)
        x_hat = torch.einsum("kt,btd->bkd", U.transpose(0, 1), x)  # [B,T,D]

        edges = self._get_edges(x_hat)
        ends = list(edges) + [T]

        bands: List[torch.Tensor] = []
        s = 0
        for e in ends:
            U_band = U[:, s:e]             # [T,K]
            x_hat_band = x_hat[:, s:e, :]  # [B,K,D]
            x_band = torch.einsum("tk,bkd->btd", U_band, x_hat_band)
            bands.append(x_band)
            s = e

        while len(bands) < self.num_bands:
            bands.append(torch.zeros_like(bands[-1]))
        return bands[:self.num_bands]


class BandRouter(nn.Module):
    """
    Input: band pooled embeddings [B, M, D]
    Output: band weights [B, M]
    """
    def __init__(self, embed_dim: int, num_bands: int = 3, temperature: float = 0.1) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.temperature = float(temperature)
        hidden = max(64, embed_dim)

        self.fc1 = nn.Linear(embed_dim * self.num_bands, hidden)
        self.fc2 = nn.Linear(hidden, self.num_bands)

    def forward(self, band_pooled: torch.Tensor) -> torch.Tensor:
        B, M, D = band_pooled.shape
        assert M == self.num_bands
        x = band_pooled.reshape(B, M * D)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        w = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        return w


def apply_band_masking(w: torch.Tensor, mask_rate: float) -> torch.Tensor:
    """
    SBM-lite: randomly mask bands then renormalize.
    w: [B,M] (already softmax)
    """
    if (not w.requires_grad) and mask_rate <= 0:
        return w
    if mask_rate <= 0:
        return w
    B, M = w.shape
    device = w.device
    keep = torch.bernoulli(torch.full((B, M), 1.0 - mask_rate, device=device))
    # ensure at least one band kept per sample
    all_zero = (keep.sum(dim=1, keepdim=True) == 0)
    keep = torch.where(all_zero, torch.ones_like(keep), keep)
    w_masked = w * keep
    w_masked = w_masked / w_masked.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return w_masked
