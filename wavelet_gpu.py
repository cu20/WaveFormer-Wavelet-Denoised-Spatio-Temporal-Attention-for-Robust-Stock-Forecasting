"""
GPU-native wavelet denoising layer for (N, T, F) feature tensors.

Key design goals (per 日志.md 2026/3/14):
  - Runs entirely on GPU via PyTorch — no pywt, no CPU loops.
  - Haar uses the lifting scheme (exact, perfect reconstruction).
  - db2/db3/db4 use periodic-padded conv1d (Haar recommended for T=8).
  - Supports soft / hard thresholding with per-signal VisuShrink threshold.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_WAVELETS = ("haar", "db1")
# For T=8, only Haar works correctly; db2+ need longer signals
_SQRT2 = 0.7071067811865476


def _haar_dwt1(x: torch.Tensor) -> tuple:
    """
    Single-level Haar DWT via lifting. x: (B, T) -> (cA, cD) each (B, T//2).
    Perfect reconstruction guaranteed.
    """
    # x[2k], x[2k+1] -> s[k]=(x[2k]+x[2k+1])/√2, d[k]=(x[2k]-x[2k+1])/√2
    ev = x[:, 0::2]   # (B, T//2)
    od = x[:, 1::2]   # (B, T//2)
    cA = (ev + od) * _SQRT2
    cD = (ev - od) * _SQRT2
    return cA, cD


def _haar_idwt1(cA: torch.Tensor, cD: torch.Tensor) -> torch.Tensor:
    """Inverse Haar: (cA, cD) -> x (B, 2*n)."""
    ev = (cA + cD) * _SQRT2
    od = (cA - cD) * _SQRT2
    B, n = cA.shape
    x = torch.empty(B, 2 * n, dtype=cA.dtype, device=cA.device)
    x[:, 0::2] = ev
    x[:, 1::2] = od
    return x


class GpuWaveletDenoiser(nn.Module):
    """
    Wavelet denoising layer operating on 3-D tensors (N, T, F).

    Denoising is applied along the **time axis T** independently for each
    (stock n, feature f) pair in the range [feature_start, feature_end).

    Parameters
    ----------
    wavelet        : 'haar' | 'db1' | 'db2' | 'db3' | 'db4'
    level          : DWT decomposition levels. None = auto (≤3).
    threshold_mode : 'soft' | 'hard'
    threshold_scale: multiply the universal threshold by this factor.
    feature_start  : first feature column to denoise (inclusive).
    feature_end    : last feature column to denoise (exclusive). None = all.
    """

    def __init__(
        self,
        wavelet: str = "haar",
        level: Optional[int] = None,
        threshold_mode: str = "soft",
        threshold_scale: float = 1.0,
        feature_start: int = 0,
        feature_end: Optional[int] = None,
    ):
        super().__init__()
        wavelet = wavelet.lower()
        if wavelet not in SUPPORTED_WAVELETS:
            raise ValueError(
                f"wavelet '{wavelet}' not supported; use 'haar' or 'db1'"
            )
        self.level = level
        self.threshold_mode = threshold_mode
        self.threshold_scale = float(threshold_scale)
        self.feature_start = feature_start
        self.feature_end = feature_end

    def _auto_level(self, t: int) -> int:
        """Max valid level for Haar: T=8 -> level 3."""
        lvl, cur = 0, t
        while cur >= 2 and lvl < 3:
            cur = cur // 2
            lvl += 1
        return max(lvl, 1)

    @staticmethod
    def _universal_threshold(cD: torch.Tensor, scale: float) -> torch.Tensor:
        """
        VisuShrink: thr = sigma * sqrt(2 * log(n)) * scale
        sigma estimated via MAD on the finest-level detail coefficients.
        cD  : (B, 1, n)
        thr : (B, 1, 1)   — one threshold per signal
        """
        n = max(cD.shape[2], 2)
        med = cD.median(dim=2, keepdim=True).values
        mad = (cD - med).abs().median(dim=2, keepdim=True).values
        sigma = mad / 0.6745
        thr = sigma * math.sqrt(2.0 * math.log(n)) * scale
        return thr.clamp(min=0.0)

    @staticmethod
    def _soft_thresh(x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
        return x.sign() * (x.abs() - thr).clamp(min=0.0)

    @staticmethod
    def _hard_thresh(x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
        return x * (x.abs() > thr)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, T, F)  ->  denoised (N, T, F)
        Uses Haar lifting scheme for exact DWT/IDWT.
        """
        if x.ndim != 3:
            raise ValueError(f"GpuWaveletDenoiser expects (N,T,F), got {tuple(x.shape)}")

        N, T, F = x.shape
        f_end = F if self.feature_end is None else min(int(self.feature_end), F)
        f_start = max(int(self.feature_start), 0)

        if f_end <= f_start or T < 4:
            return x

        level = self._auto_level(T) if self.level is None else int(self.level)
        if level < 1:
            return x

        # (N, T, F_sub) -> (B, T) where B = N*F_sub
        x_sub = x[:, :, f_start:f_end]
        F_sub = x_sub.shape[2]
        sig = x_sub.permute(0, 2, 1).reshape(N * F_sub, T)

        # ---- Haar multi-level forward DWT ----
        details = []
        cur = sig
        for _ in range(level):
            cA, cD = _haar_dwt1(cur)
            details.append(cD)
            cur = cA

        # ---- VisuShrink threshold (from finest level) applied to all detail levels ----
        thr = self._universal_threshold(details[0].unsqueeze(1), self.threshold_scale)
        thresh_fn = self._soft_thresh if self.threshold_mode == "soft" else self._hard_thresh
        details_t = [thresh_fn(d.unsqueeze(1), thr).squeeze(1) for d in details]

        # ---- multi-level inverse DWT ----
        rec = cur
        for i in range(level - 1, -1, -1):
            rec = _haar_idwt1(rec, details_t[i])

        # Ensure length T (in case of odd lengths)
        rec = rec[:, :T]

        # (B, T) -> (N, T, F_sub)
        rec = rec.reshape(N, F_sub, T).permute(0, 2, 1)

        out = x.clone()
        out[:, :, f_start:f_end] = rec.to(dtype=x.dtype)
        return out
