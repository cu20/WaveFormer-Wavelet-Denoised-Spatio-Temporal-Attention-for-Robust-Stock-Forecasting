from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch

try:
    import pywt  # type: ignore
except Exception:  # pragma: no cover
    pywt = None


ThresholdMode = Literal["soft", "hard"]


def _universal_threshold(detail_coeff: np.ndarray, n: int) -> float:
    """Universal threshold (VisuShrink) with sigma estimated by MAD."""
    if detail_coeff.size == 0:
        return 0.0
    # Robust sigma estimate for Gaussian noise
    med = np.median(detail_coeff)
    sigma = np.median(np.abs(detail_coeff - med)) / 0.6745
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0
    return float(sigma * np.sqrt(2.0 * np.log(max(n, 2))))


def wavelet_denoise_1d(
    y: np.ndarray,
    *,
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "periodization",
    threshold_mode: ThresholdMode = "soft",
    threshold_scale: float = 1.0,
) -> np.ndarray:
    """
    Wavelet denoise a 1D signal using soft/hard thresholding on detail coefficients.

    Notes:
    - This is designed for short rolling windows too; if the window is too short to
      decompose, it will return the original input.
    - NaN/Inf should be handled by the caller (we assume finite values here).
    """
    if pywt is None:
        raise ImportError(
            "PyWavelets is required for wavelet denoising. Install with: pip install PyWavelets"
        )

    y = np.asarray(y, dtype=np.float32)
    n = int(y.shape[0])
    if n < 4:
        return y

    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=n, filter_len=w.dec_len)
    if max_level < 1:
        return y

    use_level = level if level is not None else min(3, max_level)
    if use_level < 1:
        return y

    coeffs = pywt.wavedec(y, wavelet=w, mode=mode, level=use_level)
    # Estimate noise from the finest-scale detail coefficients
    thr = _universal_threshold(coeffs[-1], n)
    thr *= float(threshold_scale)
    if not np.isfinite(thr) or thr <= 0:
        return y

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], value=thr, mode=threshold_mode)

    rec = pywt.waverec(coeffs, wavelet=w, mode=mode).astype(np.float32, copy=False)
    # waverec can return length slightly different due to padding
    if rec.shape[0] != n:
        rec = rec[:n]
    return rec


@dataclass
class WaveletDenoiser:
    """
    Apply wavelet denoising to 3D features: (N, T, F), denoising along time axis T.

    Typical usage in this repo:
      - Apply only to factor features (e.g. 0:158), skip market info (158:221).
      - Call on CPU before moving tensors to GPU (for speed + pywt compatibility).
    """

    enabled: bool = True
    wavelet: str = "db4"
    level: Optional[int] = None
    mode: str = "periodization"
    threshold_mode: ThresholdMode = "soft"
    threshold_scale: float = 1.0
    feature_start: int = 0
    feature_end: Optional[int] = None  # exclusive

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if pywt is None:
            raise ImportError(
                "PyWavelets is required for wavelet denoising. Install with: pip install PyWavelets"
            )
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (N,T,F), got {tuple(x.shape)}")

        # We prefer to run pywt on CPU.
        orig_device = x.device
        x_cpu = x.detach().to("cpu")
        arr = x_cpu.numpy().astype(np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        n, t, f = arr.shape
        if t < 4:
            return x  # too short to denoise

        start = int(max(self.feature_start, 0))
        end = int(f if self.feature_end is None else min(self.feature_end, f))
        if end <= start:
            return x

        out = arr.copy()
        for fi in range(start, end):
            for ni in range(n):
                out[ni, :, fi] = wavelet_denoise_1d(
                    out[ni, :, fi],
                    wavelet=self.wavelet,
                    level=self.level,
                    mode=self.mode,
                    threshold_mode=self.threshold_mode,
                    threshold_scale=self.threshold_scale,
                )

        y = torch.from_numpy(out).to(dtype=x.dtype)
        if orig_device.type != "cpu":
            y = y.to(orig_device)
        return y

