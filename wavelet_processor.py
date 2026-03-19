"""
WaveletDenoiseProcessor — qlib-compatible processor for CPU-based Haar wavelet
denoising applied *before* RobustZScoreNorm in the data handler pipeline.

Designed for the 3-way ablation experiment:
  Baseline  vs  Denoise-before-norm  vs  Denoise-after-norm

Usage in handler config (infer_processors, prepended before RobustZScoreNorm):
    - class: WaveletDenoiseProcessor
      module_path: wavelet_processor
      kwargs:
          level: 1
          threshold_method: bayes   # bayes | visu
          threshold_scale: 0.5
          blend: 1.0                # 1.0 = full denoising
          finest_only: true
"""

import math
import numpy as np
import pandas as pd

from qlib.data.dataset.processor import Processor

_SQRT2 = 0.7071067811865476


# ---------------------------------------------------------------------------
# Haar DWT helpers (numpy, CPU)
# ---------------------------------------------------------------------------

def _haar_dwt1(x: np.ndarray):
    """Single-level Haar DWT: x (n,) -> (cA, cD) each (n//2,)."""
    ev = x[0::2]
    od = x[1::2]
    cA = (ev + od) * _SQRT2
    cD = (ev - od) * _SQRT2
    return cA, cD


def _haar_idwt1(cA: np.ndarray, cD: np.ndarray) -> np.ndarray:
    """Inverse Haar: (cA, cD) -> x (2*n,)."""
    n = len(cA)
    x = np.empty(2 * n, dtype=cA.dtype)
    x[0::2] = (cA + cD) * _SQRT2
    x[1::2] = (cA - cD) * _SQRT2
    return x


def _pad_to_pow2(x: np.ndarray):
    """Edge-replicate pad x to the next power of 2. Returns (padded, pad_left, orig_len)."""
    n = len(x)
    next_pow2 = 1
    while next_pow2 < n:
        next_pow2 *= 2
    if next_pow2 == n:
        return x, 0, n
    pad_total = next_pow2 - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded = np.concatenate([
        np.full(pad_left, x[0]),
        x,
        np.full(pad_right, x[-1]),
    ])
    return padded, pad_left, n


def _visu_threshold(cD: np.ndarray, scale: float) -> float:
    n = max(len(cD), 2)
    med = np.median(cD)
    mad = np.median(np.abs(cD - med))
    sigma = mad / 0.6745
    return float(sigma * math.sqrt(2.0 * math.log(n)) * scale)


def _bayes_threshold(cD: np.ndarray, scale: float) -> float:
    med = np.median(cD)
    mad = np.median(np.abs(cD - med))
    sigma_n = mad / 0.6745
    sigma_n2 = sigma_n ** 2
    var_cD = np.var(cD)
    sigma_s2 = max(var_cD - sigma_n2, 1e-10)
    sigma_s = math.sqrt(sigma_s2)
    return float((sigma_n2 / sigma_s) * scale)


def _soft_thresh(x: np.ndarray, thr: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def denoise_series(x: np.ndarray, level: int = 1,
                   threshold_method: str = "bayes",
                   threshold_scale: float = 0.5,
                   blend: float = 1.0,
                   finest_only: bool = True) -> np.ndarray:
    """
    Denoise a 1-D time series (stock × time axis) with Haar DWT.

    Parameters
    ----------
    x                : 1-D array, time series for a single (stock, feature) pair
    level            : number of DWT levels
    threshold_method : 'bayes' | 'visu'
    threshold_scale  : scale applied to threshold
    blend            : output = (1-blend)*raw + blend*denoised
    finest_only      : if True, only threshold finest-level detail coefficients
    """
    raw = x.copy()
    # Drop NaN mask — work on non-NaN values only
    valid_mask = ~np.isnan(x)
    if valid_mask.sum() < 4:
        return raw

    x_valid = x[valid_mask]
    orig_len = len(x_valid)

    # Edge-pad to power of 2
    x_pad, pad_left, _ = _pad_to_pow2(x_valid)

    # Forward DWT
    details = []
    cur = x_pad
    for _ in range(level):
        if len(cur) < 2:
            break
        cA, cD = _haar_dwt1(cur)
        details.append(cD)
        cur = cA

    if not details:
        return raw

    # Threshold
    details_t = []
    for i, cD in enumerate(details):
        if finest_only and i > 0:
            details_t.append(cD)
        else:
            if threshold_method == "bayes":
                thr = _bayes_threshold(cD, threshold_scale)
            else:
                thr = _visu_threshold(cD, threshold_scale)
            details_t.append(_soft_thresh(cD, thr))

    # Inverse DWT
    rec = cur
    for i in range(len(details_t) - 1, -1, -1):
        rec = _haar_idwt1(rec, details_t[i])

    # Trim padding and reconstruction
    rec = rec[pad_left: pad_left + orig_len]

    # Blend
    denoised_valid = (1.0 - blend) * x_valid + blend * rec

    out = raw.copy()
    out[valid_mask] = denoised_valid
    return out


# ---------------------------------------------------------------------------
# Qlib Processor
# ---------------------------------------------------------------------------

class WaveletDenoiseProcessor(Processor):
    """
    Applies Haar wavelet denoising column-wise (per feature, across the time
    axis of each stock) to the DataFrame produced by the data handler.

    The DataFrame index is (datetime, instrument) and columns are features.
    We iterate over each (instrument, feature) pair and denoise its time series.
    """

    def __init__(
        self,
        level: int = 1,
        threshold_method: str = "bayes",
        threshold_scale: float = 0.5,
        blend: float = 1.0,
        finest_only: bool = True,
    ):
        self.level = level
        self.threshold_method = threshold_method
        self.threshold_scale = threshold_scale
        self.blend = blend
        self.finest_only = finest_only

    def fit(self, df: pd.DataFrame = None):
        """No fitting needed."""
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df : MultiIndex DataFrame (datetime, instrument) × features
        Returns denoised DataFrame with same shape and index.
        """
        if df is None or df.empty:
            return df

        out = df.copy()

        # Group by instrument, apply denoising to each feature's time series
        instruments = df.index.get_level_values("instrument").unique()
        for inst in instruments:
            inst_data = df.xs(inst, level="instrument")   # (T, F) DataFrame
            for col in inst_data.columns:
                series = inst_data[col].values.astype(np.float64)
                denoised = denoise_series(
                    series,
                    level=self.level,
                    threshold_method=self.threshold_method,
                    threshold_scale=self.threshold_scale,
                    blend=self.blend,
                    finest_only=self.finest_only,
                )
                out.loc[(slice(None), inst), col] = denoised

        return out
