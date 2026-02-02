"""
Build an OHLCV-derived feature dataset (stock-level + market-level gating features)

Run from repo root:
  python build_ohlcv_feature_dataset.py --input data/cn_data/csi300.csv --output data/cn_data/ohlcv_feature_dataset.csv --dropna_label

This script is placed at repo root for convenience.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return a / b


def _ewm_mean(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = _safe_div(roll_up, roll_down)
    return 100 - (100 / (1 + rs))


def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("datetime")[col]
    return (df[col] - g.transform("mean")) / g.transform("std")


def _rank_pct_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("datetime")[col].rank(pct=True, method="average")


@dataclass(frozen=True)
class FeatureConfig:
    mom_windows: Tuple[int, ...] = (5, 10, 20, 60)
    vol_windows: Tuple[int, ...] = (5, 10, 20, 60)
    range_windows: Tuple[int, ...] = (5, 10, 20)
    volu_windows: Tuple[int, ...] = (5, 10, 20)
    mkt_windows: Tuple[int, ...] = (5, 20, 60)


def build_features(raw: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    df = raw.copy()
    required = ["instrument", "datetime", "$open", "$high", "$low", "$close", "$volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}. Expected: {required}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["instrument", "datetime"]).reset_index(drop=True)

    close = df["$close"]
    open_ = df["$open"]
    high = df["$high"]
    low = df["$low"]
    volu = df["$volume"].astype(float)

    # pandas future: default fill_method will change; set explicitly for stability
    df["ret_1d"] = df.groupby("instrument")["$close"].pct_change(fill_method=None)
    df["logret_1d"] = np.log(_safe_div(close, df.groupby("instrument")["$close"].shift(1)))
    df["hl_range"] = _safe_div(high - low, close)
    df["oc_ret"] = _safe_div(close - open_, open_)
    df["gap_ret"] = _safe_div(open_ - df.groupby("instrument")["$close"].shift(1), df.groupby("instrument")["$close"].shift(1))
    df["volu_log"] = np.log(volu.replace(0, np.nan))
    df["dollar_vol_proxy"] = volu * close

    prev_close = df.groupby("instrument")["$close"].shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["tr"] = tr
    df["atr_14"] = df.groupby("instrument")["tr"].transform(lambda s: s.rolling(14, min_periods=14).mean())
    df["atrp_14"] = _safe_div(df["atr_14"], close)

    df["rsi_14"] = df.groupby("instrument")["$close"].transform(lambda s: _rsi(s, 14))
    ema12 = df.groupby("instrument")["$close"].transform(lambda s: _ewm_mean(s, 12))
    ema26 = df.groupby("instrument")["$close"].transform(lambda s: _ewm_mean(s, 26))
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df.groupby("instrument")["macd"].transform(lambda s: _ewm_mean(s, 9))
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    for w in cfg.mom_windows:
        df[f"mom_{w}"] = df.groupby("instrument")["$close"].pct_change(w, fill_method=None)
        df[f"mom_log_{w}"] = df.groupby("instrument")["logret_1d"].transform(lambda s: s.rolling(w, min_periods=w).sum())
        df[f"price_ma_{w}"] = df.groupby("instrument")["$close"].transform(lambda s: s.rolling(w, min_periods=w).mean())
        df[f"price_ma_ratio_{w}"] = _safe_div(close, df[f"price_ma_{w}"])

    for w in cfg.vol_windows:
        df[f"vol_{w}"] = df.groupby("instrument")["logret_1d"].transform(lambda s: s.rolling(w, min_periods=w).std())
        # avoid DataFrameGroupBy.apply deprecation by working on the Series directly
        df[f"downvol_{w}"] = df.groupby("instrument")["logret_1d"].transform(
            lambda s: s.where(s < 0).rolling(w, min_periods=w).std()
        )

    for w in cfg.range_windows:
        df[f"hlr_mean_{w}"] = df.groupby("instrument")["hl_range"].transform(lambda s: s.rolling(w, min_periods=w).mean())
        df[f"tr_mean_{w}"] = df.groupby("instrument")["tr"].transform(lambda s: s.rolling(w, min_periods=w).mean())

    for w in cfg.volu_windows:
        df[f"volu_z_{w}"] = df.groupby("instrument")["volu_log"].transform(
            lambda s: (s - s.rolling(w, min_periods=w).mean()) / s.rolling(w, min_periods=w).std()
        )
        df[f"dvol_z_{w}"] = df.groupby("instrument")["dollar_vol_proxy"].transform(
            lambda s: (s - s.rolling(w, min_periods=w).mean()) / s.rolling(w, min_periods=w).std()
        )

    xs_cols = ["ret_1d", "mom_5", "mom_20", "vol_20", "hl_range", "volu_log", "dollar_vol_proxy"]
    for c in xs_cols:
        if c in df.columns:
            df[f"{c}_cs_z"] = _zscore_by_date(df, c)
            df[f"{c}_cs_rank"] = _rank_pct_by_date(df, c)

    def _nanmean_abs_dev(s: pd.Series) -> float:
        s = s.dropna()
        if s.empty:
            return np.nan
        mu = float(s.mean())
        return float(np.abs(s - mu).mean())

    mkt = df.groupby("datetime").agg(
        mkt_ret_mean=("ret_1d", "mean"),
        mkt_ret_median=("ret_1d", "median"),
        mkt_ret_std=("ret_1d", "std"),
        mkt_breadth_up=("ret_1d", lambda s: np.nanmean(s > 0)),
        mkt_breadth_down=("ret_1d", lambda s: np.nanmean(s < 0)),
        mkt_turnover_proxy=("dollar_vol_proxy", "sum"),
        mkt_liquidity_median=("dollar_vol_proxy", "median"),
        mkt_dispersion=("ret_1d", _nanmean_abs_dev),
    ).sort_index()

    for w in cfg.mkt_windows:
        mkt[f"mkt_ret_mean_{w}"] = mkt["mkt_ret_mean"].rolling(w, min_periods=w).mean()
        mkt[f"mkt_vol_{w}"] = mkt["mkt_ret_mean"].rolling(w, min_periods=w).std()
        mkt[f"mkt_breadth_up_{w}"] = mkt["mkt_breadth_up"].rolling(w, min_periods=w).mean()
        mkt[f"mkt_dispersion_{w}"] = mkt["mkt_dispersion"].rolling(w, min_periods=w).mean()
        mkt[f"mkt_turnover_proxy_{w}"] = mkt["mkt_turnover_proxy"].rolling(w, min_periods=w).mean()

    df = df.merge(mkt.reset_index(), on="datetime", how="left")
    df["label_ret_1d_fwd"] = df.groupby("instrument")["ret_1d"].shift(-1)
    return df


def add_master_alignment_columns(
    df: pd.DataFrame,
    trading_days_per_month: int = 20,
    beta_window: int = 21,
) -> pd.DataFrame:
    """
    Best-effort alignment to MASTER-like feature names using only OHLCV-derived data.

    Notes:
    - Fundamental/QMJ style factors (e.g. fcf_be, qmj_prof) cannot be derived from OHLCV.
      We create these columns as NaN so downstream pipelines expecting the schema won't break.
    - Return features are implemented with a common convention used in asset pricing:
      ret_{m}_{s}: m-month return skipping the most recent s months.
      We approximate 1 month = `trading_days_per_month` trading days.
    """
    out = df.copy()

    # schema columns
    out["id"] = out["instrument"]
    out["date"] = out["datetime"]

    # MASTER uses `ret` in many places; map to 1d return
    out["ret"] = out.get("ret_1d", np.nan)

    # ret_1_0: 1-month return, no skip (approx)
    m1 = trading_days_per_month

    def _ret_over(k: int) -> pd.Series:
        return out.groupby("instrument")["$close"].pct_change(k, fill_method=None)

    def _ret_m_skip(m: int, s: int) -> pd.Series:
        k = m * trading_days_per_month
        skip = s * trading_days_per_month
        # compute return ending at t-skip, then align back to t
        base = out.groupby("instrument")["$close"].pct_change(k, fill_method=None).shift(skip)
        return base

    # monthly-ish returns
    out["ret_1_0"] = _ret_over(m1)
    # 3m-1m, 6m-1m, 9m-1m, 12m-1m, 12m-7m
    out["ret_3_1"] = _ret_m_skip(3, 1)
    out["ret_6_1"] = _ret_m_skip(6, 1)
    out["ret_9_1"] = _ret_m_skip(9, 1)
    out["ret_12_1"] = _ret_m_skip(12, 1)
    out["ret_12_7"] = _ret_m_skip(12, 7)

    # beta_21d: rolling CAPM beta vs equal-weight market return (mkt_ret_mean)
    if "mkt_ret_mean" in out.columns and "ret_1d" in out.columns:
        x = out["mkt_ret_mean"]
        y = out["ret_1d"]

        # rolling beta per instrument: cov(y,x)/var(x)
        # Avoid DataFrameGroupBy.apply deprecation: compute cov/var via transforms
        out["_mkt_ret_mean"] = out["mkt_ret_mean"]
        out["_ret_1d"] = out["ret_1d"]

        # rolling cov(y, x)
        cov = out.groupby("instrument")[["_ret_1d", "_mkt_ret_mean"]].rolling(
            beta_window, min_periods=beta_window
        ).cov()
        # cov is MultiIndex with 2x2 matrix flattened; pick cov(y,x)
        # index: (instrument, original_index, variable)
        cov_yx = cov.xs("_ret_1d", level=2)["_mkt_ret_mean"].reset_index(level=0, drop=True)

        var_x = out.groupby("instrument")["_mkt_ret_mean"].transform(
            lambda s: s.rolling(beta_window, min_periods=beta_window).var()
        )
        out["beta_21d"] = cov_yx / var_x.replace(0, np.nan)
        out = out.drop(columns=["_mkt_ret_mean", "_ret_1d"])
    else:
        out["beta_21d"] = np.nan

    # ret_exc_lead1m: next-1-month excess return over market (EW)
    # lead1m implies forward return; use close-to-close forward 1m and subtract forward market return.
    if "mkt_ret_mean" in out.columns:
        fwd_stock_1m = out.groupby("instrument")["$close"].pct_change(-m1, fill_method=None)
        # market forward 1m: compounding of daily mean returns; approximate by rolling sum of log(1+r)
        mkt = out.groupby("datetime", as_index=True)["mkt_ret_mean"].first().sort_index()
        mkt_fwd_1m = (np.log1p(mkt)).rolling(m1, min_periods=m1).sum().shift(-(m1 - 1))
        mkt_fwd_1m = np.expm1(mkt_fwd_1m)
        out = out.merge(mkt_fwd_1m.rename("mkt_ret_fwd_1m").reset_index(), on="datetime", how="left")
        out["ret_exc_lead1m"] = fwd_stock_1m - out["mkt_ret_fwd_1m"]
    else:
        out["ret_exc_lead1m"] = np.nan

    # Create placeholder columns for factors that require fundamentals/QMJ/etc.
    placeholders = [
        "fcf_be",
        "div12m_me",
        "me_company",
        "debt_at",
        "qmj_prof",
        "qmj_growth",
        "qmj_safety",
    ]
    for c in placeholders:
        if c not in out.columns:
            out[c] = np.nan

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="input csv path (e.g. data/cn_data/csi300.csv)")
    parser.add_argument("--output", type=str, default=None, help="output csv path")
    parser.add_argument("--dropna_label", action="store_true", help="drop rows where forward label is NaN")
    parser.add_argument(
        "--master_align",
        action="store_true",
        help="Add best-effort MASTER-like columns (ret_6_1, beta_21d, ret_exc_lead1m, id/date, etc).",
    )
    parser.add_argument(
        "--tdpm",
        type=int,
        default=20,
        help="Trading days per month approximation used for MASTER ret_{m}_{s} features (default: 20).",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(repo_root, "data", "cn_data", "csi300.csv")
    default_output = os.path.join(repo_root, "data", "cn_data", "ohlcv_feature_dataset.csv")

    input_path = args.input or default_input
    output_path = args.output or default_output

    raw = pd.read_csv(input_path)
    feat = build_features(raw, FeatureConfig())
    if args.master_align:
        feat = add_master_alignment_columns(feat, trading_days_per_month=args.tdpm)
    if args.dropna_label:
        feat = feat.dropna(subset=["label_ret_1d_fwd"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feat.to_csv(output_path, index=False)
    # Avoid emoji/UTF-8 symbols for Windows consoles using GBK
    print(f"Saved: {output_path}")
    print(f"   rows={len(feat):,}, cols={len(feat.columns):,}")


if __name__ == "__main__":
    main()

