#!/usr/bin/env python3
"""
检查当前 qlib 本地数据（cn_data）的交易日历与行情可用日期范围。
与 WaveFormer 默认配置一致：provider_uri=~/.qlib/qlib_data/cn_data，region=cn。

用法:
  python check_qlib_data_range.py
  python check_qlib_data_range.py --provider_uri ~/.qlib/qlib_data/cn_data --market csi300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Inspect qlib calendar & feature date range")
    parser.add_argument(
        "--provider_uri",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="qlib data directory (same as workflow yaml qlib_init.provider_uri)",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="csi300",
        help="instrument pool name, e.g. csi300, csi500, all",
    )
    parser.add_argument(
        "--sample_instruments",
        type=str,
        nargs="*",
        default=["SH600000", "SH000300"],
        help="stocks to probe with D.features($close); default: one stock + CSI300 index",
    )
    args = parser.parse_args()

    # qlib on path: project may run from /root/WaveFormer
    qlib_root = Path(__file__).resolve().parent / "qlib"
    if qlib_root.is_dir() and str(qlib_root) not in sys.path:
        sys.path.insert(0, str(qlib_root))

    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    provider_uri = str(Path(args.provider_uri).expanduser())
    print(f"provider_uri: {provider_uri}")
    print(f"market:       {args.market}")
    print()

    if not Path(provider_uri).expanduser().exists():
        print(f"[ERROR] Data path does not exist: {provider_uri}")
        sys.exit(1)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # ---- Trading calendar (full range in dump) ----
    cal = D.calendar(freq="day")
    print("=== Trading calendar (day) ===")
    print(f"  first trading day: {cal[0]}")
    print(f"  last trading day:  {cal[-1]}")
    print(f"  total days:        {len(cal)}")
    print()

    # ---- Instrument pool size at calendar ends ----
    try:
        inst = D.instruments(market=args.market)
        names = D.list_instruments(instruments=inst, as_list=True)
        if isinstance(names, dict):
            n = len(names)
        else:
            n = len(list(names))
        print(f"=== Pool `{args.market}` (as of list_instruments) ===")
        print(f"  instrument count: {n}")
    except Exception as e:
        print(f"=== Pool `{args.market}` ===")
        print(f"  (could not list) {e}")
    print()

    # ---- Per-symbol $close coverage ----
    print("=== Feature date range ($close) ===")
    start_probe = "1990-01-01"
    end_probe = "2035-12-31"
    for sym in args.sample_instruments:
        try:
            df = D.features(
                [sym],
                ["$close"],
                start_time=start_probe,
                end_time=end_probe,
                freq="day",
            )
            if df is None or df.empty:
                print(f"  {sym}: no rows")
                continue
            dt = df.index.get_level_values("datetime")
            print(f"  {sym}: {dt.min().date()}  ->  {dt.max().date()}  (rows={len(df)})")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")

    print()
    print("Tip: YAML `data_handler_config.end_time` must be <= last calendar day above")
    print("     and test segment must fall inside calendar for backtest to work.")


if __name__ == "__main__":
    main()
