"""
signal_today_xgb.py — 輸出今日 Round 6 Batch 3c（Kronos embedding + XGBoost）選股訊號

用法:
  python -m finetune_tw.signal_today_xgb --config finetune_tw/configs/config_tw_daily_rtx6000.yaml \
      --xgb_model finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/xgb_batch3c_full.json
  python -m finetune_tw.signal_today_xgb --config ... --xgb_model ... --date 2026-06-20
  python -m finetune_tw.signal_today_xgb --config ... --xgb_model ... --holdings 2330,2317
"""

import argparse
import sys

import pandas as pd
import xgboost as xgb

from finetune_tw.backtest import build_model_specs, load_predictor_from_spec, rank_stocks
from finetune_tw.backtest_xgb_embedding import compute_xgb_signals, _require_model_feature_columns
from finetune_tw.config import Config
from finetune_tw.db import list_symbols
from finetune_tw.signal_today import _last_trading_day


def main() -> None:
    parser = argparse.ArgumentParser(description="輸出今日 Round 6 Batch 3c 選股訊號")
    parser.add_argument("--config", required=True, help="YAML config 路徑")
    parser.add_argument("--xgb_model", required=True,
                        help="XGBoost booster json 路徑（需有同名 .summary.json 記錄 feature_columns）")
    parser.add_argument("--model", default="pretrained",
                        help="Kronos embedding backbone，須與訓練該 xgb_model 時一致（Batch 3c 全程用 pretrained，"
                             "即未微調的 NeoQuasar/Kronos-base）")
    parser.add_argument("--date", default=None,
                        help="指定交易日（YYYY-MM-DD），預設為 DB 最新日期")
    parser.add_argument("--top_k", type=int, default=10, help="持股數")
    parser.add_argument("--holdings", default="",
                        help="目前已持有的股票代碼，逗號分隔（用於顯示換股建議）")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    feature_columns = _require_model_feature_columns(args.xgb_model)

    if args.date:
        rebal_date = pd.Timestamp(args.date)
    else:
        latest = _last_trading_day(cfg.db_path, cfg.benchmark_symbol)
        rebal_date = pd.Timestamp(latest)

    print("\n=== Round 6 Batch 3c 選股訊號（Kronos embedding + XGBoost）===")
    print(f"  embedding backbone：{args.model}  |  xgb_model：{args.xgb_model}")
    print(f"  特徵數：{len(feature_columns)}  |  top_k={args.top_k}")
    print(f"  訊號日：{rebal_date.date()}")
    print()

    specs = build_model_specs(cfg)
    if args.model not in specs:
        print(f"未知模型 key: {args.model}，可用: {list(specs)}")
        sys.exit(1)
    predictor = load_predictor_from_spec(specs[args.model], cfg)

    booster = xgb.Booster()
    booster.load_model(args.xgb_model)

    all_symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]

    signals_by_date = compute_xgb_signals(
        predictor, booster, cfg, pd.DatetimeIndex([rebal_date]), all_symbols,
        feature_columns=feature_columns,
    )
    signals = signals_by_date.get(rebal_date.strftime("%Y-%m-%d"), {})

    if not signals:
        print("警告：沒有取得任何訊號，請確認 DB 資料已更新至今日。")
        sys.exit(1)

    top_set = rank_stocks(signals, top_k=args.top_k, threshold=cfg.min_signal_threshold)
    ranked = sorted(top_set, key=lambda s: signals[s], reverse=True)

    print(f"\n【選股結果】XGBoost 分數 top {args.top_k}（{rebal_date.date()} 訊號）")
    print(f"{'排名':>4}  {'代碼':>8}  {'分數':>10}")
    print("-" * 28)
    for rank, sym in enumerate(ranked, 1):
        print(f"  {rank:>2}   {sym:>8}   {signals[sym]:>+10.4f}")

    if args.holdings:
        current = set(args.holdings.split(","))
        to_sell = current - top_set
        to_buy = top_set - current
        hold = current & top_set
        print(f"\n【換股建議】（目前持倉: {sorted(current)}）")
        if hold:
            print(f"  繼續持有：{sorted(hold)}")
        if to_sell:
            print(f"  賣出：    {sorted(to_sell)}")
        if to_buy:
            print(f"  買入：    {sorted(to_buy)}")
        if not to_sell and not to_buy:
            print("  持倉無需調整。")

    print()


if __name__ == "__main__":
    main()
