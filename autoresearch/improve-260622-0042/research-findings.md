# Research Findings — finetune_tw 改善計畫
**日期:** 2026-06-22 | **研究輪次:** 15 | **類別覆蓋:** 5/5

---

## 類別 1：ICP 挑戰（訓練目標脫鉤問題）

### F1-1 [HIGH] IC-Backtest 根本脫鉤
- **問題：** Cross-Entropy token loss 訓練 ≠ top-K portfolio 排名優化。CE loss 讓模型最小化 token-level 預測誤差，而回測需要的是截面 *相對排名* 的一致性。
- **機制：** Round 0 val_loss 3.644 > Pretrained 2.997，但 Sharpe 1.19 >> 0.03。更低 CE loss = 更保守（接近 naive no-change），分化能力反而弱。
- **證據：** 本倉庫回測資料 + arXiv 2510.14156（S&P500 listwise ranking loss 改善 top-K backtest）
- **信心：** HIGH（3+ 來源）

### F1-2 [HIGH] IC 估計噪音壓制 Early Stopping 信號
- **問題：** 現況 ic_val_symbols=150, ic_val_dates=8 → 1200 個樣本，σ(IC)≈0.08，遠大於 IC 訊號本身（0.00~0.05），信噪比 < 1。
- **機制：** Early stopper 在噪音海中選「最不壞」而非「真正好的」checkpoint。
- **解法：** 增加到 300 × 20 = 6000 樣本，σ(IC) 理論降至 ~0.035，SNR 提升 2.3×。
- **信心：** HIGH（統計推導 + Fundamental Law 文獻支持）

### F1-3 [HIGH] 訓練起點 tokenizer/predictor 不一致
- **問題：** Round 1 從 pretrained Kronos-base 出發，但沿用 Round 0 fine-tuned tokenizer。pretrained predictor 未適應 Round 0 tokenizer 學到的台股 token 分佈。
- **機制：** Tokenizer 量化碼簿（codebook）在 Round 0 已針對台股 OHLCV 調整；predictor 從 pretrained 出發等於從未見過這個碼簿的分佈。
- **解法：** Round 2 從 `j835111/kronos-tw-finetune@round-0` predictor 出發。
- **信心：** HIGH（代碼分析 + 實驗資料）

---

## 類別 2：競品差距（損失函數設計）

### F2-1 [HIGH] Ranking Loss 顯著優於 Pointwise CE 用於 top-K 策略
- **問題：** CE loss 是 pointwise，不直接優化 top-K 排名。
- **機制：** arXiv 2510.14156（CIKM 2025）系統比較 pointwise / pairwise / listwise loss 於 S&P500 日線，發現 listwise 在 IC、ICIR、portfolio returns 均勝出。LambdaRank/LambdaMART 對截面 momentum 策略 Sharpe ratio 有顯著提升。
- **解法選項：**
  - A. 加入 pairwise ranking loss 作為輔助 loss（0.8×CE + 0.2×pairwise）—風險低
  - B. 完全替換為 IC-weighted listwise loss —潛力高、風險高
- **信心：** HIGH（同行評審論文）

### F2-2 [MEDIUM] Label Horizon Paradox：最佳監督 horizon ≠ 推理 horizon
- **問題：** 當前 train: 全 10-horizon CE loss；backtest 用 h5。
- **機制：** arXiv 2602.03395 發現在金融預測中，最佳訓練 label 往往是「中間 horizon」，因為近 horizon 信噪比低（noise），遠 horizon signal 也衰減。h3 可能比 h5 有更高信噪比作為訓練目標。
- **解法：** 嘗試訓練時對 h1~h5 的 loss 加非線性權重（bell curve on h3），或 bi-level optimization 自動搜尋最佳 horizon。
- **信心：** MEDIUM（新論文 2026-02，實證於 S&P500，未在 TWSE 驗證）

### F2-3 [MEDIUM] Kronos 論文未針對 IC-IR 優化（文獻缺口）
- **問題：** Kronos 預訓練目標是 token prediction accuracy，非截面排名。
- **機制：** Financial Fine-tuning 論文（arXiv 2412.09880）指出 one-shot pretrained 和 SFT 之間有顯著差距，SFT 需要 task-specific signal，不只是 CE loss。
- **解法：** 加入 IC-aware fine-tuning 目標層（auxiliary head）。
- **信心：** MEDIUM（arXiv 2412.09880 + 本實驗觀察）

---

## 類別 3：市場趨勢

### F3-1 [HIGH] Warmup + Cosine Decay 是 fine-tuning 黃金標準
- **問題：** 現況 OneCycleLR 在 6 epoch 內可能過早 decay。
- **機制：** Financial Fine-tuning 論文和通用 LLM fine-tuning 文獻一致建議：linear warmup (5% steps) → cosine decay，rewarming 後 decay 必要。固定 peak LR 是最重要超參數。
- **解法：** lr=1e-5 → 5e-5 (warmup 3% steps) → cosine decay to 1e-6，15~20 epochs。
- **信心：** HIGH（多篇同行評審文獻）

### F3-2 [MEDIUM] 台灣市場短期動量弱，5天 IC 衰減與 Kronos 特性吻合
- **問題：** 本實驗 h1→h5 IC 衰減 46%（pretrained），但 Round 0 只衰減 23%，代表 fine-tuning 學到了台股特有動量維持性。
- **機制：** 台灣市場研究顯示短期（1-5天）IC 弱但一致；動量效應在 5-20 天最明顯。Round 0 可能在此 range 有優勢。
- **信心：** MEDIUM（台灣學術文獻 + 本實驗資料）

---

## 類別 4：UX & 驗證體驗

### F4-1 [HIGH] IC-IR@h5 比 val_ic 更適合 Early Stopping
- **問題：** val_ic 平均值對小樣本下噪音敏感；IC-IR = IC/σ(IC) 衡量一致性。
- **機制：** ICIR 是截面預測能力的信噪比指標。優先選信號穩定（方差小）的 checkpoint，而非單次 IC 最高。Round 0 IC-IR@h1=0.625 > Pretrained 0.601，正 IC 日比例也高（72.8% vs 70.9%）。
- **解法：** `ic_ir_h5` 作為 EarlyStopper 指標（需在 ic_validation.py 新增計算）。
- **信心：** HIGH（本實驗資料充分支持）

### F4-2 [MEDIUM] train_log.csv 遺失防護（基礎設施）
- **問題：** Round 1 train_log.csv 遺失，無法分析訓練歷程。
- **機制：** `_clear_stale` 在每次啟動時清空 outputs/，rclone 只備份 best_model。
- **解法：** HF push_best_model 同時 push train_log.csv；已有 `hf_utils.push_best_model()` 接口。
- **信心：** HIGH（代碼審查確認）

---

## 類別 5：收益與成長路徑

### F5-1 [HIGH] Round 0 是最接近目標的訓練起點，應作為 Round 2 基礎
- **問題：** Round 1 放棄 Round 0 優勢（台股動量學習），從 pretrained 重來導致退步。
- **機制：** Round 0 Sharpe 1.19 vs 基準 1.536，差距縮小到 22%。Round 1 完全退化（Sharpe 0.15）。
- **解法：** Round 2 = Round 0 predictor + 調整 IC-IR early stop + 更大驗證集。
- **信心：** HIGH（充分實驗證據）

### F5-2 [MEDIUM] 擴展評估指標多元化：不只 val_ic，也監控 MAPE_ratio、direction accuracy
- **問題：** MAPE_ratio 2.01× 是 Round 1 失敗的早期信號，但訓練時未監控。
- **機制：** 訓練時若 MAPE > 1.5× naive，代表模型越來越保守，應早期 flag。
- **信心：** MEDIUM（本實驗反推）
