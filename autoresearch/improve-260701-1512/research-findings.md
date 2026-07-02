# Research Findings — Kronos TW Round 6 方向
> 研究日期：2026-07-01 | 迭代次數：10 | 策略：已窮盡五輪 fine-tuning，尋找根本性不同方向

---

## 研究背景

**現況**：Round 0 是唯一可執行版本（open/open Sharpe 1.12，新基準）。Round 1-5 全部退步。  
**核心問題**：
1. 模型預測力極弱（IC~0.04，方向~50%）
2. fine-tuning 使 Kronos 退化（val_loss 從 2.99 → 3.64 → 持續上升）
3. val 集 IC-IR 指標不能預測回測 Sharpe

---

## Insight #1 — Fine-tuning TSFM 在金融領域系統性失敗

**來源**：arXiv:2511.18578（Re(Visiting) Time Series Foundation Models in Finance，2025-11）  
**核心發現**：
- 以超過 20 億筆全球股市觀測資料進行測試，結果明確：**"Off-the-shelf pre-trained TSFMs perform poorly in zero-shot AND fine-tuning settings"**
- 唯一有效的是**從頭開始在金融資料上訓練的模型（domain-specific pre-training from scratch）**
- 這直接解釋了 Round 1-5 為何全部失敗：fine-tuning pretrained TSFM 對金融交叉截面預測本質上效果有限

**分類**：ICP challenges — 根本原因診斷  
**信心**：HIGH（2 億筆實證 + 論文明確結論）

---

## Insight #2 — LambdaRankIC：低 SNR 環境的排名損失

**來源**：arXiv:2605.00501（LambdaRankIC: Directly Optimizing Rank IC for Financial Prediction，2026-05）  
**核心發現**：
- 現有模型用 regression loss 或 NDCG-oriented ranking，**與 Rank IC 不對齊**
- LambdaRankIC 導出 pairwise rank swap 的 closed-form lambda gradients，在 XGBoost 中直接優化 Rank IC
- 在**低 SNR 與 heavy-tail noise 環境下**（正是我們的情況），一致優於 regression 與 NDCG 方法
- 結果：out-of-sample Rank IC、ICIR、monthly return、Sharpe 均最佳

**關鍵機制**：不是改進神經網路本身，而是用**正確的目標函數**（Rank IC）訓練一個 XGBoost，以 Kronos 嵌入向量作為特徵。

**分類**：Competitor gaps — 技術差異化  
**信心**：HIGH（模擬研究 + 真實市場驗證）

---

## Insight #3 — Kronos Embedding 作為特徵，而非直接預測

**來源**：  
- Stock2Vec 雙階段架構（embedding layer → tree-based model）  
- Deep Embedding Forest 模式（embeddings → XGBoost/LightGBM）  
- arXiv:2509.23695（Estimating TSFM Transferability via In-Context Learning）

**核心發現**：
- 把凍結的 Kronos-base 當**特徵抽取器**，用最後一層 hidden state 作為 XGBoost/LightGBM 輸入
- 完全迴避 catastrophic forgetting 問題（Kronos 不更新）
- Kronos 的 token space IC~0.04 弱，但 hidden state 的**高維表達**可能含有更豐富的截面信息，tree-based model 更善於提取非線性特徵組合
- 兩階段架構在文獻中有多個成功案例

**分類**：UX & experience — 改變推理架構  
**信心**：MEDIUM（間接證據，我們的情境未直接驗證）

---

## Insight #4 — MoFO：無需重放資料的防遺忘 Fine-tuning

**來源**：arXiv:2407.20999（MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning）  
**核心發現**：
- 每步只更新**動量幅度最大**的 top-K% 參數，其他參數凍結
- 結果：fine-tuned 效果與全參數訓練相當，但與 pretrained 更接近 → 遺忘更少
- **不需要 pretrained 訓練資料，不需要 EWC 的 Fisher 估計**
- 可直接套用至我們的 `train_predictor.py`，只需替換 optimizer 為 MoFO

**對照我們的問題**：Round 1-5 val_loss 單調上升 = catastrophic forgetting。MoFO 直接解決。  
**分類**：ICP challenges — 技術障礙消除  
**信心**：MEDIUM（LLM 驗證，TSFM 未直接驗證）

---

## Insight #5 — L2-SP：最簡單的防遺忘正則化

**來源**：multiple EWC/regularization papers（arXiv:2603.18596）  
**核心發現**：
- 在 fine-tuning loss 中加入 `λ * ||θ - θ_pretrained||²`
- 懲罰任何參數偏離 pretrained 的程度
- 比 EWC 更簡單（不需 Fisher matrix），比 LoRA 更直接
- 對每個參數施加等強度懲罰，保護整個 pretrained landscape

**對照我們的問題**：Kronos-base pretrained 的 val_loss=2.99 是 token prediction 的上限，fine-tuning 把它推高到 3.64。L2-SP 可限制偏離程度。  
**分類**：ICP challenges  
**信心**：MEDIUM

---

## Insight #6 — SSPT：股票特化預訓練任務（KDD 2025）

**來源**：arXiv:2506.16746（Pre-training Time Series Models with Stock Data Customization，KDD 2025）  
**核心發現**：
三個自監督預訓練任務（只需日線價格資料 + 基本公司資訊）：
1. **股票代碼分類**：預測哪一支股票
2. **產業分類**：預測哪個產業
3. **移動平均預測**：預測 MA 值

在五個股票資料集（四個市場）上一致優於現有方法，Sharpe Ratio 提升。  
**應用方式**：在 Kronos-base 的基礎上，先用台股資料做這三個自監督任務進行**持續預訓練**，再 fine-tune predictor → 先讓模型了解台股結構，再學預測。

**分類**：UX & experience — 預訓練改良  
**信心**：MEDIUM（不同架構，但台股有 1091 支股票可做分類任務）

---

## Insight #7 — "The Finetuner's Fallacy"：先在目標資料預訓練

**來源**：arXiv:2603.16177（The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data）  
**核心發現**：
- 先用微調資料持續預訓練（繼續做 next-token prediction），再 fine-tune，往往優於直接 fine-tune
- 讓模型先熟悉資料分佈，再學任務目標
- **對我們的情境**：先在台股 OHLCV 序列上做 masked/autoregressive 預訓練（使用台股 tokenizer），再做 predictor fine-tuning

**分類**：ICP challenges  
**信心**：LOW（direct evidence 不足，但理論支持）

---

## Insight #8 — Pretrained TSFM 有隱含市場狀態概念

**來源**：arXiv:2509.05801（time2time: Causal Intervention in Hidden States）、arXiv:2511.15324（On the Internal Semantics of TSFMs）  
**核心發現**：
- 早期 layers 編碼局部時間域結構（AR, trend, level shifts）
- 深層 layers 編碼 dispersion 和 change-point 信號
- **Hidden state 可以被直接操作**來影響預測（activation transplantation）
- 這意味著 Kronos 的中間層可能有台股動量/回歸的資訊，只是 autoregressive head 無法直接轉化

**分類**：UX & experience  
**信心**：LOW（研究性，未驗證於台股）

---

## 關鍵新方向總結

### 立即可行（工程難度 LOW-MEDIUM）
| 方向 | 核心機制 | 預期效果 | 信心 |
|------|---------|---------|------|
| **Kronos Embedding + LambdaRankIC (XGBoost)** | 凍結 Kronos → 提取 hidden states → XGBoost+LambdaRankIC | 繞過 fine-tuning，直接優化 Rank IC | HIGH |
| **L2-SP + Pretrained 重啟** | 損失加 λ‖θ-θ₀‖²，限制偏離 pretrained | 防止 val_loss 上升，Sharpe 保持 | MEDIUM |
| **MoFO optimizer** | 只更新動量最大參數，其他凍結 | 類似 FPT 但更靈活，防遺忘 | MEDIUM |
| **Close-to-close IC-IR@h1 + pretrained restart（無 ranking loss）** | 上輪缺口：Round 5 用 ranking loss，沒有只試此組合 | 可能突破 Round 0 | MEDIUM |

### 中期研究（工程難度 MEDIUM-HIGH）
| 方向 | 核心機制 | 預期效果 | 信心 |
|------|---------|---------|------|
| **SSPT 持續預訓練** | 股票分類 + 產業分類 + MA 預測（台股資料）| 改善 Kronos 對台股的 domain adaptation | MEDIUM |
| **Continual pretrain 再 fine-tune** | 先在台股 token 序列繼續做 next-token prediction，再 fine-tune | 先讓分佈對齊，再學任務 | LOW-MEDIUM |
