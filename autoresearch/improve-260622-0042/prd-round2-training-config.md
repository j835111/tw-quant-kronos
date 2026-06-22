# PRD: Round 2 訓練起點 + 超參數配置

> Auto-generated from research findings. DECISION NEEDED items require your judgment.

---

## 問題陳述

Round 1 有三個致命設計缺陷：
1. **起點錯誤**：從 pretrained Kronos-base 出發，tokenizer/predictor 不一致，6 epoch 不足以重學台股分佈
2. **LR 太保守**：1e-5 在 6 epoch 內無法從 pretrained 出發學到顯著的台股排名信號
3. **訓練太短**：6 epoch 遠不夠（Round 0 跑了 20 epoch 才收斂）

Round 2 的目標是：在 Round 0 的成果上「精煉」（更好的 early stopping），而非從頭學習。

---

## 使用者故事

1. 作為研究員，我需要 Round 2 訓練從 Round 0 best predictor 出發，以保留已學到的台股動量效應。
2. 作為訓練腳本，我需要足夠的 epoch + 適當的 LR schedule 讓模型在正確方向微調，不過擬合也不欠擬合。
3. 作為 MoLab 使用者，我需要訓練可以在 sandbox 重啟後自動 resume（透過 HF Hub）。

---

## 需求

### config_tw_daily_rtx6000.yaml 改動

```yaml
# ── 起點 ──────────────────────────────
pretrained_predictor: "j835111/kronos-tw-finetune"
hf_revision: "round-0"                    # DECISION NEEDED: 確認 branch name
# hf_subfolder 需要在 resolve_src 中處理，見下方

# ── 訓練長度 ──────────────────────────
basemodel_epochs: 20                      # Round 1: 6 → Round 2: 20
train_steps_per_epoch: 1000               # 維持不變

# ── Learning Rate ──────────────────────
predictor_lr: 0.00005                     # Round 1: 1e-5 → Round 2: 5e-5

# ── Early Stopping ──────────────────────
early_stop_patience: 3                    # Round 1: 2 → Round 2: 3
ic_val_symbols: 300                       # Round 1: 150 → Round 2: 300
ic_val_dates: 20                          # Round 1: 8  → Round 2: 20

# ── HF 備份 ──────────────────────────
hf_repo: "j835111/kronos-tw-finetune"
hf_revision_out: "round-2"               # 輸出到新 branch
```

### `train_predictor.py` LR Scheduler 調整

**現況：** `OneCycleLR(pct_start=0.03, div_factor=10)`
- warmup 佔 3% steps（20 epoch × 1000 steps = 600 steps warmup）
- initial_lr = 5e-5 / 10 = 5e-6

**建議：**
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=cfg.predictor_lr,   # 5e-5
    steps_per_epoch=train_steps_per_epoch,
    epochs=cfg.basemodel_epochs,
    pct_start=0.05,       # 5% warmup = 1000 steps（20 epoch × 1000）
    div_factor=25,        # initial_lr = 2e-6（溫柔起步）
    final_div_factor=1e4, # final_lr = 5e-9（充分收斂）
    anneal_strategy='cos',
)
```

### `hf_utils.py` / `resolve_src` 處理 round-0 起點

```python
# 新增 config 欄位
hf_repo: str = ""
hf_revision: str = ""        # 作為 from_pretrained 的 revision 參數
hf_revision_out: str = ""    # 訓練完 push 的目標 revision

# resolve_src 邏輯：若本地 path 不存在，且 hf_repo+hf_revision 指定了，
# 則從 HF 下載 predictor/best_model subdir
```

---

## 技術方案

### Round 0 predictor 路徑確認

```bash
# 確認 HF repo 上有正確的 round-0 predictor
hf ls j835111/kronos-tw-finetune --revision round-0
# 預期看到 predictor/best_model/config.json, model.safetensors 等
```

若 round-0 branch 上的 predictor/best_model 存在，只需 config 改 `pretrained_predictor + hf_revision` 即可。

### Config 新增欄位（config.py）

```python
@dataclass
class Config:
    ...
    # HF 版控
    hf_repo: str = ""
    hf_revision: str = ""       # 讀取起點 revision
    hf_revision_out: str = ""   # push 目標 revision
```

---

## 驗收條件

- [ ] `config_tw_daily_rtx6000.yaml` 更新，`pretrained_predictor` 指向 round-0
- [ ] 訓練啟動後 epoch 1 的 val_ic / ic_ir_h5 不為全負（代表起點有效）
- [ ] `train_log.csv` 有 20 行記錄（或 early stop 在中間）
- [ ] best checkpoint 由 IC-IR@h5 選出，並自動 push 到 `j835111/kronos-tw-finetune@round-2`
- [ ] Round 2 回測 Sharpe > 1.19（超越 Round 0）

---

## 風險

| 風險 | 機率 | 緩解 |
|------|------|------|
| round-0 HF branch 的 predictor 不完整/遺失 | 中 | 先 `hf ls` 確認；若遺失需重跑 Round 0 |
| lr=5e-5 對 Round 0 base 過大，破壞已學特徵 | 中 | 監控 epoch 1 loss；若 ic_ir_h5 暴跌立即停止；可降到 2e-5 |
| 20 epoch 在 RTX Pro 6000 花 10 小時+ | 低 | MoLab sandbox 限制通常 > 12 小時；HF resume 保護 |

---

## DECISION NEEDED

1. **Round 0 predictor 存放位置**：HF `j835111/kronos-tw-finetune@round-0/predictor/best_model` 是否存在？需先 `hf ls` 確認。若不存在，需決定是否先重跑 Round 0（成本：~6 小時）。
2. **LR 大小**：5e-5 是建議值。若 Round 0 predictor 已高度適應台股，可以保守用 2e-5。`DECISION: 建議先用 2e-5，若 5 epoch 後 ic_ir_h5 < round-0 ic_ir_h5，升到 5e-5 重試。`
3. **hf_revision_out 命名**：用 `round-2` 還是 `round-1-retry`？建議 `round-2`（清晰語義）。

## Open Questions

- 是否需要先跑一個 dry-run（max_steps=100）驗證起點 predictor 能正常 load？
- Round 2 train_end_date 是否維持 2023-12-31？（建議：是，保持 val period 一致，方便對比）
