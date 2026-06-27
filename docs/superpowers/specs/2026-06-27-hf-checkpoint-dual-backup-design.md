# Design — finetune_tw 雙軌 Checkpoint 備援（`/mnt/first` + HF）

**日期：** 2026-06-27
**狀態：** 已通過設計審查，待使用者 review spec
**前置診斷：** MoLab sandbox 內 `/marimo/Kronos` 多次只剩目錄骨架；`finetune_tw/outputs/`、`checkpoints/`、`.git` metadata 皆可能部分或全部遺失

## 背景與問題

目前 `finetune_tw` 的訓練恢復策略依賴兩種狀態來源：

- 本地 workspace：`/marimo/Kronos/finetune_tw/outputs/...`
- 可選 rclone：`gdrive:Kronos/outputs/.../checkpoints`

這個設計在 MoLab 上不夠穩定。實際觀測到的失敗模式如下：

1. **sandbox 沒 terminated，但 `/marimo/Kronos` 只剩骨架目錄**
   - `finetune_tw/outputs/tw_daily/tokenizer/best_model/` 變成空目錄
   - `finetune_tw/outputs/tw_daily/predictor/best_model/` 變成空目錄
   - `finetune_tw/outputs/tw_daily/predictor/checkpoints/` 變成空目錄
   - `finetune_tw/outputs/tw_daily/token_cache/` 變成空目錄
   - `train_log.csv`、監控 log、pid files 消失
2. **`.git` 目錄殘缺**
   - `git status` / `git rev-parse` 失敗
   - `.git` 只剩少量子目錄，不再是有效 repo
3. **目前 HF 只保存 `best_model`**
   - `predictor/checkpoints/ckpt-*.pt` 沒有上傳到 HF
   - sandbox 換掉或本地 state 清空後，無法從 HF resume

根因不是單一程式錯誤，而是**目前把可恢復訓練狀態與 disposable code checkout 綁在同一棵 `/marimo/Kronos` 目錄**。當 MoLab 對 workspace 做重置或半重置時，code 與 state 一起受損。

## 目標

把 `finetune_tw` 訓練流程改成雙軌備援：

- **主來源：** `/mnt/first/kronos_state` 作為本地持久化 state
- **備援來源：** Hugging Face Hub checkpoint branch

成功條件：

1. sandbox 重啟但 `/mnt/first` 未換卷時，可直接從本地 checkpoint resume
2. `/mnt/first` 狀態缺失時，可從 HF branch 拉回最新 checkpoint 再 resume
3. HF 上傳失敗時，訓練**不中斷**，僅記錄 warning
4. `/marimo/Kronos` 即使被清空，只需重 clone code，不影響訓練 state 恢復
5. sandbox / session 重開後，可透過單一腳本重建 code checkout、接回 state、並重新啟動訓練與監控

## 設計原則

1. **Code 與 state 分離**
   - `/marimo/Kronos` 只放程式碼 checkout
   - 所有可恢復訓練資產移到 `/mnt/first/kronos_state`
2. **本地優先，遠端備援**
   - 正常 resume 先讀本地
   - 只有本地缺失時才打 HF
3. **最小遠端保留集合**
   - HF 只保留最新 3 個 `ckpt-*.pt`
   - 避免 checkpoint branch 無限制膨脹
4. **上傳失敗不影響訓練**
   - HF 只是 backup，不成為單點失敗
5. **恢復操作可一鍵執行**
   - 不把人工多步驟 shell 操作當成正式恢復流程
   - 需要有可重複、可審查的 bootstrap script

## 目錄佈局

### Code checkout

- `/marimo/Kronos`

用途：

- clone repo
- 啟動訓練程式
- 每次 sandbox 壞掉時可整棵重建

### 持久化 state

- `/mnt/first/kronos_state/data/tw_stocks.db`
- `/mnt/first/kronos_state/outputs/tw_daily/tokenizer/best_model/`
- `/mnt/first/kronos_state/outputs/tw_daily/tokenizer/checkpoints/`
- `/mnt/first/kronos_state/outputs/tw_daily/predictor/best_model/`
- `/mnt/first/kronos_state/outputs/tw_daily/predictor/checkpoints/`
- `/mnt/first/kronos_state/outputs/tw_daily/token_cache/`
- `/mnt/first/kronos_state/outputs/tw_daily/predictor/train_log.csv`

`Config` 層面不再使用 `/marimo/Kronos/finetune_tw/outputs` 作為 `output_dir`。

## HF 佈局

沿用現有 model repo：

- `hf_repo = "j835111/kronos-tw-finetune"`

區分兩種用途的 revision：

### Inference / best model revision

- `hf_revision_out = "round-3"`

保存內容：

- `predictor/best_model/`
- `predictor/train_log.csv`
- （若已有既存流程）tokenizer best model

### Checkpoint backup revision

- `hf_checkpoint_revision_out = "checkpoints-round-3"`

保存內容：

- `predictor/checkpoints/ckpt-*.pt`
- `tokenizer/checkpoints/ckpt-*.pt`（若 tokenizer 訓練也要同規則保護）

這樣做的理由：

- 不把恢復用 checkpoint 和推理用 `best_model` 混在同一 revision
- 不污染既有 round-based model loading 流程
- restore 路徑可明確寫死為 `checkpoints-round-3`

## 恢復流程

### Predictor

`train_predictor.py` 的 checkpoint 恢復順序改成：

1. **本地 checkpoint**
   - 檢查 `/mnt/first/kronos_state/outputs/tw_daily/predictor/checkpoints/ckpt-*.pt`
   - 有檔案：直接 `_load_latest_checkpoint(...)`
2. **HF checkpoint branch**
   - 本地沒有 checkpoint 時，從
     `cfg.hf_repo@cfg.hf_checkpoint_revision_out/predictor/checkpoints/`
     下載 checkpoint 到本地 checkpoint 目錄
   - 下載後再 `_load_latest_checkpoint(...)`
3. **都沒有**
   - 從 `cfg.hf_revision` 指向的 `predictor/best_model` 起訓

### Tokenizer

`train_tokenizer.py` 也採同樣模式：

1. 先讀本地 `/mnt/first/.../tokenizer/checkpoints`
2. 本地沒有時，從 HF `tokenizer/checkpoints/` 拉回
3. 都沒有時，從 `cfg.pretrained_tokenizer` 起訓

## 一鍵恢復腳本

除了訓練程式本身的 restore 邏輯，還需要一個**session / sandbox 重開後的 bootstrap script**。沒有這個元件，恢復流程仍然依賴人工重新 clone、手動檢查 config、手動啟訓，實務上仍然脆弱。

### 腳本位置

- `scripts/resume_molab_training.sh`

### 目標

在新的或被半重置的 MoLab sandbox 中，透過一條命令完成：

1. 重建 `/marimo/Kronos` code checkout
2. 驗證 `/mnt/first/kronos_state` 是否存在
3. 驗證 config 指向 `/mnt/first/kronos_state`
4. 本地找 checkpoint，必要時 fallback HF
5. 啟動或 resume tokenizer / predictor 訓練
6. 重啟訓練監控腳本

### 腳本介面

最小介面：

- `scripts/resume_molab_training.sh --config finetune_tw/configs/config_tw_daily_rtx6000.yaml --stage predictor`

可選參數：

- `--stage tokenizer|predictor`
- `--repo-url <git remote>`
- `--repo-dir /marimo/Kronos`
- `--state-dir /mnt/first/kronos_state`
- `--branch <git branch>`

預設行為：

- `repo-dir=/marimo/Kronos`
- `state-dir=/mnt/first/kronos_state`
- `stage=predictor`

### 腳本職責

#### 1. Code checkout bootstrap

- 若 `/marimo/Kronos` 不存在：重新 clone
- 若 `/marimo/Kronos` 存在但不是有效 git repo：刪除後重 clone
- 若存在有效 repo：切到指定 branch，必要時 `fetch` / `reset --hard` 到指定 revision

注意：因為 `/marimo/Kronos` 被明確視為 disposable code checkout，所以腳本可以主動重建這棵目錄；這不適用於 `/mnt/first/kronos_state`

#### 2. State preflight

- 檢查 `/mnt/first/kronos_state` 是否存在
- 若不存在：
  - 建立目錄骨架
  - 記錄「本地 state 缺失，將依賴 HF restore 或重新訓練」
- 檢查 `db_path` 與 `output_dir` 是否落在 `state-dir` 之下
- 若 config 仍指向 `/marimo/Kronos/...`，直接報錯退出，不啟動訓練

#### 3. Restore preflight

- tokenizer stage：
  - 檢查本地 tokenizer checkpoint
  - 若無，本地 tokenizer best_model 是否存在
  - 若仍無，之後交由程式內 HF restore / pretrained 路徑處理
- predictor stage：
  - 檢查本地 predictor checkpoint
  - 若無，再檢查 tokenizer best_model 是否存在
  - 若 tokenizer best_model 也無，優先嘗試從 HF restore tokenizer，再啟動 predictor

#### 4. Launch

- 使用 `python -m finetune_tw.train_tokenizer --config ...` 或
  `python -m finetune_tw.train_predictor --config ...`
- 以 background process 啟動，stdout / stderr 導到 state-dir 下的 log
- 將 pid 寫到 `state-dir` 對應位置，而不是寫在 `/marimo/Kronos`

#### 5. Monitoring

- 腳本需一併重啟 monitor process
- monitor 讀取 `train_log.csv`、`week23_train_stdout.log`、`checkpoints/`
- monitor 輸出也寫到 `state-dir`

### 腳本失敗策略

- code clone 失敗：直接 exit，因為無法啟動訓練
- config 指錯路徑：直接 exit，因為這是危險配置
- 本地 state 缺失：允許繼續，但須明確記 log
- HF restore 失敗：允許繼續，交由訓練程式決定是否從較早狀態或 pretrained 起跑
- monitor 啟動失敗：訓練可繼續，但需回傳非零或明確 warning，避免誤以為監控存在

## 保存流程

### Predictor checkpoint

每次 `global_step % cfg.save_steps == 0`：

1. 存本地 checkpoint 到 `/mnt/first/.../predictor/checkpoints/ckpt-{step}.pt`
2. 背景上傳這個 checkpoint 到 HF `checkpoints-round-3`
3. 上傳完成後，清理遠端舊檔，只保留最新 3 個

### Tokenizer checkpoint

同樣在 tokenizer 的 `save_steps` 路徑上做：

1. 本地保存
2. 背景上傳到 HF `checkpoints-round-3`
3. 遠端只保留最新 3 個

### Best model

`best_model` 與 `train_log.csv` 仍維持現有語意：

- 只在 `is_best` 為真時推到 `round-3`
- 不與 rolling checkpoints 混在一起

## 失敗處理

HF 相關操作的錯誤策略：

- upload 失敗：`print("[hf] checkpoint push failed: ...")`，訓練繼續
- remote prune 失敗：記 log，訓練繼續
- restore 失敗：記 log，若本地也沒有 checkpoint，回退到原始啟動路徑

明確不做：

- 不因 HF 備份失敗而 `raise`
- 不讓訓練主迴圈阻塞等待遠端同步完成

## Config 變更

`finetune_tw/config.py` 新增：

- `hf_checkpoint_revision_out: str = ""`
- `hf_checkpoint_keep_last_n: int = 3`

MoLab 專用 config（如 `config_tw_daily_rtx6000.yaml`）改成：

- `db_path: "/mnt/first/kronos_state/data/tw_stocks.db"`
- `output_dir: "/mnt/first/kronos_state/outputs"`
- `hf_checkpoint_revision_out: "checkpoints-round-3"`
- `hf_checkpoint_keep_last_n: 3`

## 程式元件

### `finetune_tw/hf_utils.py`

新增 helper：

- `push_checkpoint(local_path, repo_id, path_in_repo, revision)`
- `restore_checkpoints(local_dir, repo_id, subfolder, revision)`
- `prune_checkpoints(repo_id, subfolder, revision, keep_last_n)`

要求：

- API 設計與既有 `push_best_model` / `push_file` 風格一致
- thread-based 背景上傳可沿用
- prune 以 `ckpt-<step>.pt` 的 step 數字排序

### `finetune_tw/train_predictor.py`

修改點：

- `_gdrive_restore_checkpoints(...)` 之後，新增 HF restore fallback
- 每次 `_save_checkpoint(...)` 後，新增 HF push + remote prune
- `remote_root` 的語意保留給 rclone，不與 HF checkpoint revision 混用

### `finetune_tw/train_tokenizer.py`

修改點：

- 與 predictor 對齊
- 不再讓 tokenizer 與 predictor 的恢復策略分叉

### `scripts/resume_molab_training.sh`

新增腳本：

- 重建 `/marimo/Kronos`
- 驗證 state-dir 與 config
- 啟動 tokenizer / predictor
- 重啟 monitor

要求：

- 可重複執行（idempotent）
- 對「repo 消失但 state 仍在」情況穩定
- 不把 state 寫回 `/marimo/Kronos`
- log 輸出應足夠支持人工除錯

## 驗證策略

### 單元測試

新增 / 擴充測試以覆蓋：

1. **本地優先**
   - 本地有 checkpoint 時，不觸發 HF restore
2. **HF fallback**
   - 本地空時，呼叫 HF restore，再載入最新 checkpoint
3. **遠端 prune**
   - 給定多個 `ckpt-*.pt`，只保留最新 3 個
4. **HF 失敗不中斷**
   - upload / prune 拋錯時只記錄，不中止訓練
5. **tokenizer / predictor 一致**
   - 兩條訓練路徑都會走同樣的 HF checkpoint 備援
6. **bootstrap script**
   - repo 不存在、repo 損壞、state 存在、state 缺失等情況下，腳本都做出可預期行為

### 手動驗證

在 MoLab 上做兩種恢復演練：

1. **本地恢復**
   - 人工中斷後重新啟動同一 sandbox
   - 確認從 `/mnt/first/.../checkpoints` 續跑
2. **遠端恢復**
   - 暫時移走本地 checkpoint 目錄
   - 確認程式能從 HF `checkpoints-round-3` 拉回最新 checkpoint
3. **一鍵恢復**
   - 手動刪除 `/marimo/Kronos`
   - 保留 `/mnt/first/kronos_state`
   - 執行 `scripts/resume_molab_training.sh`
   - 確認 code checkout 被重建、訓練恢復、monitor 重啟

## 不在範圍（YAGNI）

- 不在這一輪改動訓練策略、loss、backtest 邏輯
- 不在這一輪引入新的雲端儲存（S3 / GCS / HF buckets）
- 不在這一輪保留 `/marimo/Kronos` 的任何訓練狀態；該目錄明確視為 disposable

## 風險

1. **HF checkpoint 體積大**
   - `ckpt-*.pt` 約 1GB 級別，上傳時間不短
   - 因此採背景上傳 + 只保留最新 3 個
2. **branch 汙染**
   - 若 checkpoint 與 best_model 混 branch，後續 restore / inference 會變複雜
   - 本設計用獨立 revision 避免此問題
3. **本地持久化卷語意不完全透明**
   - `/mnt/first` 對「session 死掉但 sandbox 沒換」較穩，但不是跨所有 MoLab 生命週期都保證
   - 所以仍需要 HF 備援

## 產出物

- `finetune_tw/config.py`
- `finetune_tw/hf_utils.py`
- `finetune_tw/train_predictor.py`
- `finetune_tw/train_tokenizer.py`
- `finetune_tw/configs/config_tw_daily_rtx6000.yaml`
- `scripts/resume_molab_training.sh`
- 對應單元測試檔
