from dataclasses import dataclass
import yaml


@dataclass
class Config:
    # Data
    db_path: str = "finetune_tw/data/tw_stocks.db"
    lookback_window: int = 90
    predict_window: int = 10
    max_context: int = 512
    clip: float = 5.0
    train_end_date: str = "2023-12-31"
    val_end_date: str = "2024-06-30"

    # Training
    tokenizer_epochs: int = 30
    basemodel_epochs: int = 20
    batch_size: int = 16
    save_steps: int = 500
    log_interval: int = 50
    tokenizer_lr: float = 2e-4
    predictor_lr: float = 4e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0.1
    num_workers: int = 2
    persistent_workers: bool = False
    prefetch_factor: int = 2
    train_steps_per_epoch: int = 0
    val_steps_per_epoch: int = 0
    amp_dtype: str = "bf16"
    enable_tf32: bool = True
    token_cache_enabled: bool = False
    token_cache_dtype: str = "uint16"
    seed: int = 42
    # Early stopping / price-space validation
    early_stop_patience: int = 2
    ic_val_symbols: int = 150
    ic_val_dates: int = 8
    val_ic_horizons: int = 5
    ranking_loss_alpha: float = 0.0
    ranking_loss_horizon: int = 5

    # Model paths
    pretrained_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    pretrained_predictor: str = "NeoQuasar/Kronos-base"
    exp_name: str = "tw_daily"
    output_dir: str = "finetune_tw/outputs"

    # HF Hub versioning (optional)
    hf_repo: str = ""           # e.g. "j835111/kronos-tw-finetune"
    hf_revision: str = ""       # revision to load pretrained_predictor from HF
    hf_revision_out: str = ""   # revision to push best_model to HF after training
    hf_checkpoint_revision_out: str = ""
    hf_checkpoint_keep_last_n: int = 3

    # Backtest
    top_k: int = 20
    hold_days: int = 5
    pred_len: int = 10
    test_start_date: str = "2024-07-01"
    benchmark_symbol: str = "^TWII"
    min_signal_threshold: float = 0.0  # skip stocks with predicted return below this; 0.0 = only positive

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
