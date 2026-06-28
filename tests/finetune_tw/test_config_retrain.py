from finetune_tw.config import Config


def test_config_defaults_have_ic_fields():
    cfg = Config()
    assert cfg.early_stop_patience == 2
    assert cfg.ic_val_symbols == 150
    assert cfg.ic_val_dates == 8
    assert cfg.val_ic_horizons == 5
    assert cfg.hf_checkpoint_revision_out == ""
    assert cfg.hf_checkpoint_keep_last_n == 3


def test_retrain_yaml_loads():
    cfg = Config.from_yaml("finetune_tw/configs/config_tw_daily_retrain.yaml")
    assert cfg.predictor_lr == 1e-5
    assert cfg.basemodel_epochs == 6
    assert cfg.early_stop_patience == 2


def test_rtx6000_yaml_uses_persistent_storage():
    cfg = Config.from_yaml("finetune_tw/configs/config_tw_daily_rtx6000.yaml")
    assert cfg.db_path == "/mnt/first/kronos_state/data/tw_stocks.db"
    assert cfg.output_dir == "/mnt/first/kronos_state/outputs"
    assert cfg.hf_checkpoint_revision_out == "checkpoints-round-3"
    assert cfg.hf_checkpoint_keep_last_n == 3
