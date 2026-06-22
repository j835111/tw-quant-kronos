import pandas as pd

from finetune_tw.config import Config
from finetune_tw.walkforward import WalkForwardFold, oof_folds, single_fold


def test_single_fold_embargo_gap():
    fold = single_fold("2015-01-01", "2023-12-31", "2024-06-30", embargo_days=110)
    assert isinstance(fold, WalkForwardFold)
    embargo_ts = pd.Timestamp(fold.embargo_end)
    train_end_ts = pd.Timestamp(fold.train_end)
    assert (embargo_ts - train_end_ts).days >= 110
    assert fold.val_start == fold.embargo_end
    assert fold.val_end == "2024-06-30"


def test_single_fold_no_overlap():
    fold = single_fold("2015-01-01", "2022-12-31", "2023-12-31", embargo_days=100)
    assert fold.val_start > fold.train_end
    assert fold.val_end >= fold.val_start


def test_oof_folds_count():
    folds = oof_folds("2015-01-01", "2023-12-31", n_folds=4, embargo_days=110)
    assert len(folds) == 4


def test_oof_folds_expanding():
    folds = oof_folds("2015-01-01", "2023-12-31", n_folds=3, embargo_days=110)
    for i in range(1, len(folds)):
        assert folds[i].train_start == "2015-01-01"
        assert folds[i].train_end > folds[i - 1].train_end


def test_oof_folds_val_no_overlap_with_train():
    folds = oof_folds("2015-01-01", "2023-12-31", n_folds=3, embargo_days=110)
    for fold in folds:
        assert fold.val_start > fold.train_end


def test_config_defaults_include_stacking_fields():
    cfg = Config()
    assert cfg.mc_sample_count == 20
    assert cfg.stacking_enabled is False
    assert cfg.analog_enabled is False
    assert cfg.analog_n_neighbors == 20
    assert cfg.analog_window == 20
    assert cfg.stacking_train_start == "2018-01-01"
    assert cfg.stacking_train_end == "2023-12-31"
    assert cfg.wf_embargo_days == 110


def test_config_from_yaml_keeps_stacking_defaults_when_absent():
    cfg = Config.from_yaml("finetune_tw/configs/config_tw_daily_retrain.yaml")
    assert cfg.mc_sample_count == 20
    assert cfg.stacking_enabled is False
    assert cfg.analog_enabled is False
    assert cfg.analog_n_neighbors == 20
    assert cfg.analog_window == 20
    assert cfg.stacking_train_start == "2018-01-01"
    assert cfg.stacking_train_end == "2023-12-31"
    assert cfg.wf_embargo_days == 110
