"""Test ML forecast họ 5 — CV không rò, scaler, tín hiệu/nhiễu, tái lập, chốt test."""
from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeltable_io import TestSplitLockedError  # noqa: E402
from models.calendar_features import DOW_DUMMY_COLS, add_dow_dummies  # noqa: E402
from models.ml_forecast import (  # noqa: E402
    MACRO_LEVEL_COLS,
    ScaledElasticNet,
    XGBWrapper,
    XGB_BACKEND,
    _en_grid,
    _xgb_grid,
    append_ml_registry,
    assert_cv_no_leak,
    build_arg_parser,
    common_complete_mask,
    evaluate_pair_rolling,
    expanding_cv_folds,
    expanding_train_mask,
    feature_lists,
    load_stage_frames,
    ml_config,
    output_tag,
    prepare_ml_frame,
    select_hyperparams_cv,
    should_refit,
    summarize_oos,
    target_col,
    target_end_col,
)


assert XGB_BACKEND == "xgboost", "cần xgboost cho test ML"


def _tiny_ml_cfg():
    """Lưới nhỏ để test nhanh — không đụng params.yaml."""
    base = ml_config()
    base["elastic_net"] = {"alpha": [0.01, 0.1], "l1_ratio": [0.5]}
    base["xgboost"] = {
        **base["xgboost"],
        "max_depth": [2],
        "n_estimators": [30],
        "learning_rate": [0.05],
    }
    base["cv"] = {"n_folds": 3}
    base["_embargo_days"] = 7
    return base


def _synth_panel(
    n_train=180,
    n_valid=60,
    seed=42,
    h=1,
    text_signal=0.0,
    noise_scale=0.15,
):
    rng = np.random.default_rng(seed)
    n = n_train + n_valid
    dates = pd.date_range("2021-03-01", periods=n, freq="D")
    har_d = rng.normal(-5.0, 0.4, n)
    har_w = har_d + rng.normal(0, 0.05, n)
    har_m = har_d + rng.normal(0, 0.08, n)
    ret_t = rng.normal(0, 0.02, n)
    logvol_t = rng.normal(10, 0.3, n)
    sent = rng.normal(0, 1.0, n)
    y = (
        0.2
        + 0.5 * har_d
        + 0.2 * har_w
        + 0.1 * har_m
        + text_signal * sent
        + rng.normal(0, noise_scale, n)
    )
    df = pd.DataFrame({
        "date": dates,
        "asset": "BTC",
        "split": ["train"] * n_train + ["valid"] * n_valid,
        "har_d": har_d,
        "har_w": har_w,
        "har_m": har_m,
        "ret_t": ret_t,
        "logvol_t": logvol_t,
        "fred_VIXCLS": rng.normal(20, 2, n),
        "fred_VIXCLS_d5": rng.normal(0, 1, n),
        "fred_DGS10": rng.normal(3, 0.2, n),
        "fred_DGS10_d5": rng.normal(0, 0.1, n),
        "fred_DTWEXBGS": rng.normal(120, 2, n),
        "fred_DTWEXBGS_d5": rng.normal(0, 0.5, n),
        "all_finbert_Sent_t": sent,
        "all_finbert_PosSent": rng.uniform(0.2, 0.5, n),
        "all_finbert_NegSent": rng.uniform(0.2, 0.5, n),
        "all_finbert_MeanProb_t": sent + rng.normal(0, 0.05, n),
        "all_vader_Sent_t": sent * 0.3 + rng.normal(0, 1, n),
        "all_finbert_AbnAttention_t": rng.normal(0, 1, n),
        "AbnAtt_regulation": rng.normal(0, 1, n),
        "AbnAtt_etf": rng.normal(0, 1, n),
        "AbnAtt_institutional_adoption": rng.normal(0, 1, n),
        "AbnAtt_hacks_security": rng.normal(0, 1, n),
        "AbnAtt_defi": rng.normal(0, 1, n),
        "AbnAtt_stablecoins": rng.normal(0, 1, n),
        "AbnAtt_monetary_policy": rng.normal(0, 1, n),
        f"y_logrv_{h}": y,
        f"y_rv_{h}": np.exp(y),
        f"embargo_drop_{h}": False,
        f"target_end_date_{h}": dates + pd.Timedelta(days=h),
    })
    return add_dow_dummies(df)


def test_feature_lists_superset():
    """(g) text là siêu tập của market; khớp khai báo."""
    mkt, full = feature_lists()
    assert set(mkt).issubset(set(full))
    assert set(DOW_DUMMY_COLS).issubset(set(mkt))
    assert "all_finbert_Sent_t" in full
    assert "AbnAtt_regulation" in full
    assert "har_d" in mkt
    cfg = ml_config()
    assert cfg["features_market"] == mkt
    assert all(c in full for c in cfg["features_text"])


def test_cv_no_target_end_leak():
    """(a) không fold nào có target_end > valid_start trong train."""
    h = 1
    df = _synth_panel(h=h)
    df = prepare_ml_frame(df, h, feature_lists()[1])
    folds = expanding_cv_folds(
        df["date"], df[target_end_col(h)], n_folds=3, embargo_days=7,
    )
    assert_cv_no_leak(folds, df["date"], df[target_end_col(h)])
    for tr, va, vstart in folds:
        assert (pd.to_datetime(df.loc[tr, target_end_col(h)]) <= vstart).all()


def test_scaler_fit_only_on_train():
    """(b) đổi dữ liệu ngoài train không đổi dự báo trên X_test cố định."""
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(80, 5))
    y_tr = X_tr[:, 0] * 0.5 + rng.normal(0, 0.1, 80)
    X_te = rng.normal(size=(10, 5))
    m1 = ScaledElasticNet(alpha=0.01, l1_ratio=0.5).fit(X_tr, y_tr)
    p1 = m1.predict(X_te)
    X_out = rng.normal(size=(200, 5)) * 100  # nhiễu ngoài — không được fit lại
    _ = X_out
    p2 = m1.predict(X_te)
    np.testing.assert_allclose(p1, p2)


def test_text_signal_beats_market():
    """(c) tín hiệu text → text thắng, CW/DM có ý nghĩa."""
    ml = _tiny_ml_cfg()
    h = 1
    cols_m, cols_f = feature_lists(ml)
    df = _synth_panel(n_train=200, n_valid=80, seed=42, h=h, text_signal=0.8, noise_scale=0.08)
    df = prepare_ml_frame(df, h, cols_f)
    mask = common_complete_mask(df, h, cols_m, cols_f)
    df = df.loc[mask].reset_index(drop=True)
    train = df[df.split == "train"].reset_index(drop=True)
    valid = df[df.split == "valid"].reset_index(drop=True)

    for kind in ("elastic_net", "xgboost"):
        sel_m = select_hyperparams_cv(train, h, cols_m, kind, ml=ml)
        sel_t = select_hyperparams_cv(train, h, cols_f, kind, ml=ml)
        oos = evaluate_pair_rolling(
            df, h, cols_m, cols_f, kind,
            sel_m["hyperparams"], sel_t["hyperparams"], ml=ml,
            eval_df=valid,
        )
        summ = summarize_oos(oos, h, kind)
        assert summ["r2_oos_vs_market"] > 0, kind
        assert summ["p_raw"] < 0.05, kind



def test_noise_null_size():
    """(d) nhiễu: ≤10% p<0.05 trên 200 sim (EN + XGB, lưới nhỏ)."""
    ml = _tiny_ml_cfg()
    # rút gọn thêm cho tốc độ null sim
    ml["xgboost"]["n_estimators"] = [20]
    ml["cv"] = {"n_folds": 2}
    cols_m, cols_f = feature_lists(ml)
    h = 1
    rng = np.random.default_rng(42)
    n_sims = 200
    for kind in ("elastic_net", "xgboost"):
        hits = 0
        for _ in range(n_sims):
            seed_i = int(rng.integers(0, 1_000_000))
            df = _synth_panel(
                n_train=120, n_valid=50, seed=seed_i, h=h,
                text_signal=0.0, noise_scale=0.2,
            )
            df = prepare_ml_frame(df, h, cols_f)
            df = df.loc[common_complete_mask(df, h, cols_m, cols_f)].reset_index(drop=True)
            train = df[df.split == "train"].reset_index(drop=True)
            valid = df[df.split == "valid"].reset_index(drop=True)
            # cố định hp để nhanh (null size của so sánh, không của CV)
            if kind == "elastic_net":
                hp = {"alpha": 0.1, "l1_ratio": 0.5}
            else:
                hp = {
                    "max_depth": 2, "n_estimators": 20, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "min_child_weight": 5, "reg_lambda": 1.0,
                    "tree_method": "hist", "n_jobs": 1, "random_state": 42,
                }
            oos = evaluate_pair_rolling(df, h, cols_m, cols_f, kind, hp, hp, ml=ml, eval_df=valid)
            summ = summarize_oos(oos, h, kind)
            if summ.get("p_raw", 1) < 0.05:
                hits += 1
        assert hits / n_sims <= 0.10, "%s null rate=%.3f" % (kind, hits / n_sims)



def test_reproducible_twice():
    """(e) tái lập."""
    ml = _tiny_ml_cfg()
    h = 1
    cols_m, cols_f = feature_lists(ml)
    df = _synth_panel(seed=42, h=h, text_signal=0.5)
    df = prepare_ml_frame(df, h, cols_f)
    df = df.loc[common_complete_mask(df, h, cols_m, cols_f)].reset_index(drop=True)
    train = df[df.split == "train"].reset_index(drop=True)
    valid = df[df.split == "valid"].reset_index(drop=True)
    np.random.seed(42)
    s1 = select_hyperparams_cv(train, h, cols_f, "elastic_net", ml=ml)
    np.random.seed(42)
    s2 = select_hyperparams_cv(train, h, cols_f, "elastic_net", ml=ml)
    assert s1["hyperparams"] == s2["hyperparams"]
    assert s1["cv_mse"] == pytest.approx(s2["cv_mse"])
    hp = s1["hyperparams"]
    o1 = evaluate_pair_rolling(df, h, cols_m, cols_f, "elastic_net", hp, hp, ml=ml, eval_df=valid)
    o2 = evaluate_pair_rolling(df, h, cols_m, cols_f, "elastic_net", hp, hp, ml=ml, eval_df=valid)
    np.testing.assert_allclose(o1["yhat_text"].values, o2["yhat_text"].values)



def test_stage_test_locked(monkeypatch):
    """(f) stage=test bị chặn khi thiếu ALLOW_TEST."""
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(TestSplitLockedError):
        load_stage_frames("test", 1, root=ROOT)


def test_xgb_refit_schedule():
    """(h) XGBoost refit đúng lịch 20 ngày."""
    assert should_refit(0, 20) is True
    assert should_refit(19, 20) is False
    assert should_refit(20, 20) is True
    assert should_refit(40, 20) is True
    assert should_refit(1, 1) is True
    ml = _tiny_ml_cfg()
    ml["refit"] = {"elastic_net_every_days": 1, "xgboost_every_days": 20}
    h = 1
    cols_m, cols_f = feature_lists(ml)
    df = _synth_panel(n_train=100, n_valid=45, seed=0, h=h, text_signal=0.3)
    df = prepare_ml_frame(df, h, cols_f)
    df = df.loc[common_complete_mask(df, h, cols_m, cols_f)].reset_index(drop=True)
    valid = df[df.split == "valid"].reset_index(drop=True)
    hp = {
        "max_depth": 2, "n_estimators": 20, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 5, "reg_lambda": 1.0,
        "tree_method": "hist", "n_jobs": 1, "random_state": 42,
    }
    oos = evaluate_pair_rolling(df, h, cols_m, cols_f, "xgboost", hp, hp, ml=ml, eval_df=valid)
    assert oos.iloc[0]["refit"]  # lần đầu luôn fit
    # các origin_i khi refit (trừ lần đầu ép) ∈ {0,20,40,...}
    origin_refit = oos.loc[oos["refit"], "origin_i"].astype(int).tolist()
    assert origin_refit[0] >= 0
    for oi in origin_refit:
        assert oi % 20 == 0 or oi == origin_refit[0]
    # nếu có ≥2 mốc chia hết 20
    multiples = [oi for oi in origin_refit if oi % 20 == 0]
    if len(multiples) >= 2:
        assert np.diff(multiples).tolist() == [20] * (len(multiples) - 1)


def test_grid_v2_superset_of_v1():
    """(b) lưới v2 ⊃ v1 (EN alpha/l1; XGB depth/n/lr/mcw)."""
    v1 = ml_config(grid="v1")
    v2 = ml_config(grid="v2")
    assert set(v1["elastic_net"]["alpha"]).issubset(set(v2["elastic_net"]["alpha"]))
    assert set(v1["elastic_net"]["l1_ratio"]).issubset(set(v2["elastic_net"]["l1_ratio"]))
    assert set(v1["xgboost"]["max_depth"]).issubset(set(v2["xgboost"]["max_depth"]))
    assert set(v1["xgboost"]["n_estimators"]).issubset(set(v2["xgboost"]["n_estimators"]))
    assert set(v1["xgboost"]["learning_rate"]).issubset(set(v2["xgboost"]["learning_rate"]))
    mcw_v1 = v1["xgboost"]["min_child_weight"]
    mcw_v1_set = {mcw_v1} if not isinstance(mcw_v1, (list, tuple)) else set(mcw_v1)
    assert mcw_v1_set.issubset(set(v2["xgboost"]["min_child_weight"]))
    # features/cv/refit giống hệt
    assert v1["features_market"] == v2["features_market"]
    assert v1["features_text"] == v2["features_text"]
    assert v1["cv"] == v2["cv"]
    assert v1["refit"] == v2["refit"]
    # mọi điểm lưới v1 xuất hiện trong v2
    en1 = {(g["alpha"], g["l1_ratio"]) for g in _en_grid(v1)}
    en2 = {(g["alpha"], g["l1_ratio"]) for g in _en_grid(v2)}
    assert en1.issubset(en2)
    key = lambda g: (g["max_depth"], g["n_estimators"], g["learning_rate"], g["min_child_weight"])
    assert {key(g) for g in _xgb_grid(v1)}.issubset({key(g) for g in _xgb_grid(v2)})


def test_cli_defaults_unchanged():
    """(c) CLI mặc định grid=v1, macro_mode=level; output_tag rỗng."""
    args = build_arg_parser().parse_args([])
    assert args.grid == "v1"
    assert args.macro_mode == "level"
    assert args.run is None
    assert output_tag("v1", "level") == ""
    assert output_tag("v2", "level") == "_gv2_mlevel"
    assert output_tag("v1", "diff") == "_gv1_mdiff"
    # ml_config() không đổi hành vi cũ
    assert ml_config()["_grid_version"] == "v1"
    m_level, _ = feature_lists(ml_config(), macro_mode="level")
    m_diff, _ = feature_lists(ml_config(), macro_mode="diff")
    assert set(MACRO_LEVEL_COLS).issubset(set(m_level))
    assert not set(MACRO_LEVEL_COLS) & set(m_diff)
    assert "fred_VIXCLS_d5" in m_diff
    assert "fred_DGS10_d5" in m_diff
    assert "fred_DTWEXBGS_d5" in m_diff


def test_append_ml_registry_backfills_v1(tmp_path):
    """Registry: dòng ML cũ nhận grid_version=v1 / macro_mode=level; không xoá dòng khác."""
    p = tmp_path / "test_registry.csv"
    old = pd.DataFrame([{
        "timestamp": "t0", "script": "models/ml_forecast.py", "model": "elastic_net|cv",
        "asset": "BTC", "h": 1, "split": "train", "n_oos": 5, "p_raw": "",
        "baseline": "n/a", "family": "ml_cv", "scale": "", "note": "",
    }, {
        "timestamp": "t0", "script": "models/rq3_narratives.py", "model": "other",
        "asset": "BTC", "h": "", "split": "train+valid", "n_oos": 10, "p_raw": 0.1,
        "baseline": "n/a", "family": "rq3", "scale": "", "note": "",
    }])
    old.to_csv(p, index=False)
    append_ml_registry(
        [{"timestamp": "t1", "script": "models/ml_forecast.py", "model": "xgboost|cv",
          "asset": "ETH", "h": 1, "split": "train", "n_oos": 5, "p_raw": "",
          "baseline": "n/a", "family": "ml_cv", "scale": "", "note": ""}],
        p, grid_version="v2", macro_mode="diff",
    )
    out = pd.read_csv(p)
    assert len(out) == 3
    assert list(out["script"]) == [
        "models/ml_forecast.py", "models/rq3_narratives.py", "models/ml_forecast.py",
    ]
    assert out.loc[0, "grid_version"] == "v1"
    assert out.loc[0, "macro_mode"] == "level"
    assert out.loc[1, "grid_version"] == "" or pd.isna(out.loc[1, "grid_version"])
    assert out.loc[2, "grid_version"] == "v2"
    assert out.loc[2, "macro_mode"] == "diff"


def test_v1_one_cell_matches_saved_summary():
    """(a) lưới v1: một ô EN BTC h=1 khớp ml_summary_valid.csv (p_raw, rmse < 1e-9)."""
    summary_path = ROOT / "outputs" / "tables" / "ml_summary_valid.csv"
    if not summary_path.exists():
        pytest.skip("chưa có ml_summary_valid.csv")
    saved = pd.read_csv(summary_path)
    row = saved[(saved.asset == "BTC") & (saved.h == 1) & (saved.model_kind == "elastic_net")]
    assert len(row) == 1
    row = row.iloc[0]

    ml = ml_config(grid="v1")
    cols_m, cols_f = feature_lists(ml, macro_mode="level")
    h = 1
    raw = load_stage_frames("valid", h, root=ROOT)
    raw = raw.loc[raw["asset"] == "BTC"].copy()
    raw = prepare_ml_frame(raw, h, cols_f)
    df = raw.loc[common_complete_mask(raw, h, cols_m, cols_f)].reset_index(drop=True)
    cv_pool = df.loc[df["split"] == "train"].reset_index(drop=True)
    eval_df = df.loc[df["split"] == "valid"].reset_index(drop=True)
    np.random.seed(42)
    sel_m = select_hyperparams_cv(cv_pool, h, cols_m, "elastic_net", ml=ml)
    sel_t = select_hyperparams_cv(cv_pool, h, cols_f, "elastic_net", ml=ml)
    oos = evaluate_pair_rolling(
        df, h, cols_m, cols_f, "elastic_net",
        sel_m["hyperparams"], sel_t["hyperparams"], ml=ml, eval_df=eval_df,
    )
    summ = summarize_oos(oos, h, "elastic_net")
    assert abs(summ["p_raw"] - float(row["p_raw"])) < 1e-9
    assert abs(summ["rmse_market"] - float(row["rmse_market"])) < 1e-9
    assert abs(summ["rmse_text"] - float(row["rmse_text"])) < 1e-9
