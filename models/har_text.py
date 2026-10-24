"""HAR + biến văn bản vs HAR (Spec B) — triển khai họ 5 PREREG.

Mặc định stage='valid' (train + valid). stage='test' cần ALLOW_TEST=1.

Mô hình cố định:
  M_fin / M_vad / M_att (+ độ bền M_att_des_dow ngoài PREREG)

So sánh chính: văn bản+DOW vs har_dow (cùng 6 biến giả DOW hai vế).
Độ bền: văn bản vs har_plain (cột *_vs_har_plain).

Holm: 9 phép (3 mô hình PREREG × 3 h) / tài sản trên p CW vs har_dow.
Seed 42. Registry append-only (+ cột baseline) trừ khi register=False.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeltable_io import TestSplitLockedError, load_model_table  # noqa: E402
from models.calendar_features import (  # noqa: E402
    DOW_DUMMY_COLS,
    add_dow_dummies,
    deseasonalize_by_dow_past_only,
)
from models.har_baseline import (  # noqa: E402
    ASSETS,
    FEATURES,
    FEATURES_DOW,
    SEED,
    _r2_oos,
    clark_west_pvalue,
    cw_series,
    embargo_col,
    expanding_train,
    fit_ols_nw,
    load_params,
    nw_bandwidth,
    predict_row,
    prepare_xy,
    project_root,
    qlike,
    target_col,
)

TEXT_MODELS: dict[str, str] = {
    "M_fin": "all_finbert_Sent_t",
    "M_vad": "all_vader_Sent_t",
    "M_att": "all_finbert_AbnAttention_t",
}
# Ngoài PREREG — chỉ độ bền
ROBUST_EXTRA: dict[str, str] = {
    "M_att_des_dow": "all_finbert_AbnAttention_des_dow",
}


def holm_adjust(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    if m == 0:
        return []
    order = np.argsort(np.asarray(pvalues, dtype=float))
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        raw = float(pvalues[int(idx)]) * (m - rank)
        val = min(1.0, max(running, raw))
        adj[int(idx)] = val
        running = val
    return adj.tolist()


def prepare_xy_text(df: pd.DataFrame, h: int, text_col: str) -> pd.DataFrame:
    """HAR + DOW + text; dropna mẫu chung."""
    base = add_dow_dummies(prepare_xy(df, h))
    if text_col not in df.columns:
        raise KeyError("Thiếu cột văn bản %s" % text_col)
    keys = ["date", "asset"]
    extra = df[keys + [text_col]].copy()
    extra["date"] = pd.to_datetime(extra["date"]).dt.tz_localize(None).dt.normalize()
    out = base.merge(extra.drop_duplicates(keys), on=keys, how="left")
    need = [target_col(h), *FEATURES, text_col, *DOW_DUMMY_COLS]
    out = out.dropna(subset=need)
    return out.sort_values("date").reset_index(drop=True)


def attach_deseason_abn(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thêm all_finbert_AbnAttention_des_dow (past-only DOW mean) — ngoài PREREG."""
    src = "all_finbert_AbnAttention_t"
    out_col = "all_finbert_AbnAttention_des_dow"
    train2 = train.copy()
    valid2 = valid.copy()
    train2["date"] = pd.to_datetime(train2["date"]).dt.tz_localize(None).dt.normalize()
    valid2["date"] = pd.to_datetime(valid2["date"]).dt.tz_localize(None).dt.normalize()
    both = pd.concat([
        train2.assign(_split="train"),
        valid2.assign(_split="valid"),
    ], ignore_index=True).sort_values(["asset", "date"])
    parts = []
    for _, g in both.groupby("asset", sort=False):
        g = g.copy()
        g[out_col] = deseasonalize_by_dow_past_only(g[src], g["date"]).to_numpy()
        parts.append(g)
    both2 = pd.concat(parts, ignore_index=True)
    return (
        both2.loc[both2["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True),
        both2.loc[both2["_split"] == "valid"].drop(columns=["_split"]).reset_index(drop=True),
    )


def check_abn_attention_identical(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    cols = ["all_vader_AbnAttention_t", "all_finbert_AbnAttention_t"]
    parts = [d[cols] for d in (train, valid) if all(c in d.columns for c in cols)]
    if not parts:
        return {"identical": None, "note": "missing columns"}
    s = pd.concat(parts, ignore_index=True).dropna()
    if s.empty:
        return {"identical": None, "n": 0}
    equal = bool((s.iloc[:, 0].values == s.iloc[:, 1].values).all())
    corr = float(np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1]) if len(s) > 1 else float("nan")
    return {"identical": equal, "corr": corr, "n_nonnull": int(len(s))}


def ensure_registry_baseline_column(path: Path) -> None:
    """Gán baseline=har_plain cho dòng cũ thiếu cột; không xóa dòng."""
    if not path.exists():
        return
    old = pd.read_csv(path)
    if "baseline" not in old.columns:
        old["baseline"] = "har_plain"
    else:
        old["baseline"] = old["baseline"].fillna("har_plain")
        old.loc[old["baseline"].astype(str).str.strip() == "", "baseline"] = "har_plain"
    old.to_csv(path, index=False)


def append_test_registry(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_registry_baseline_column(path)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(path, index=False)


def run_pair_oos(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    h: int,
    text_col: str,
) -> tuple[pd.DataFrame, dict]:
    """OOS: text vs har_plain (độ bền) và text+DOW vs har_dow (chính)."""
    ycol = target_col(h)
    plain_har = list(FEATURES)
    plain_txt = list(FEATURES) + [text_col]
    dow_har = list(FEATURES_DOW)
    dow_txt = list(FEATURES_DOW) + [text_col]
    ecol = embargo_col(h)
    if ecol not in history.columns:
        history = history.copy()
        history[ecol] = False

    rows = []
    for _, row in valid.iterrows():
        t = row["date"]
        train_t = expanding_train(history, t, h)
        train_t = train_t.dropna(subset=[ycol, *dow_txt])
        if len(train_t) < len(dow_txt) + 2:
            continue

        res_hp = fit_ols_nw(train_t, h, plain_har)
        res_tp = fit_ols_nw(train_t, h, plain_txt)
        res_hd = fit_ols_nw(train_t, h, dow_har)
        res_td = fit_ols_nw(train_t, h, dow_txt)

        yhat_hp = predict_row(res_hp, row, plain_har)
        yhat_tp = predict_row(res_tp, row, plain_txt)
        yhat_hd = predict_row(res_hd, row, dow_har)
        yhat_td = predict_row(res_td, row, dow_txt)
        y = float(row[ycol])
        rv = float(row[f"y_rv_{h}"]) if f"y_rv_{h}" in row.index and pd.notna(row.get(f"y_rv_{h}")) else float(np.exp(y))

        rows.append({
            "date": t,
            "n_train_t": len(train_t),
            "train_idx": tuple(train_t.index.tolist()),
            "y_logrv": y,
            "yhat_har_plain": yhat_hp,
            "yhat_text_plain": yhat_tp,
            "yhat_har_dow": yhat_hd,
            "yhat_text_dow": yhat_td,
            # alias chính = DOW
            "yhat_har": yhat_hd,
            "yhat_text": yhat_td,
            "e_har_plain": y - yhat_hp,
            "e_text_plain": y - yhat_tp,
            "e_har_dow": y - yhat_hd,
            "e_text_dow": y - yhat_td,
            "e_har": y - yhat_hd,
            "e_text": y - yhat_td,
            "d_cw_vs_har_dow": float(cw_series(np.array([y]), np.array([yhat_td]), np.array([yhat_hd]))[0]),
            "d_cw_vs_har_plain": float(cw_series(np.array([y]), np.array([yhat_tp]), np.array([yhat_hp]))[0]),
            "d_cw": float(cw_series(np.array([y]), np.array([yhat_td]), np.array([yhat_hd]))[0]),
            "rv": rv,
            "rv_hat_har_dow": float(np.exp(yhat_hd)),
            "rv_hat_text_dow": float(np.exp(yhat_td)),
            "rv_hat_har_plain": float(np.exp(yhat_hp)),
            "rv_hat_text_plain": float(np.exp(yhat_tp)),
            "qlike_har_dow": float(qlike(np.array([rv]), np.array([np.exp(yhat_hd)]))[0]),
            "qlike_text_dow": float(qlike(np.array([rv]), np.array([np.exp(yhat_td)]))[0]),
            "qlike_har_plain": float(qlike(np.array([rv]), np.array([np.exp(yhat_hp)]))[0]),
            "qlike_text_plain": float(qlike(np.array([rv]), np.array([np.exp(yhat_tp)]))[0]),
            "qlike_har": float(qlike(np.array([rv]), np.array([np.exp(yhat_hd)]))[0]),
            "qlike_text": float(qlike(np.array([rv]), np.array([np.exp(yhat_td)]))[0]),
        })

    oos = pd.DataFrame(rows)
    meta: dict = {}
    if oos.empty:
        for k in (
            "n_oos", "n_train_first", "n_train_last",
            "r2_oos_vs_har_dow", "r2_oos_vs_har_plain",
            "rmse_har_dow", "rmse_text_dow", "rmse_har_plain", "rmse_text_plain",
            "qlike_har_dow", "qlike_text_dow", "qlike_har_plain", "qlike_text_plain",
            "cw_pvalue_raw", "cw_pvalue_vs_har_plain",
            "r2_oos_vs_har", "rmse_har", "rmse_text", "qlike_har", "qlike_text",
        ):
            meta[k] = 0 if k.startswith("n_") else float("nan")
        return oos, meta

    sse_td = float((oos["e_text_dow"] ** 2).sum())
    sse_hd = float((oos["e_har_dow"] ** 2).sum())
    sse_tp = float((oos["e_text_plain"] ** 2).sum())
    sse_hp = float((oos["e_har_plain"] ** 2).sum())
    _, _, p_dow = clark_west_pvalue(oos["d_cw_vs_har_dow"].values, h)
    _, _, p_plain = clark_west_pvalue(oos["d_cw_vs_har_plain"].values, h)
    meta.update({
        "n_oos": len(oos),
        "n_train_first": int(oos["n_train_t"].iloc[0]),
        "n_train_last": int(oos["n_train_t"].iloc[-1]),
        "r2_oos_vs_har_dow": _r2_oos(sse_td, sse_hd),
        "r2_oos_vs_har_plain": _r2_oos(sse_tp, sse_hp),
        "rmse_har_dow": float(np.sqrt((oos["e_har_dow"] ** 2).mean())),
        "rmse_text_dow": float(np.sqrt((oos["e_text_dow"] ** 2).mean())),
        "rmse_har_plain": float(np.sqrt((oos["e_har_plain"] ** 2).mean())),
        "rmse_text_plain": float(np.sqrt((oos["e_text_plain"] ** 2).mean())),
        "qlike_har_dow": float(oos["qlike_har_dow"].mean()),
        "qlike_text_dow": float(oos["qlike_text_dow"].mean()),
        "qlike_har_plain": float(oos["qlike_har_plain"].mean()),
        "qlike_text_plain": float(oos["qlike_text_plain"].mean()),
        "cw_pvalue_raw": p_dow,  # chính = vs har_dow
        "cw_pvalue_vs_har_plain": p_plain,
        # alias
        "r2_oos_vs_har": _r2_oos(sse_td, sse_hd),
        "rmse_har": float(np.sqrt((oos["e_har_dow"] ** 2).mean())),
        "rmse_text": float(np.sqrt((oos["e_text_dow"] ** 2).mean())),
        "qlike_har": float(oos["qlike_har_dow"].mean()),
        "qlike_text": float(oos["qlike_text_dow"].mean()),
    })
    return oos, meta


def final_text_coef(history: pd.DataFrame, h: int, text_col: str, with_dow: bool = True) -> tuple[float, float]:
    cols = list(FEATURES_DOW if with_dow else FEATURES) + [text_col]
    hist = history.dropna(subset=[target_col(h), *cols])
    res = fit_ols_nw(hist, h, cols)
    return float(res.params[text_col]), float(res.bse[text_col])


def run_asset_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    asset: str,
    h: int,
    model_name: str,
    text_col: str,
    *,
    prior: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """OOS trên ``valid`` (hoặc test khi gọi với prior=valid, valid=test).

    history = train [+ prior] + eval; expanding_train chỉ dùng quá khứ trước t.
    """
    tr = prepare_xy_text(train.loc[train["asset"] == asset], h, text_col)
    va = prepare_xy_text(valid.loc[valid["asset"] == asset], h, text_col)
    if tr.empty or va.empty:
        raise ValueError("Mẫu rỗng: asset=%s h=%s model=%s" % (asset, h, model_name))
    parts = [tr]
    n_prior = 0
    if prior is not None:
        pr = prepare_xy_text(prior.loc[prior["asset"] == asset], h, text_col)
        if pr.empty:
            raise ValueError("Mẫu prior rỗng: asset=%s h=%s model=%s" % (asset, h, model_name))
        parts.append(pr)
        n_prior = len(pr)
    parts.append(va)
    history = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    oos, meta = run_pair_oos(history, va, h, text_col)
    coef, se = final_text_coef(history, h, text_col, with_dow=True)
    summary = {
        "asset": asset,
        "h": h,
        "model": model_name,
        "text_col": text_col,
        "baseline": "har_dow",
        "n_train_pool": len(tr) + n_prior,
        "n_valid_pool": len(va),
        **meta,
        "coef_text": coef,
        "se_text_nw": se,
        "nw_bandwidth": nw_bandwidth(h),
        "prereg_family5": model_name in TEXT_MODELS,
    }
    return oos.drop(columns=["train_idx"], errors="ignore"), summary


def run_all(
    horizons: list[int] | None = None,
    assets: tuple[str, ...] = ASSETS,
    root: Path | None = None,
    out_dir: Path | None = None,
    panel: str = "title",
    seed: int = SEED,
    models: dict[str, str] | None = None,
    include_deseason_robust: bool = True,
    register: bool = True,
    stage: str = "valid",
) -> pd.DataFrame:
    root = root or project_root()
    params = load_params(root)
    horizons = horizons or [int(x) for x in params.get("horizons", [1, 3, 7])]
    models = models or dict(TEXT_MODELS)
    seed = int(params.get("seed", seed))
    stage = str(stage).lower()
    if stage not in ("valid", "test"):
        raise ValueError("stage ∈ {'valid','test'}, nhận %r" % stage)
    if stage == "test" and os.environ.get("ALLOW_TEST") != "1":
        raise TestSplitLockedError("stage='test' cần ALLOW_TEST=1")
    random.seed(seed)
    np.random.seed(seed)

    out_dir = out_dir or (root / "outputs" / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_path = out_dir / "test_registry.csv"
    if register:
        ensure_registry_baseline_column(reg_path)

    train0 = load_model_table("B", "train", horizon=1, drop_embargo=True, root=root, panel=panel)
    valid0 = load_model_table("B", "valid", horizon=1, drop_embargo=True, root=root, panel=panel)
    abn_note = check_abn_attention_identical(train0, valid0)
    print("AbnAttention vader vs finbert (all):", abn_note)

    summaries: list[dict] = []
    oos_by_key: dict[tuple[str, int], list[pd.DataFrame]] = {}
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    registry_rows: list[dict] = []

    for h in horizons:
        train = load_model_table("B", "train", horizon=h, drop_embargo=True, root=root, panel=panel)
        valid = load_model_table("B", "valid", horizon=h, drop_embargo=True, root=root, panel=panel)
        test = None
        if stage == "test":
            test = load_model_table("B", "test", horizon=h, drop_embargo=True, root=root, panel=panel)
        run_models = dict(models)
        # Deseason robust chỉ trên valid (train↔valid); không áp cho stage=test
        if include_deseason_robust and stage == "valid" and "M_att" in models:
            train_d, valid_d = attach_deseason_abn(train, valid)
        else:
            train_d = valid_d = None

        for asset in assets:
            for model_name, text_col in run_models.items():
                if stage == "valid":
                    oos, summary = run_asset_model(
                        train, valid, asset, h, model_name, text_col
                    )
                else:
                    oos, summary = run_asset_model(
                        train, test, asset, h, model_name, text_col, prior=valid
                    )
                oos = oos.assign(model=model_name, asset=asset, h=h)
                oos_by_key.setdefault((asset, h), []).append(oos)
                summaries.append(summary)
                registry_rows.append({
                    "timestamp": ts,
                    "script": "models/har_text.py",
                    "model": model_name,
                    "asset": asset,
                    "h": h,
                    "split": stage,
                    "n_oos": summary["n_oos"],
                    "p_raw": summary["cw_pvalue_raw"],
                    "baseline": "har_dow",
                })
                print(
                    "%s %s h=%d | n_oos=%d R2vsDOW=%.4f R2vsPlain=%.4f "
                    "CW_dow=%.4g CW_plain=%.4g coef=%.4f(%.4f)"
                    % (
                        asset, model_name, h, summary["n_oos"],
                        summary["r2_oos_vs_har_dow"], summary["r2_oos_vs_har_plain"],
                        summary["cw_pvalue_raw"], summary["cw_pvalue_vs_har_plain"],
                        summary["coef_text"], summary["se_text_nw"],
                    )
                )

            if include_deseason_robust and train_d is not None and "M_att" in models:
                name, col = "M_att_des_dow", ROBUST_EXTRA["M_att_des_dow"]
                try:
                    oos, summary = run_asset_model(train_d, valid_d, asset, h, name, col)
                except (ValueError, KeyError) as e:
                    print("skip", name, asset, h, e)
                    continue
                summary["prereg_family5"] = False
                summary["note"] = "deseason DOW AbnAttention — ngoài PREREG"
                oos = oos.assign(model=name, asset=asset, h=h)
                oos_by_key.setdefault((asset, h), []).append(oos)
                summaries.append(summary)
                registry_rows.append({
                    "timestamp": ts,
                    "script": "models/har_text.py",
                    "model": name,
                    "asset": asset,
                    "h": h,
                    "split": stage,
                    "n_oos": summary["n_oos"],
                    "p_raw": summary["cw_pvalue_raw"],
                    "baseline": "har_dow",
                })
                print(
                    "%s %s h=%d [robust] | R2vsDOW=%.4f CW=%.4g"
                    % (asset, name, h, summary["r2_oos_vs_har_dow"], summary["cw_pvalue_raw"])
                )

    summary_df = pd.DataFrame(summaries)
    summary_df["cw_pvalue_holm"] = np.nan
    # Holm chỉ trên 9 phép PREREG (không gồm M_att_des_dow)
    prereg_models = [m for m in models if m in TEXT_MODELS]
    model_order = {m: i for i, m in enumerate(prereg_models)}
    for asset in assets:
        mask = (summary_df["asset"] == asset) & (summary_df["model"].isin(prereg_models))
        sub = summary_df.loc[mask].copy()
        if sub.empty:
            continue
        sub["_ord"] = sub["model"].map(model_order) * 100 + sub["h"]
        sub = sub.sort_values("_ord")
        adj = holm_adjust(sub["cw_pvalue_raw"].astype(float).tolist())
        summary_df.loc[sub.index, "cw_pvalue_holm"] = adj

    for (asset, h), parts in oos_by_key.items():
        pd.concat(parts, ignore_index=True).to_csv(out_dir / f"har_text_{asset}_h{h}.csv", index=False)

    sum_path = out_dir / "har_text_summary.csv"
    summary_df.to_csv(sum_path, index=False)
    if register:
        append_test_registry(registry_rows, reg_path)

    meta = {
        "seed": seed,
        "panel": panel,
        "stage": stage,
        "prereg_family": 5,
        "primary_baseline": "har_dow",
        "robustness_baseline": "har_plain",
        "holm": "per asset, 9 PREREG tests (3 models × 3 h) on CW vs har_dow",
        "abn_attention_note": abn_note,
        "models": models,
        "extra_robust": list(ROBUST_EXTRA) if include_deseason_robust and stage == "valid" else [],
        "summary_path": str(sum_path),
        "register": register,
    }
    (out_dir / "har_text_summary.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("Summary ->", sum_path)
    if register:
        print("Registry ->", reg_path)
    else:
        print("Registry skipped (register=False)")
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="HAR + text vs har_dow (PREREG family 5)")
    parser.add_argument("--panel", default="title", choices=["title", "fulltext", "nlp_text"])
    args = parser.parse_args()
    run_all(panel=args.panel)


if __name__ == "__main__":
    main()
