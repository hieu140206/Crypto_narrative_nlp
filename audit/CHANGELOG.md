# CHANGELOG (Cursor ghi, audit đọc)

## [2026-10-25] LOSO robustness (robust_loso)
- **Sửa gì / vì sao:** Leave-one-source-out: full + exclude từng s ∈ {coindesk, cointelegraph, decrypt, blockworks}; rebuild daily panel + narratives + model_table A/B; Spec A ΔR²adj FinBERT/VADER vs market (MBB 10/5000/seed42); Family 5 HAR+text valid 2023 (Holm-9/asset/source); coverage median/%days&lt;3. `exclude_sources` trên `build_daily_panel` / `build_narratives` / `build_model_table` / loaders; không ghi đè parquet mặc định.
- **File:** `models/robust_loso.py`, `tests/test_robust_loso.py`, `build_daily_panel.py`, `build_narratives.py`, `build_model_table.py`, `models/rq1_rq2_sentiment.py` (`run_rq1_vs_market`, `table_path`), `models/har_text.py` (`exclude_sources`), `data/processed/loso/<id>/`, `outputs/tables/robust_loso_{A,B,C}.csv`, `robust_loso_check.json`, `test_registry.csv`
- **Số liệu:** Check full vs `model_table_B` train+valid: ok, n=2072, 89 cols, max|Δ|=0. Registry +150 (family=`robust_loso`) → 607 (đã bỏ 150 dòng trùng từ lần chạy dở trước). Không ALLOW_TEST; dates ≤2023-12-31.
- **Cách kiểm lại:** `python -m models.robust_loso`; `pytest -q tests/test_robust_loso.py`; đối chiếu `outputs/tables/robust_loso_check.json` và CSV A/B/C
- **Chưa chắc:** —

## [2026-10-25] Descriptive RSS holdout + cryptonews cross-source
- **Sửa gì / vì sao:** Phân tích mô tả (không confirmatory): (1) pipeline RSS holdout frozen clean→score title→narratives v4 + Wilson CI vs main 2025H1; (2) cryptonews-only daily text series vs four-source main, Pearson/Spearman + MBB CI (block=10, B=5000, seed=42), 3 versions × 2 periods. Không append `test_registry.csv`; không ALLOW_TEST; không sửa config/PREREG.
- **File:** `models/descript_rss_cryptonews.py`, `tests/test_descript_rss_cryptonews.py`, `outputs/tables/descript_rss_holdout_*.csv`, `outputs/tables/descript_cryptonews_*.csv`, interim `rss_holdout*.parquet`, `sentiment_doc_cryptonews.parquet`, `cryptonews_narrative_doc.parquet`, sentiment cache title
- **Số liệu:** narratives.yaml LF sha256=`78b5a05b…99d9` (khớp frozen); RSS raw 256→clean 213; cryptonews 6676 bài, median 7/ngày, %days&lt;3=20.1%; corr rows=36; pytest **141 passed**. Không đọc outcomes test-split.
- **Cách kiểm lại:** `python -m models.descript_rss_cryptonews`; `pytest -q tests/test_descript_rss_cryptonews.py`; đối chiếu CSV `outputs/tables/descript_*`
- **Chưa chắc:** —

## [2026-10-25] Explor CryptoBERT (PREREG §5)
- **Sửa gì / vì sao:** Phân tích thăm dò `ElKulako/cryptobert` (family=`explor_cryptobert`): A nhãn 630; B subsample 25% stratified source×month (seed 42, ≤2023-12-31); B1 corr; B2 Spec A ΔR²adj vs market (MBB 5000); B3 HAR+DOW M_cbert valid 2023 (Holm explor-only 6). Không đụng confirmatory pipelines / test / config / PREREG.
- **File:** `models/explor_cryptobert.py`, `tests/test_explor_cryptobert.py`, `outputs/tables/explor_cryptobert_*.csv|parquet|json`, `outputs/tables/cryptobert_subsample_ids.csv`, `outputs/tables/test_registry.csv` (+24), cache `data/interim/sentiment_cache/cryptobert__*__title.parquet`
- **Số liệu:** pool 40037 → subsample 10007; registry 433→457; HF rev `9e37c910fe87727cb842a9ac55c6388256fe0f15`; pytest 137 passed. Không đọc split test.
- **Cách kiểm lại:** `python -m models.explor_cryptobert`; `pytest -q tests/test_explor_cryptobert.py`; đối chiếu CSV dưới `outputs/tables/explor_cryptobert_*`
- **Chưa chắc:** —

## [2026-10-25] run_test_once: out_dir real → outputs/test
- **Sửa gì / vì sao:** Mode real ghi kết quả vào `outputs/test/` (không phải `tables/`). Registry vẫn append `outputs/tables/test_registry.csv` qua `registry_path=`. Dry/pseudo không đụng tables.
- **File:** `models/run_test_once.py`, `models/har_text.py`, `models/ml_forecast.py`, `tests/test_run_test_once.py`
- **Số liệu:** pytest 126 passed. Không chạy stage=test.
- **Cách kiểm lại:** `pytest -q tests/test_run_test_once.py`
- **Chưa chắc:** —

## [2026-10-25] run_test_once: khóa sớm + --pseudo-test
- **Sửa gì / vì sao:** TEST_LOCK ghi `status=started` trước khi đọc test; `completed` sau thành công; lỗi giữ `started`. Thêm `--pseudo-test` (nhánh stage=test + `split_override` dịch lùi, cấm date≥2024). `load_splits_with_override` trong `modeltable_io`.
- **File:** `models/run_test_once.py`, `models/har_text.py`, `models/ml_forecast.py`, `modeltable_io.py`, `tests/test_run_test_once.py`, `outputs/pseudotest/*`
- **Số liệu:** pseudo vs valid HAR 18/18 Δ=0; ML 12/12 Δ=0 (HP khớp). Registry 379; không TEST_LOCK. pytest 123 passed.
- **Cách kiểm lại:** `python -m models.run_test_once --pseudo-test`; `pytest -q`
- **Chưa chắc:** —

## [2026-10-25] Locked test-run script + dry-run valid
- **Sửa gì / vì sao:** Thêm `models/run_test_once.py` — một process: Family 5 HAR+text (9 phép/asset, `include_deseason_robust=False`) rồi Family 5-ML (`grid=v2`, `macro_mode=level`, EN+XGB). Khoá `outputs/TEST_LOCK` sau lần thật (exit 2 nếu đã có; không cờ bypass). `--dry-run valid` → `outputs/dryrun/`, `register=False`, không tạo lock. `har_text.run_all` thêm `register`/`stage` (test cần `ALLOW_TEST=1`). Đã chạy dry-run; **không** chạy stage=test.
- **File:** `models/run_test_once.py`, `models/har_text.py`, `tests/test_run_test_once.py`, `outputs/dryrun/*`, `audit/CHANGELOG.md`
- **Số liệu:** Dry-run vs validation — **match tuyệt đối** (max|Δ|=0) trên 18 hàng HAR PREREG (`cw_pvalue_raw/holm`, R², RMSE, coef) và 12 hàng ML v2 level (`p_raw/holm`, R² vs market/HAR+DOW, RMSE, cv_mse, HP). Registry **379→379** (không đổi). `TEST_LOCK` không tồn tại.
- **Cách kiểm lại:** `python -m models.run_test_once --dry-run valid`; đối chiếu `outputs/dryrun/har_text_summary.csv` ↔ `outputs/tables/har_text_summary.csv` (lọc M_fin/M_vad/M_att); `outputs/dryrun/ml_summary_valid_gv2_mlevel.csv` ↔ `outputs/tables/ml_summary_valid_gv2_mlevel.csv`; `pytest -q`
- **Chưa chắc:** —

## [2026-10-25] ML --run valid lưới v2 (level + diff)
- **Sửa gì / vì sao:** Chạy hai lượt valid: `--grid v2 --macro_mode level` rồi `diff`. Không sửa config/PREREG; không chạy v1/test.
- **File:** `outputs/tables/ml_*_gv2_mlevel.*`, `ml_*_gv2_mdiff.*`, `test_registry.csv` (+72)
- **Số liệu:** params sha256=4fe733cf…19b3 (khớp). Registry 307→379. Phân bố grid×macro: (v1,level)=36; (v2,level)=36; (v2,diff)=36; trống=271 (non-ML).
- **Cách kiểm lại:** đối chiếu `ml_summary_valid_gv2_m{level,diff}.csv`; `pytest -q`
- **Chưa chắc:** —

## [2026-10-25] ML lưới v2 + macro_mode (chưa chạy đánh giá)
- **Sửa gì / vì sao:** Giữ nguyên khối `ml:` (v1). Thêm `ml_v2:` (EN alpha→3.0; XGB depth/n/lr mở rộng + min_child_weight [5,20]). CLI `--grid {v1,v2}` / `--macro_mode {level,diff}`; `diff` bỏ FRED mức. Registry: cột `grid_version`/`macro_mode` (backfill ML cũ = v1/level khi append). Tên file mặc định v1+level không đổi.
- **File:** `config/params.yaml`, `models/ml_forecast.py`, `tests/test_ml_forecast.py`
- **Số liệu:** không chạy `--run`. sha256 params = 4fe733cf…19b3; narratives = 7f33bbbf…711b. pytest 113 passed.
- **Cách kiểm lại:** `pytest -q`; `python -m models.ml_forecast` (không `--run`); chờ chủ repo commit params rồi mới chạy v2.
- **Chưa chắc:** —

## [2026-10-25] ML forecast --run valid (họ 5)
- **Sửa gì / vì sao:** Sửa `evaluate_pair_rolling` nhận `history` + `eval_df` (expanding_train dùng đủ train, OOS chỉ valid). Lưu detail JSON (HP, EN coef, XGB gain), HAR+DOW R² tham chiếu. Chạy `python -m models.ml_forecast --run valid` (không ALLOW_TEST). Không sửa lưới/`params.yaml`/`PREREG.md`.
- **File:** `models/ml_forecast.py`, `tests/test_ml_forecast.py`, `outputs/tables/ml_*_valid.*`, `test_registry.csv`
- **Số liệu:** sha256 params = cc7ff1f8…6610 (khớp). Registry 271→307 (+36). n_oos BTC/ETH: h1=352, h3=350, h7=346. Xem báo cáo chat.
- **Cách kiểm lại:** `pytest -q`; `python -m models.ml_forecast --run valid`; đối chiếu `ml_summary_valid.csv` + `ml_detail_*.json`
- **Chưa chắc:** —

## [2026-10-25] ML forecast scaffold (họ 5 EN+XGB) — chưa chạy thật
- **Sửa gì / vì sao:** Thêm khối `ml:` vào `params.yaml` (lưới cố định). `models/ml_forecast.py` — Elastic Net + XGBoost, CV expanding+embargo, scaler chỉ train, CW/DM-HLN, Holm 6, stage valid/test (test khoá). Chỉ test dữ liệu giả; `--run` mới đánh giá thật.
- **File:** `config/params.yaml`, `models/ml_forecast.py`, `tests/test_ml_forecast.py`
- **Số liệu:** không chạy trên panel thật. Backend xgboost 3.4.1.
- **Cách kiểm lại:** `pytest -q tests/test_ml_forecast.py`; `python -m models.ml_forecast` (in ml + sha256); không `--run`
- **Chưa chắc:** —

## [2026-10-25] RQ4 calendar-day lags + BIC mẫu chung
- **Sửa gì / vì sao:** `rq4_leadlag.py` — reindex lịch đầy đủ, `shift(k)` theo ngày; NaN ở t−k loại khỏi pt; BIC trên mẫu hợp lệ chung p=1..7; mỗi pt ghi n riêng; registry `note=calendar_lag` (append).
- **File:** `models/rq4_leadlag.py`, `tests/test_rq4.py`, `outputs/tables/rq4_*.csv`, `test_registry.csv`
- **Số liệu:** BTC main p=1, n_SR=959 / n_RS=907; BIC min tại p=1; registry +16 (tổng 271). Không sửa PREREG; không commit
- **Cách kiểm lại:** `pytest -q`; `python -m models.rq4_leadlag`
- **Chưa chắc:** —

## [2026-10-25] RQ4 + noprice panel + H2 raw / p-floor
- **Sửa gì / vì sao:** H2 Wald chính trên Pos/Neg gốc (+ cột 1SD phụ); p_boot sàn 1/(B+1) nhãn `<1/5000`. Panel `daily_panel_noprice` + model_table A/B noprice (Sent chỉ price_desc=0; NV gốc giữ; NV_noprice; min_n=5). `models/rq4_leadlag.py` VAR+BIC+Granger tổng hệ số HAC+IRF. RQ1/RQ2 robust noprice.
- **File:** `build_daily_panel.py`, `build_model_table.py`, `modeltable_io.py`, `config/paths.yaml`, `models/rq1_rq2_sentiment.py`, `models/rq4_leadlag.py`, `tests/test_rq4.py`, `tests/test_panel_build.py`, outputs rq4_*/rq1_*/rq2_*, figures, registry
- **Số liệu:** price_desc ~30% bài; nan_sent_days(all)=109; RQ4 BTC Sent→ret không Holm-sig, ret→Sent có; registry +44 lần này (tổng 255). Không sửa PREREG.md
- **Cách kiểm lại:** `pytest -q`; `python -m build_daily_panel --panel noprice`; `python -m build_model_table --panel noprice`; `python -m models.rq4_leadlag`; `python -m models.rq1_rq2_sentiment`
- **Chưa chắc:** —

## [2026-10-25] RQ1 + RQ2 sentiment (PREREG họ 1–2)
- **Sửa gì / vì sao:** `models/rq1_rq2_sentiment.py` — họ 1 ΔR²_adj FinBERT vs VADER + MBB(10, 5000); họ 2 Wald Pos+Neg (A + B h=1,3,7); import HAC/Holm/registry từ rq3; DOW + controls; robust MeanProb/nlp_text/NV5%/ETH. Build `model_table_*_fulltext` cho robust nlp_text.
- **File:** `models/rq1_rq2_sentiment.py`, `tests/test_rq1_rq2.py`, `outputs/tables/rq1_*.csv`, `rq2_*.csv`, `rq1_rq2_summary.csv`, `test_registry.csv`, `data/processed/model_table_{A,B}_fulltext.parquet`
- **Số liệu:** registry +22; xem bảng báo cáo. Không sửa PREREG.md; không commit
- **Cách kiểm lại:** `pytest -q`; `python -m models.rq1_rq2_sentiment`; `git rev-parse HEAD`
- **Chưa chắc:** —

## [2026-10-25] RQ3 narratives (PREREG họ 3, Spec A)
- **Sửa gì / vì sao:** `models/rq3_narratives.py` — 7 AbnAtt_k × {y_log_rv,y_abvol,y_ret}; z-score trên mẫu ước lượng; OLS+NW(bw=5); kiểm soát trễ + DOW; Holm 21/asset; diag lead/lag + robust (total/joint/|z|≤4) chỉ BTC; registry append family=rq3_*.
- **File:** `models/rq3_narratives.py`, `tests/test_rq3.py`, `outputs/tables/rq3_*.csv`, `test_registry.csv`
- **Số liệu:** main 42 dòng (BTC+ETH); diag 42; robust 63; registry +147 (tổng sau chạy xem báo cáo). confirmed Holm<0.05: BTC 0 / ETH 0 trên train+valid.
- **Cách kiểm lại:** `pytest -q`; `python -m models.rq3_narratives`; HEAD phải 806b462 hoặc commit chủ repo; không sửa PREREG.md; không commit
- **Chưa chắc:** —

## [2026-10-25] HAR DOW baseline + text vs har_dow
- **Sửa gì / vì sao:** Baseline thiếu hiệu ứng ngày trong tuần. Thêm `models/calendar_features.py` (6 biến giả DOW từ ngày t, bỏ Mon; deseason AbnAtt past-only). `har_plain` giữ nguyên; `har_dow` = plain+DOW. `har_text` so chính vs `har_dow`, cột độ bền vs `har_plain`; `M_att_des_dow` ngoài PREREG. Registry: 18 dòng cũ `baseline=har_plain`, dòng mới `har_dow`.
- **File:** `models/calendar_features.py`, `models/har_baseline.py`, `models/har_text.py`, `tests/test_har_baseline.py`, `tests/test_har_text.py`, `outputs/tables/har_baseline_*.csv`, `har_text_*.csv`, `test_registry.csv`
- **Số liệu (valid):** har_dow vs har_plain R²_OOS h=1 BTC/ETH **0.253/0.173**, CW p≈0; h=3 **0.173/0.114**; h=7 **0.011/0.006**. Text vs har_dow: R²_OOS gần 0 / âm; CW Holm (9/asset) đều ≥0.74. Registry 18+24=42. HEAD **806b462**.
- **Cách kiểm lại:** `pytest -q` → **70 passed**; `python -m models.har_baseline`; `python -m models.har_text`; không sửa PREREG.md; không commit
- **Chưa chắc:** —

## [2026-10-25] HAR+text (PREREG họ 5) vs HAR
- **Sửa gì / vì sao:** `models/har_text.py` — M_fin/M_vad/M_att; mẫu chung; R²/CW vs HAR; Holm theo asset; `test_registry.csv` append-only. Mở rộng `fit_ols_nw`/`predict_row` trong har_baseline.
- **File:** `models/har_text.py`, `models/har_baseline.py`, `tests/test_har_text.py`, `outputs/tables/har_text_*.csv`, `test_registry.csv`
- **Số liệu:** xem summary; AbnAttention vader≡finbert (all); không đụng test/PREREG.md
- **Cách kiểm lại:** `pytest -q`; HEAD sau commit chủ repo (feat/tv4) — không commit agent
- **Chưa chắc:** —

## [2026-10-24] HAR baseline: chặn rò target + B0/B1
- **Sửa gì / vì sao:** `expanding_train` yêu cầu `target_end_date_h <= t`; benchmark B0 (mean y) + B1 (const+har_m); `naive_har_m` chỉ đối chiếu; CW HAR vs B0 và vs B1; R²/RMSE/QLIKE cho cả ba.
- **File:** `models/har_baseline.py`, `tests/test_har_baseline.py`, `outputs/tables/har_baseline_*.csv`
- **Số liệu:** R²_OOS vs B0 ~0.48–0.72; vs B1 ~0.03–0.09; CW vs B0/B1 đều p≪0.05 trên valid
- **Cách kiểm lại:** `pytest -q` → **59 passed**; HEAD 664e95f
- **Chưa chắc:** —

## [2026-10-24] HAR-RV baseline (train+valid, Spec B)
- **Sửa gì / vì sao:** `models/har_baseline.py` — OLS+NW trên train; OOS rolling-origin valid vs bench `har_m`; Clark–West một phía; `rv_hat=exp(ŷ)` không Jensen; chỉ `load_model_table`.
- **File:** `models/har_baseline.py`, `models/__init__.py`, `tests/test_har_baseline.py`, `outputs/tables/har_baseline_*.csv`
- **Số liệu:** xem `har_baseline_summary.csv` (BTC/ETH × h=1,3,7); n_oos ≈ 350–356
- **Cách kiểm lại:** `pytest -q` → **56 passed**; HEAD 664e95f; không đụng test
- **Chưa chắc:** —

## [2026-10-24] HAR min_periods + bỏ incomplete khỏi trung bình
- **Sửa gì / vì sao:** `har_w` rolling(5, min_periods=4), `har_m` rolling(22, min_periods=18); incomplete → NaN trong cửa sổ (không điền); docstring giải thích. Test 1 incomplete / 3 incomplete / har_m tay.
- **File:** `build_model_table.py`, `tests/test_model_table.py`, model_table parquet
- **Số liệu:** har_w/har_m NaN train+valid = **0**; har_d NaN = incomplete (train 10, valid 2). B usable (har_*+y_logrv_1): train BTC/ETH **661**/671; valid **363**/365; sau ~embargo_drop_1: **654** / **356** mỗi asset
- **Cách kiểm lại:** `pytest -q` → **50 passed**; HEAD 664e95f
- **Chưa chắc:** —

## [2026-10-24] Model table Spec A/B + chốt test + assert sentiment
- **Sửa gì / vì sao:** Nợ: assert mọi `in_main_sample` có cả vader+finbert. `build_model_table.py` Spec A/B (all_/own_/narratives/FRED as-of/HAR/target/embargo); `modeltable_io.load_model_table` khoá test trừ `ALLOW_TEST=1`; `--panel fulltext`; `own_n_min` trong params.
- **File:** `build_daily_panel.py`, `build_model_table.py`, `modeltable_io.py`, `tests/test_model_table.py`, `tests/test_panel_build.py`, `config/params.yaml`, `config/paths.yaml`, `data/processed/model_table_{A,B}.parquet`
- **Số liệu:** A/B shape **(3166, 87)/(3166, 100)** = 1583 ngày × 2 asset; first `all_*_AbnAttention` **2021-03-31**; embargo_drop train/valid = **14**/h (7 ngày × 2 asset)
- **Cách kiểm lại:** `pytest -q`; không đọc split=test; `git rev-parse HEAD` vẫn 664e95f
- **Chưa chắc:** QUESTIONS #7 (params hash), #8 (embargo_drop_* vs một cột)

## [2026-10-24] Narrative dict v4 (cbdc + regulation/ai + VIF AbnAtt)
- **Sửa gì / vì sao:** Tách `cbdc` (exploratory) khỏi monetary; thu hẹp regulation (SEC/CFTC/FCA/bill…; loại Coinbase International launches + lawsuit tư); ai_general loại how to / ChatGPT picks, GPU chỉ với token|agent; VIF trên `AbnAtt_k` (không còn Att_k); samples v4.
- **File:** `config/narratives.yaml`, `build_narratives.py`, `tests/test_narratives.py`, `audit/narrative_samples_v4.md`, narrative parquet/log
- **Số liệu trước → sau:**
  - matched **39.3% → 39.7%**; other **60.7% → 60.3%**
  - monetary **2.2→1.8%**; `cbdc` **0.6%** (exploratory); regulation **6.3→6.6%**; ai_general 0.2%
  - VIF `AbnAtt_k` 7 core: regulation 1.47, defi 1.35, etf 1.26, stablecoins 1.21, institutional 1.17, monetary 1.11, hacks 1.08 (không >10)
- **Cách kiểm lại:** `pytest -q` → **39 passed**; `audit/narrative_samples_v4.md`
- **Chưa chắc / cần quyết:** —

## [2026-10-24] Narrative dict v3 (template price + thu hẹp FP)
- **Sửa gì / vì sao:** Cờ `is_template_price` (Price analysis/prediction* hoặc ≥4 ticker CSV) → chỉ other + price_desc; thu hẹp monetary/nft/regulation/etf/institutional/hacks; `ai_tokens`→`ai_general`; CBDC + OP_RETURN; excludes; samples v3 30+30 other; báo VIF/AbnAtt corr/overlap.
- **File:** `config/narratives.yaml`, `build_narratives.py`, `tests/test_narratives.py`, `audit/narrative_samples_v3.md`, narrative parquet/log
- **Số liệu trước → sau:**
  - `is_template_price`: **635** bài (1.01%); matched **45.4% → 39.3%**; other **54.6% → 60.7%**
  - core: regulation 6.3%, etf **6.3→4.2%**, institutional **3.3→2.9%**, hacks **2.1→2.0%**, defi 4.8%, stablecoins 3.1%, monetary **3.2→2.2%**
  - exploratory: nft **7.9→4.2%**, ai_general 0.2%, mining 4.5%, politics 3.8%, market_failure 3.4%, tech 3.2%
  - VIF trên `Att_k` 7 core (train 2021-03-31→2022-12-31): ~1.01–1.05
- **Cách kiểm lại:** `pytest -q` → **37 passed**; `audit/narrative_samples_v3.md`; log JSON (corr + overlap)
- **Chưa chắc / cần quyết:** —

## [2026-10-24] Narrative dict v2 (sửa FP + exploratory + price_desc)
- **Sửa gì / vì sao:** Thu hẹp breach/compound/circle; bổ sung institutional/monetary; thêm 5 narrative exploratory + cờ `price_desc`; VIF chỉ nhóm chính; samples v2 (30 title/narrative).
- **File:** `config/narratives.yaml`, `build_narratives.py`, `tests/test_narratives.py`, `audit/narrative_samples_v2.md`, narrative parquet/log
- **Số liệu trước → sau:**
  - matched: **25.9% → 45.4%**; other: **74.1% → 54.6%**
  - core: regulation 6.4%, etf 6.3%, defi 4.9%, institutional **2.8→3.3%**, monetary **2.8→3.2%**, hacks 2.1%, stablecoins ~3.1%
  - exploratory: nft_meme_web3 7.9%, mining 4.5%, politics 3.9%, market_failure 3.6%, tech_upgrade 3.2%, ai_tokens 0.3%
  - `price_desc` **29.9%** bài; VIF core ~1.01–1.04
- **Cách kiểm lại:** `pytest -q` → **33 passed**; `audit/narrative_samples_v2.md`
- **Chưa chắc / cần quyết:** —

## [2026-10-24] Narrative RQ3 (title, đa nhãn)
- **Sửa gì / vì sao:** Từ điển 8 narrative + `build_narratives.py` (doc/daily Att/LogRatio/AbnAtt, VIF train); samples kiểm tay.
- **File:** `config/narratives.yaml`, `build_narratives.py`, `config/paths.yaml`, `tests/test_narratives.py`, `audit/narrative_samples.md`, `data/interim/narrative_doc.parquet`, `data/processed/daily_narratives.parquet`
- **Số liệu trước → sau:**
  - doc 62 758; matched **25.9%**, other **74.1%**
  - share: regulation 6.4%, etf 6.3%, defi 4.9%, stablecoins 3.2%, institutional 2.8%, monetary 2.8%, hacks 2.1%, ai_tokens 0.3%
  - daily 4749 (=1583×3); VIF Att (train) đều ~1.0–1.05 (không cảnh báo >10)
- **Cách kiểm lại:** `pytest -q` → **31 passed**; đọc `audit/narrative_samples.md`
- **Chưa chắc / cần quyết:** ai_tokens coverage thấp (0.3%) — có thể mở rộng từ điển sau

## [2026-10-24] build_daily_panel (title + fulltext)
- **Sửa gì / vì sao:** Gộp sentiment → panel ngày×scope {all,BTC,ETH}; AbnAttention past-only; MeanProb; tách bản title / nlp_text.
- **File:** `build_daily_panel.py`, `config/paths.yaml`, `tests/test_panel_build.py`, `.cursor/rules/20-data-contracts.mdc`, `data/processed/daily_panel*.parquet`
- **Số liệu trước → sau:**
  - panel title & fulltext: **4749** dòng = 1583 ngày × 3 scope (2021-03-01→2025-06-30)
  - NewsVolume(all) tổng = **62758**; zero-news: all 4 / BTC 11 / ETH 13
- **Cách kiểm lại:** `pytest -q` → **27 passed**; log JSON cạnh parquet
- **Chưa chắc / cần quyết:** —

## [2026-10-24] Bản chính sentiment = title; fulltext = robustness
- **Sửa gì / vì sao:** Decrypt description rỗng lệch nguồn → bản chính score trên `title`; giữ `nlp_text` làm robustness. Slim thêm `title`; score `--text-col`; đổi tên kết quả Colab cũ → `sentiment_doc_fulltext.*` + `text_variant='nlp_text'`.
- **File:** `make_text_main_slim.py`, `score_sentiment.py`, `config/paths.yaml`, `.cursor/rules/50-nlp-features.mdc`, `20-data-contracts.mdc`, `docs/DATA_NOTES.md`, `tests/test_sentiment.py`, `data/interim/text_main_slim.parquet`, `sentiment_doc_fulltext*`
- **Số liệu trước → sau:**
  - slim: 62 758 dòng, cols=`doc_id,title,nlp_text` (13.1 MB)
  - fulltext (đổi tên từ sentiment_doc cũ): **125 516** dòng, `text_variant=nlp_text`, 62 758/model
  - `sentiment_doc.parquet` (title): chưa có — chờ Colab
  - smoke title VADER 20 dòng OK
- **Cách kiểm lại:** `pytest -q` → **17 passed, 5 skipped** (skip title main vì chưa file)
- **Chưa chắc / cần quyết:** —

## [2026-10-24] Nhận sentiment_doc từ Colab (T4) + pytest
- **Sửa gì / vì sao:** Đưa 3 file Colab vào `data/interim/` (`sentiment_doc.parquet`, `sentiment_doc_fullmeta.parquet`, `sentiment_doc.log.json`); kiểm tra và ghi audit.
- **File:** `data/interim/sentiment_doc*.parquet` + `.log.json` (không commit data)
- **Số liệu trước → sau:**
  - **125 516** dòng (= 62 758 × 2 mô hình); mỗi model đủ 62 758 `doc_id` unique
  - Nhãn VADER: pos 30 658 / neu 15 345 / neg 16 755
  - Nhãn FinBERT: pos 17 411 / neu 28 694 / neg 16 653
  - **truncated = 0** (cả hai model trong fullmeta)
  - Thời gian: VADER ~7.4 s (62 258 mới); FinBERT ~223.8 s (~3.7 phút) trên **Tesla T4**, batch=64, dtype=**fp32**
  - Môi trường: `torch 2.11.0+cu130`, `transformers 5.20.2`, FinBERT rev `0bc2ac10dda6…`
- **Cách kiểm lại:** `pytest -q` → **16 passed, 4 skipped**
- **Chưa chắc / cần quyết:** —

## [2026-10-24] Chuẩn bị Colab: slim + score flags/revision/fp32
- **Sửa gì / vì sao:** `make_text_main_slim.py` → `text_main_slim.parquet` (doc_id+nlp_text); `score_sentiment.py` thêm `--input/--output/--device/--batch-size`, sort theo độ dài trước batch, pin FinBERT revision, log torch/transformers, dtype=fp32.
- **File:** `make_text_main_slim.py`, `score_sentiment.py`, `config/paths.yaml`
- **Số liệu trước → sau:** text_clean 192.4 MB → slim **10.2 MB** / 62758 dòng; FinBERT rev `0bc2ac10dda6…`
- **Cách kiểm lại:** `python make_text_main_slim.py`; `python score_sentiment.py --input data/interim/text_main_slim.parquet --device cpu --limit 20 --models vader`
- **Chưa chắc / cần quyết:** —

## [2026-10-24] DGS10+2, market_end params, score_sentiment trial 500
- **Sửa gì / vì sao:** DGS10 `usable_from` +2 ngày; `sample.market_end` vào params; ghi chú FRED current-vintage; thêm `score_sentiment.py` (VADER+FinBERT, cache, truncate 512); trial 500.
- **File:** `build_market_daily.py`, `market_data.py`, `config/params.yaml`, `docs/DATA_NOTES.md`, `score_sentiment.py`, `tests/test_market.py`, `tests/test_sentiment.py`, `data/interim/fred.parquet` (DGS10), `sentiment_doc_trial500.parquet`
- **Số liệu trước → sau:** trial 500×2 model = 1000 dòng; FinBERT CPU ~286 ms/doc → ETA full ~5.0 giờ; VADER <1 ms/doc
- **Cách kiểm lại:** `pytest -q` → **16 passed, 4 skipped**; `python score_sentiment.py --limit 500 --out-suffix _trial500`
- **Chưa chắc / cần quyết:** dọn ổ C: để cài torch CUDA trước khi full 62758 (xem QUESTIONS #6)

## [2026-10-24] build_market_daily + FRED + tests
- **Sửa gì / vì sao:** Thêm `build_market_daily.py` (logic từ `market_data.py`): Binance 1h → `market_daily.parquet`; FRED VIXCLS/DGS10/DTWEXBGS → `fred.parquet` + `usable_from_utc`. Tests lịch/rv/abvol/FRED.
- **File:** `build_market_daily.py`, `config/paths.yaml` (+`fred_path`), `tests/test_market.py`, `tests/conftest.py`
- **Số liệu trước → sau:** market **4384** dòng = 2192×2 (BTC/ETH), `complete_day=False` **15/coin, total 30** (đúng 31 giờ thiếu); FRED 30700 dòng
- **Cách kiểm lại:** `pytest -q` → **14 passed, 4 skipped**; đối chiếu rv 3 ngày trong test
- **Chưa chắc / cần quyết:** Ảnh ghi "đổi volume_t không đổi abvol_t" nhưng công thức abvol có log(vol_t); test kiểm không nhìn tương lai (abvol_{t-1} bất biến). Binance 2026 để sau.

## [2026-10-24] Thu RSS mới + clean_rss; xác nhận text_clean DoD
- **Sửa gì / vì sao:** `rss_holdout.py` (thêm 2026-10-24) rồi `clean_rss.py`; đối chiếu text_clean theo checklist ảnh.
- **File:** `data/raw/rss_holdout/2026-10-24.jsonl`, `data/interim/rss_holdout.parquet` (+ log); text_clean không đổi
- **Số liệu trước → sau:** RSS clean 171 → **213** dòng (4 file jsonl). text_clean vẫn **82741** / **62758** in_main_sample; dump cryptonews 2023-06-26 & 2023-10-23
- **Cách kiểm lại:** đọc `text_clean.log.json` → `dump_day_keys`; `pd.read_parquet(...).shape`
- **Chưa chắc / cần quyết:** —

## [2026-10-24] Task ảnh: day_utc + gitignore + chạy clean_merge
- **Sửa gì / vì sao:** Đổi `_day` → `day_utc` trong `step_flag_main_sample` (itertuples); thêm `data/` + `cursor_pack/` vào `.gitignore`; `.gitattributes` `* text=auto`; bỏ em-dash trong comment `clean_rss.py`; chạy lại `clean_merge.py` trên CSV thật.
- **File:** `clean_merge.py`, `clean_rss.py`, `.gitignore`, `.gitattributes`
- **Số liệu trước → sau:** `text_clean.parquet` 82741 dòng; `2020-01-01`→`2025-06-30`; dump cryptonews 26/6 & 23/10/2023 (1440 dòng bỏ); junk 13, short 133, exact dup title 1590; main_sample True 62758 / False 21423
- **Cách kiểm lại:** `pytest -q` → **7 passed, 4 skipped**; `git status` không còn `data/`
- **Chưa chắc / cần quyết:** panel tests vẫn skip (chưa `daily_panel.parquet`)

## [2026-10-24] Cài cursor_pack + sửa TV2 theo FINDINGS (chưa có raw CV)
- **Sửa gì / vì sao:** Cài pack vào gốc repo; chặn commit/push; sửa `clean_merge.py` / `clean_rss.py` theo `audit/FINDINGS.md` (khung mẫu từ `params.yaml`, `news_date`, `in_main_sample`, dump/cryptonews, junk/short title, `nlp_text` thống nhất, dedup RSS có chữ số, tag BTC/ETH, thu hẹp `crypto_relevant`).
- **File:** `.cursor/` (rules, commands, permissions.json), `config/`, `tests/`, `audit/`, `tools/`, `AGENTS.md`, `pytest.ini`, `docs/README.md`, `clean_merge.py`, `clean_rss.py`, `data/raw/rss_holdout` (junction), `data/interim/rss_holdout.parquet`
- **Số liệu trước → sau:**
  - pytest baseline: `1 passed, 10 skipped` (thiếu `text_clean.parquet` / `daily_panel.parquet`)
  - `clean_rss.py` trên holdout hiện có: 212 → 171 dòng; off_topic_dropped=21; dump title dedup digits bật
  - CryptoVision: chưa chạy (thiếu CSV) → text/panel tests vẫn skip
- **Cách kiểm lại:** `pytest -q`; khi có CSV: `python clean_merge.py` rồi `pytest -q` (các test text phải xanh)
- **Chưa chắc / cần quyết:** xem `audit/QUESTIONS.md` (docs ×3, raw CV, market/panel/Binance)
