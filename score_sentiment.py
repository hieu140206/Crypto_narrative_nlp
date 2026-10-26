"""score_sentiment.py - Điểm sentiment cấp bài (TV3).

Bản CHÍNH: --text-col title (mặc định) → sentiment_doc.parquet.
Robustness: --text-col nlp_text → sentiment_doc_fulltext.parquet.

Đầu ra: doc_id, model, label, p_pos, p_neg, p_neu, text_variant.
Cache: (doc_id, model, checkpoint, text_variant). FinBERT pin revision, fp32,
sort theo độ dài trước batch; log torch/transformers.

Cách chạy:
  python score_sentiment.py --device cuda --batch-size 64 --text-col title
  python score_sentiment.py --text-col nlp_text   # robustness fulltext
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)

FINBERT_ID = "ProsusAI/finbert"
FINBERT_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
FINBERT_CHECKPOINT = f"{FINBERT_ID}@{FINBERT_REVISION[:12]}"
FINBERT_MAX_LEN = 512
VADER_CHECKPOINT = "vader_lexicon_nltk"
CACHE_DIR_NAME = "sentiment_cache"
TEXT_COLS = ("title", "nlp_text")


def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "config").exists():
            return p
    return here.parent


def load_config(root: Path) -> tuple[dict, dict]:
    import yaml
    params = yaml.safe_load((root / "config" / "params.yaml").read_text(encoding="utf-8")) or {}
    paths = yaml.safe_load((root / "config" / "paths.yaml").read_text(encoding="utf-8")) or {}
    return params, paths


def abs_path(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def cache_path(cache_dir: Path, model: str, checkpoint: str, text_variant: str) -> Path:
    safe = checkpoint.replace("/", "_").replace("\\", "_").replace("@", "_")
    return cache_dir / f"{model}__{safe}__{text_variant}.parquet"


def load_cache(path: Path) -> pd.DataFrame:
    cols = ["doc_id", "model", "checkpoint", "text_variant", "label",
            "p_pos", "p_neg", "p_neu", "truncated"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_parquet(path)
    if "text_variant" not in df.columns:
        df["text_variant"] = "nlp_text"  # cache cũ = fulltext
    return df


def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def sort_by_text_length(docs: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = docs.copy()
    out["_len"] = out[text_col].fillna("").astype(str).str.len()
    return out.sort_values("_len", kind="mergesort").drop(columns=["_len"]).reset_index(drop=True)


def score_vader(texts: list[str], doc_ids: list[str], text_variant: str) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for doc_id, text in zip(doc_ids, texts):
        s = analyzer.polarity_scores(text or "")
        compound = s["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        p_pos, p_neg, p_neu = float(s["pos"]), float(s["neg"]), float(s["neu"])
        z = p_pos + p_neg + p_neu
        if z > 0:
            p_pos, p_neg, p_neu = p_pos / z, p_neg / z, p_neu / z
        rows.append({
            "doc_id": doc_id,
            "model": "vader",
            "checkpoint": VADER_CHECKPOINT,
            "text_variant": text_variant,
            "label": label,
            "p_pos": p_pos,
            "p_neg": p_neg,
            "p_neu": p_neu,
            "truncated": False,
        })
    return pd.DataFrame(rows)


def score_finbert(
    texts: list[str],
    doc_ids: list[str],
    text_variant: str,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[pd.DataFrame, int, dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import transformers

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    env = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": "fp32",
        "finbert_id": FINBERT_ID,
        "finbert_revision": FINBERT_REVISION,
        "text_variant": text_variant,
    }
    print(f"  FinBERT device={device}" + (
        f" ({env['cuda_name']})" if env["cuda_name"] else ""
    ))
    print(f"  torch={env['torch']} transformers={env['transformers']} dtype=fp32 "
          f"revision={FINBERT_REVISION[:12]} text_variant={text_variant}")

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_ID, revision=FINBERT_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_ID, revision=FINBERT_REVISION, torch_dtype=torch.float32
    )
    model.to(device)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    n_trunc = 0
    rows = []
    for i in range(0, len(texts), batch_size):
        batch_texts = [t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
        batch_ids = doc_ids[i:i + batch_size]
        enc = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=FINBERT_MAX_LEN, return_tensors="pt",
        )
        lengths = tokenizer(
            batch_texts, truncation=False, padding=False, return_length=True
        )["length"]
        truncated_flags = [L > FINBERT_MAX_LEN for L in lengths]
        n_trunc += sum(truncated_flags)

        input_ids = enc["input_ids"].to(device)
        attention = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for j, doc_id in enumerate(batch_ids):
            p = probs[j]
            p_map = {id2label[k]: float(p[k]) for k in range(len(p))}
            p_pos = p_map.get("positive", 0.0)
            p_neg = p_map.get("negative", 0.0)
            p_neu = p_map.get("neutral", 0.0)
            label = max(
                [("positive", p_pos), ("negative", p_neg), ("neutral", p_neu)],
                key=lambda x: x[1],
            )[0]
            rows.append({
                "doc_id": doc_id,
                "model": "finbert",
                "checkpoint": FINBERT_CHECKPOINT,
                "text_variant": text_variant,
                "label": label,
                "p_pos": p_pos,
                "p_neg": p_neg,
                "p_neu": p_neu,
                "truncated": bool(truncated_flags[j]),
            })
        if (i // batch_size + 1) % 20 == 0 or i + batch_size >= len(texts):
            print(f"  FinBERT {min(i + batch_size, len(texts))}/{len(texts)}")
    return pd.DataFrame(rows), n_trunc, env


def merge_cache(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new
    keep = existing[~existing["doc_id"].isin(new["doc_id"])]
    return pd.concat([keep, new], ignore_index=True)


def run_model(
    name: str,
    docs: pd.DataFrame,
    cache_dir: Path,
    batch_size: int,
    device: str | None,
    text_col: str,
) -> tuple[pd.DataFrame, dict]:
    text_variant = text_col  # 'title' | 'nlp_text'
    checkpoint = VADER_CHECKPOINT if name == "vader" else FINBERT_CHECKPOINT
    path = cache_path(cache_dir, name, checkpoint, text_variant)
    cached = load_cache(path)
    # chỉ dùng cache cùng text_variant
    if len(cached) and "text_variant" in cached.columns:
        cached = cached[cached["text_variant"] == text_variant]
    done = set(cached["doc_id"].astype(str)) if len(cached) else set()
    todo = docs[~docs["doc_id"].astype(str).isin(done)].reset_index(drop=True)
    if len(todo):
        todo = sort_by_text_length(todo, text_col)
    meta = {
        "model": name, "checkpoint": checkpoint, "text_variant": text_variant,
        "cached": len(done), "todo": len(todo), "truncated_new": 0,
        "sorted_by_length": True,
    }
    print(f"[{name}/{text_variant}] cache={len(done)} todo={len(todo)} -> {path.name}")
    if todo.empty:
        return cached, meta

    texts = todo[text_col].fillna("").astype(str).tolist()
    ids = todo["doc_id"].astype(str).tolist()
    t0 = time.perf_counter()
    if name == "vader":
        scored = score_vader(texts, ids, text_variant)
    elif name == "finbert":
        scored, n_trunc, env = score_finbert(
            texts, ids, text_variant, batch_size=batch_size, device=device
        )
        meta["truncated_new"] = n_trunc
        meta["env"] = env
    else:
        raise ValueError(name)
    elapsed = time.perf_counter() - t0
    meta["seconds"] = elapsed
    meta["per_doc_sec"] = elapsed / max(len(todo), 1)
    print(f"  scored {len(todo)} in {elapsed:.1f}s ({meta['per_doc_sec']*1000:.1f} ms/doc)")

    out = merge_cache(cached, scored)
    save_cache(out, path)
    return out, meta


def load_docs(text_path: Path, text_col: str, limit: int | None, seed: int) -> pd.DataFrame:
    import pyarrow.parquet as pq
    schema_cols = set(pq.read_schema(text_path).names)
    need = ["doc_id", text_col]
    if text_col not in schema_cols:
        sys.exit("File %s thiếu cột %s (có: %s)" % (text_path, text_col, sorted(schema_cols)))
    read_cols = ["doc_id", text_col]
    # giữ cột kia nếu có (không bắt buộc)
    for extra in ("title", "nlp_text"):
        if extra in schema_cols and extra not in read_cols:
            read_cols.append(extra)
    if "in_main_sample" in schema_cols:
        df = pd.read_parquet(text_path, columns=read_cols + ["in_main_sample"])
        docs = df.loc[df["in_main_sample"]].drop(columns=["in_main_sample"]).copy()
        print(f"input={text_path.name} in_main_sample={len(docs)} text_col={text_col}")
    else:
        docs = pd.read_parquet(text_path, columns=read_cols).copy()
        print(f"input={text_path.name} rows={len(docs)} text_col={text_col} (slim)")
    docs = docs.reset_index(drop=True)
    if limit is not None:
        docs = docs.sample(n=min(limit, len(docs)), random_state=seed)
        docs = docs.sort_values("doc_id").reset_index(drop=True)
        print(f"limit={limit} -> {len(docs)} bài")
    return docs


def default_out_path(root: Path, paths: dict, text_col: str) -> Path:
    if text_col == "nlp_text":
        rel = paths.get("sentiment_doc_fulltext_path", "data/interim/sentiment_doc_fulltext.parquet")
    else:
        rel = paths["sentiment_doc_path"]
    return abs_path(root, rel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score article-level sentiment")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-col", default="title", choices=list(TEXT_COLS),
                        help="title=bản chính; nlp_text=robustness fulltext")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=["vader", "finbert"],
                        choices=["vader", "finbert"])
    parser.add_argument("--out-suffix", default="")
    args = parser.parse_args()

    root = project_root()
    params, paths = load_config(root)
    seed = params.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    text_col = args.text_col
    text_variant = text_col

    slim = abs_path(root, paths.get("text_main_slim_path", "data/interim/text_main_slim.parquet"))
    clean = abs_path(root, paths["text_clean_path"])
    if args.input:
        text_path = abs_path(root, args.input)
    elif slim.exists():
        text_path = slim
    else:
        text_path = clean

    out_path = abs_path(root, args.output) if args.output else default_out_path(root, paths, text_col)
    if args.out_suffix:
        out_path = out_path.with_name(out_path.stem + args.out_suffix + out_path.suffix)
    cache_dir = abs_path(root, "data/interim") / CACHE_DIR_NAME

    docs = load_docs(text_path, text_col, args.limit, seed)

    env_log = {
        "finbert_revision": FINBERT_REVISION, "dtype": "fp32",
        "text_col": text_col, "text_variant": text_variant,
    }
    try:
        import torch
        import transformers
        env_log["torch"] = torch.__version__
        env_log["transformers"] = transformers.__version__
        env_log["cuda_available"] = torch.cuda.is_available()
    except ImportError as exc:
        env_log["import_error"] = str(exc)
    print(f"env torch={env_log.get('torch')} transformers={env_log.get('transformers')} "
          f"fp32 text_col={text_col} finbert@{FINBERT_REVISION[:12]}")

    all_parts = []
    log = {
        "n_docs": len(docs), "models": {}, "limit": args.limit,
        "input": str(text_path), "device": args.device, "batch_size": args.batch_size,
        "text_col": text_col, "text_variant": text_variant, "env": env_log,
    }
    for name in args.models:
        part, meta = run_model(name, docs, cache_dir, args.batch_size, args.device, text_col)
        part = part[part["doc_id"].astype(str).isin(set(docs["doc_id"].astype(str)))]
        all_parts.append(part)
        log["models"][name] = meta

    result = pd.concat(all_parts, ignore_index=True)
    if "text_variant" not in result.columns:
        result["text_variant"] = text_variant
    out_cols = ["doc_id", "model", "label", "p_pos", "p_neg", "p_neu", "text_variant"]
    export = result[out_cols].copy()

    z = export["p_pos"] + export["p_neg"] + export["p_neu"]
    export["p_pos"] = export["p_pos"] / z
    export["p_neg"] = export["p_neg"] / z
    export["p_neu"] = export["p_neu"] / z
    assert (export["p_pos"] + export["p_neg"] + export["p_neu"] - 1.0).abs().max() < 1e-6
    assert (export["text_variant"] == text_variant).all()
    for m, g in export.groupby("model"):
        assert g["doc_id"].is_unique, m
        assert len(g) == len(docs), (m, len(g), len(docs))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_parquet(out_path, index=False)
    result.to_parquet(out_path.with_name(out_path.stem + "_fullmeta.parquet"), index=False)

    if "finbert" in log["models"] and log["models"]["finbert"].get("per_doc_sec"):
        rate = log["models"]["finbert"]["per_doc_sec"]
        log["finbert_eta_full_hours"] = rate * 62758 / 3600
        print(f"Ước FinBERT full 62758 ≈ {log['finbert_eta_full_hours']:.2f} giờ")

    log["rows_out"] = len(export)
    log["out_path"] = str(out_path)
    log["label_counts"] = {
        m: export.loc[export["model"] == m, "label"].value_counts().to_dict()
        for m in export["model"].unique()
    }
    log_path = out_path.with_suffix(".log.json")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Da luu {len(export)} dong -> {out_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
