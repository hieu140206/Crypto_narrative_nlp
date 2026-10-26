import glob
import json
import os
import pandas as pd
import numpy as np

from sentiment_features import SentimentScorer
from narrative_features import extract_narrative_flags, extract_top_keywords
from attention_features import calculate_daily_metrics

def main():
    print("=== 1. Tải và chuẩn hóa Dữ liệu Text ===")
    file_paths = glob.glob('rss_holdout/*.jsonl') + glob.glob('data/raw/*.jsonl') + glob.glob('/content/Crypto_narrative_nlp/rss_holdout/*.jsonl')
    all_data = []

    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))

    if not all_data:
        print("Không tìm thấy dữ liệu JSONL thô!")
        return

    df = pd.DataFrame(all_data)
    print(f"Đã tải {len(df)} bài báo.")

    # Xử lý text và chuẩn hóa ngày UTC
    text_col = 'title' if 'title' in df.columns else df.columns[0]
    df['clean_text'] = df[text_col].fillna("").astype(str)

    if 'published_parsed' in df.columns:
        df['date'] = df['published_parsed'].apply(
            lambda x: pd.to_datetime(x[:10]) if isinstance(x, str) else pd.to_datetime('2026-10-22')
        )
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.floor('D')
    else:
        df['date'] = pd.to_datetime('2026-10-22')

    df_sample = df.copy()

    # 2. EDA: Trích xuất Keyword nổi bật
    print("=== Top Narrative Keywords trong Dữ liệu ===")
    top_keywords = extract_top_keywords(df_sample['clean_text'], top_n=15)
    for kw, count in top_keywords:
        print(f"- {kw}: {count} lần")

    # 3. Tính điểm Sentiment (VADER / FinBERT / CryptoBERT)
    print("=== 2. Chạy Sentiment Inference ===")
    scorer = SentimentScorer(use_gpu=True)
    texts_list = df_sample['clean_text'].tolist()

    # VADER
    df_sample['vader_score'] = scorer.get_vader_scores(texts_list)
    df_sample['vader_label'] = ['positive' if s >= 0.05 else ('negative' if s <= -0.05 else 'neutral') for s in df_sample['vader_score']]

    # FinBERT
    print("Running FinBERT...")
    df_sample['finbert_label'], df_sample['finbert_score'], _ = scorer.predict_transformer_batch(
        texts_list, model_type="finbert", batch_size=16
    )

    # CryptoBERT
    print("Running CryptoBERT...")
    df_sample['cryptobert_label'], df_sample['cryptobert_score'], _ = scorer.predict_transformer_batch(
        texts_list, model_type="cryptobert", batch_size=16
    )

    # 4. Gom nhóm Chỉ số Sentiment & Abnormal Attention theo Ngày
    print("=== 3. Tổng hợp Daily Sentiment & Attention Metrics ===")
    vader_metrics = calculate_daily_metrics(df_sample, 'vader')
    finbert_metrics = calculate_daily_metrics(df_sample, 'finbert')
    cryptobert_metrics = calculate_daily_metrics(df_sample, 'cryptobert')

    merged_metrics = vader_metrics.merge(finbert_metrics, on='date').merge(cryptobert_metrics, on='date')

    # 5. Gom nhóm Tỷ trọng Chủ đề (Narrative Attention Share)
    print("=== 4. Phân loại và Gom nhóm Narrative Share ===")
    df_narratives = extract_narrative_flags(df_sample, text_col='clean_text')
    narrative_cols = [c for c in df_narratives.columns if c.startswith('is_')]
    
    narrative_daily = df_narratives.groupby('date')[narrative_cols].mean().reset_index()
    for col in narrative_cols:
        clean_name = col.replace('is_', 'attention_')
        narrative_daily.rename(columns={col: clean_name}, inplace=True)

    # Ghép bảng chỉ số cuối cùng
    daily_panel = merged_metrics.merge(narrative_daily, on='date', how='left')

    # 6. Trích xuất file Parquet
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/daily_panel.parquet'
    daily_panel.to_parquet(output_path, index=False)
    print(f"=== HOÀN THÀNH: Xuất file thành công tại {output_path} ({len(daily_panel)} hàng, {len(daily_panel.columns)} cột) ===")

if __name__ == "__main__":
    main()