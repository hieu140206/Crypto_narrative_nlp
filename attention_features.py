import pandas as pd
import numpy as np

def calculate_daily_metrics(df_data: pd.DataFrame, prefix: str, window: int = 30) -> pd.DataFrame:
    """
    Tính các chỉ số theo ngày không bị look-ahead bias:
    - Sent_t, PosSent, NegSent, NewsVolume_t, AbnAttention_t
    """
    df_sorted = df_data.sort_values('date').copy()
    grouped = df_sorted.groupby('date')

    daily_data = []
    label_col = f'{prefix}_label'

    for date, group in grouped:
        total_docs = len(group)
        pos_docs = len(group[group[label_col] == 'positive']) if label_col in group.columns else 0
        neg_docs = len(group[group[label_col] == 'negative']) if label_col in group.columns else 0

        sent_t = (pos_docs - neg_docs) / total_docs if total_docs > 0 else 0.0
        pos_ratio = pos_docs / total_docs if total_docs > 0 else 0.0
        neg_ratio = neg_docs / total_docs if total_docs > 0 else 0.0

        daily_data.append({
            'date': date,
            f'{prefix}_Sent_t': sent_t,
            f'{prefix}_PosSent': pos_ratio,
            f'{prefix}_NegSent': neg_ratio,
            f'{prefix}_NewsVolume_t': total_docs
        })

    metrics_df = pd.DataFrame(daily_data)
    if metrics_df.empty:
        return metrics_df

    metrics_df = metrics_df.sort_values('date').reset_index(drop=True)

    # Rolling Z-Score với shift(1) để triệt tiêu look-ahead bias
    vol_col = f'{prefix}_NewsVolume_t'
    mean_30 = metrics_df[vol_col].shift(1).rolling(window=window, min_periods=1).mean()
    sd_30 = metrics_df[vol_col].shift(1).rolling(window=window, min_periods=1).std().fillna(1.0)
    sd_30 = sd_30.replace(0, 1.0)

    metrics_df[f'{prefix}_AbnAttention_t'] = (metrics_df[vol_col] - mean_30) / sd_30
    metrics_df[f'{prefix}_AbnAttention_t'] = metrics_df[f'{prefix}_AbnAttention_t'].fillna(0.0)

    return metrics_df