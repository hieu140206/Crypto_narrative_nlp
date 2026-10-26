import re
import yaml
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Tải tài nguyên stopwords
nltk.download('stopwords', quiet=True)

def load_narrative_keywords(config_path: str = "config/narrative_keywords.yaml") -> dict:
    """Đọc file từ khóa YAML"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_top_keywords(texts: list, top_n: int = 15, custom_stops: set = None) -> list:
    """Trích xuất từ khóa tần suất cao phục vụ EDA"""
    stop_words = set(stopwords.words('english'))
    default_crypto_stops = {'crypto', 'cryptocurrency', 'bitcoin', 'btc', 'eth', 'ethereum', 'amp', 'u', 'new', 'news'}
    if custom_stops:
        default_crypto_stops.update(custom_stops)
    stop_words.update(default_crypto_stops)

    words = []
    for text in texts:
        if not text or pd.isna(text):
            continue
        cleaned = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        words.extend([w for w in cleaned.split() if w not in stop_words and len(w) > 3])

    counter = Counter(words)
    return counter.most_common(top_n)

def extract_narrative_flags(df: pd.DataFrame, text_col: str = "clean_text", config_path: str = "config/narrative_keywords.yaml") -> pd.DataFrame:
    """Tạo flag 0/1 cho 6 chủ đề Narrative"""
    keywords_dict = load_narrative_keywords(config_path)
    df_out = df.copy()

    for category, keywords in keywords_dict.items():
        pattern = r'\b(' + '|'.join([re.escape(k) for k in keywords]) + r')\b'
        df_out[f"is_{category}"] = df_out[text_col].str.lower().str.contains(pattern, regex=True, na=False).astype(int)

    return df_out