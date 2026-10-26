pip install -q vaderSentiment transformers torch
import random
import numpy as np
import torch
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cố định Seed theo quy ước dự án
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class SentimentScorer:
    """Mô-đun quản lý suy diễn Sentiment cho VADER, FinBERT và CryptoBERT"""
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        print(f"=== SentimentScorer đang hoạt động trên thiết bị: {self.device} ===")
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.finbert_labels = {0: 'positive', 1: 'negative', 2: 'neutral'}
        self.cryptobert_labels = {0: 'negative', 1: 'positive', 2: 'neutral'}
        
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.cryptobert_tokenizer = None
        self.cryptobert_model = None

    def get_vader_scores(self, texts: list) -> list:
        """Tính compound score [-1, 1] của VADER"""
        results = []
        for text in texts:
            if not text or pd.isna(text):
                results.append(0.0)
            else:
                results.append(self.vader_analyzer.polarity_scores(str(text))['compound'])
        return results

    def _load_finbert(self):
        if self.finbert_model is None:
            print("Đang khởi tạo FinBERT (ProsusAI/finbert)...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
            self.finbert_model.eval()

    def _load_cryptobert(self):
        if self.cryptobert_model is None:
            print("Đang khởi tạo CryptoBERT (ElKulako/cryptobert)...")
            self.cryptobert_tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
            self.cryptobert_model = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert").to(self.device)
            self.cryptobert_model.eval()

    def predict_transformer_batch(self, texts: list, model_type: str = "finbert", batch_size: int = 16):
        """Suy diễn batch cho Transformer models"""
        if model_type == "finbert":
            self._load_finbert()
            tokenizer, model, labels_map = self.finbert_tokenizer, self.finbert_model, self.finbert_labels
        elif model_type == "cryptobert":
            self._load_cryptobert()
            tokenizer, model, labels_map = self.cryptobert_tokenizer, self.cryptobert_model, self.cryptobert_labels
        else:
            raise ValueError("model_type phải là 'finbert' hoặc 'cryptobert'")

        results_label = []
        results_score = []
        net_scores = []

        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) if t else "" for t in texts[i:i + batch_size]]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=-1)
            scores, indices = torch.max(probs, dim=-1)
            
            probs_np = probs.cpu().numpy()
            for idx, score, p in zip(indices.cpu().numpy(), scores.cpu().numpy(), probs_np):
                label = labels_map[idx]
                results_label.append(label)
                results_score.append(float(score))
                
                # Net score calculation
                pos_p = p[0] if (model_type == 'finbert' and idx == 0) else (p[1] if (model_type == 'cryptobert' and idx == 1) else 0.0)
                neg_p = p[1] if (model_type == 'finbert' and idx == 1) else (p[0] if (model_type == 'cryptobert' and idx == 0) else 0.0)
                net_scores.append(float(pos_p - neg_p))

        return results_label, results_score, net_scores
