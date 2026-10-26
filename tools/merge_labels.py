"""Gộp nhãn của A và B, tính Cohen's kappa và tỷ lệ bất đồng.

Chạy:  python tools/merge_labels.py label_A_done.csv label_B_done.csv labeling/label_key.csv out.csv [pilot]
- Nếu thêm tham số cuối là 'pilot' thì chỉ tính trên lô pilot (45 dòng đầu).
Ghi out.csv gồm id, label_A, label_B, pricemove_A, pricemove_B, disagree (để thảo luận, điền label_final sau).
"""
import sys
import pandas as pd

LABELS = ["negative", "neutral", "positive"]

def kappa(a, b, cats):
    a, b = list(a), list(b)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

A = pd.read_csv(sys.argv[1], encoding="utf-8-sig"); B = pd.read_csv(sys.argv[2], encoding="utf-8-sig")
out_path = sys.argv[4]
pilot = len(sys.argv) > 5 and sys.argv[5] == "pilot"
m = A[["id", "batch", "headline", "label", "pricemove"]].merge(
    B[["id", "label", "pricemove"]], on="id", suffixes=("_A", "_B"))
if pilot:
    m = m[m["batch"] == "pilot"]
for c in ("label_A", "label_B"):
    m[c] = m[c].astype(str).str.strip().str.lower()
bad = m[~m["label_A"].isin(LABELS) | ~m["label_B"].isin(LABELS)]
if len(bad):
    sys.exit(f"{len(bad)} dòng thiếu/sai nhãn (chỉ dùng {LABELS}). Ví dụ id: {bad['id'].head(5).tolist()}")
m["disagree"] = m["label_A"] != m["label_B"]
pm = m[["pricemove_A", "pricemove_B"]].astype(int)
print(f"n={len(m)}  đồng ý={1 - m['disagree'].mean():.3f}  kappa nhãn={kappa(m['label_A'], m['label_B'], LABELS):.3f}"
      f"  kappa pricemove={kappa(pm['pricemove_A'], pm['pricemove_B'], [0, 1]):.3f}")
print(pd.crosstab(m["label_A"], m["label_B"]).to_string())
m.assign(label_final="").to_csv(out_path, index=False, encoding="utf-8-sig")
print("ghi", out_path)
