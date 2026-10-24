"""Tách label_task.csv thành 2 file riêng cho 2 người gán (blind, chỉ headline).

Chạy:  python tools/split_label_files.py labeling/label_task.csv out_dir
Tạo:   out_dir/label_A.csv  (id,batch,headline,assets,label,pricemove)
       out_dir/label_B.csv  (cùng cột)
Mỗi người chỉ nhận file của mình; không gửi label_key.csv, không gửi cột của người kia.
"""
import sys
from pathlib import Path
import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
t = pd.read_csv(src, encoding="utf-8-sig")
base = t[["id", "batch", "headline", "assets"]].copy()
for who in ("A", "B"):
    f = base.copy()
    f["label"] = ""
    f["pricemove"] = ""
    f.to_csv(out / f"label_{who}.csv", index=False, encoding="utf-8-sig")
print(f"{len(t)} dòng -> {out}/label_A.csv, label_B.csv")
