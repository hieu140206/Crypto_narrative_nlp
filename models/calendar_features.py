"""Biến lịch dùng chung (không ghi vào parquet).

DOW: dayofweek của ngày gốc t (UTC/naive date), Monday=0.
Bỏ Monday → 6 biến giả Tue…Sun.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# pandas dayofweek: Mon=0 … Sun=6; bỏ Mon
DOW_DUMMY_COLS: tuple[str, ...] = (
    "dow_Tue", "dow_Wed", "dow_Thu", "dow_Fri", "dow_Sat", "dow_Sun",
)


def dow_dummy_frame(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """6 cột giả từ dates; chỉ phụ thuộc ngày t (không nhìn tương lai)."""
    # Series has .dt; DatetimeIndex exposes dayofweek/tz directly
    if isinstance(dates, pd.DatetimeIndex):
        d = dates
        if d.tz is not None:
            d = d.tz_convert("UTC").tz_localize(None)
        dow = d.dayofweek
        # positional index — DatetimeIndex labels would break dum.loc[i]
        idx = pd.RangeIndex(len(d))
    else:
        d = pd.to_datetime(pd.Series(dates))
        if getattr(d.dt, "tz", None) is not None:
            d = d.dt.tz_convert("UTC").dt.tz_localize(None)
        dow = d.dt.dayofweek
        idx = getattr(dates, "index", None)
    dow = np.asarray(dow)
    data = {name: (dow == i).astype(np.float64) for i, name in enumerate(DOW_DUMMY_COLS, start=1)}
    return pd.DataFrame(data, index=idx)


def add_dow_dummies(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Gắn 6 biến giả DOW vào bản sao df (không sửa parquet nguồn)."""
    out = df.copy()
    dum = dow_dummy_frame(out[date_col])
    for c in DOW_DUMMY_COLS:
        out[c] = dum[c].to_numpy()
    return out


def deseasonalize_by_dow_past_only(
    values: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    """Khử mùa DOW: value_t − mean({value_s: same weekday, s < t}).

    Chỉ dùng quá khứ nghiêm (đến t−1). Ngày chưa có quá khứ cùng thứ → NaN.
    Biến phụ / độ bền — ngoài PREREG.
    """
    d = pd.to_datetime(pd.Series(dates))
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        # naive: tz_localize(None) is no-op if already naive; strip if localized via object
        try:
            d = d.dt.tz_localize(None)
        except TypeError:
            pass
    d = d.dt.normalize()
    v = values.astype(float).to_numpy()
    dow = d.dt.dayofweek.to_numpy()
    order = np.argsort(d.to_numpy())
    out = np.full(len(v), np.nan, dtype=float)
    hist: list[list[float]] = [[] for _ in range(7)]
    for i in order:
        w = int(dow[i])
        if hist[w]:
            out[i] = v[i] - float(np.mean(hist[w]))
        if np.isfinite(v[i]):
            hist[w].append(float(v[i]))
    return pd.Series(out, index=values.index, name=getattr(values, "name", None))
