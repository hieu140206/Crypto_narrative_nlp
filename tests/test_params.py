import pandas as pd

def test_splits_ordered_and_embargo(params):
    s = params["split"]; e = params["split"]["embargo_days"]
    tr, va, te = (pd.Timestamp(s[k][1]) for k in ("train", "validation", "test"))
    assert pd.Timestamp(s["train"][1]) < pd.Timestamp(s["validation"][0])
    assert pd.Timestamp(s["validation"][1]) < pd.Timestamp(s["test"][0])
    assert e >= max(params["horizons"])
    assert pd.Timestamp(s["test"][1]) == pd.Timestamp(params["sample"]["t_end"])
    assert pd.Timestamp(params["sample"]["start"]) == pd.Timestamp(s["train"][0])
