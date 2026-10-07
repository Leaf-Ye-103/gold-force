import io
import pandas as pd
from .ledger import records, numeric, end_of_day

DEFAULT_SERIES = {
    "real_yield": ("DFII10", "percent", 2),
    "dollar": ("DTWEXBGS", "broad_dollar_index", 10),
    "vix": ("VIXCLS", "index", 2),
    "stocks": ("SP500", "index", 2),
}


def load(client, cfg, start, end, refresh=False):
    frames = []
    for name, (default_id, unit, default_delay) in DEFAULT_SERIES.items():
        item = cfg.get(name, {})
        if item.get("enabled", True) is False:
            continue
        series_id = item.get("id", default_id)
        body, meta = client.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                                {"id": series_id, "cosd": start, "coed": end}, refresh)
        f = pd.read_csv(io.BytesIO(body), na_values=["."])
        if len(f.columns) != 2 or series_id not in f:
            raise ValueError(f"FRED schema changed for {series_id}")
        dates = pd.to_datetime(f.iloc[:, 0])
        values = numeric(f[series_id])
        mask = dates.between(start, end) & values.notna()
        if not mask.any():
            raise ValueError(f"No FRED observations: {series_id}")
        # H.10 broad dollar daily observations are released in weekly batches.
        # Ten days is an explicit approximation, not DXY and not vintage data.
        available = end_of_day(dates[mask], item.get("assumed_delay_days", default_delay))
        frames.append(records(name, dates[mask], values[mask], available, unit,
                              "FRED:" + series_id, meta["retrieved_at"], "assumed_fred_delay"))
    return pd.concat(frames, ignore_index=True)
