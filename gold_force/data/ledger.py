from pathlib import Path
import numpy as np
import pandas as pd

COLS = ["series", "observation_date", "available_at", "value", "unit", "source",
        "vintage_at", "availability_basis"]
EXPECTED_UNITS = {
    "gold_price": "usd_per_troy_oz", "gld_price": "usd_per_share", "gld_holdings": "tonnes", "global_etf_flow": "tonnes",
    "cb_purchase": "tonnes", "bar_coin": "tonnes", "etf_demand": "tonnes",
    "mine_supply": "tonnes", "recycling": "tonnes", "otc_demand": "tonnes",
    "open_interest": "contracts", "real_yield": "percent", "vix": "index", "stocks": "index",
    "dollar": "broad_dollar_index",
    **{g+"_net_pct_oi": "percent_oi" for g in ("mm", "swap", "producer", "other", "nonreportable")},
}


def utc(value):
    return pd.to_datetime(value, utc=True)


def end_of_day(dates, days=0):
    """Explicit NY timezone (including DST); this is a policy, not verified releases."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize() + pd.Timedelta(days=days)
    return (dates + pd.Timedelta(hours=23, minutes=59)).tz_localize("America/New_York").tz_convert("UTC")


def numeric(values):
    return pd.to_numeric(values.astype(str).str.replace(",", "", regex=False)
                         .str.replace("$", "", regex=False).str.strip(), errors="coerce")


def records(series, dates, values, available, unit, source, vintage, basis):
    frame = pd.DataFrame({"series": series, "observation_date": pd.to_datetime(dates),
                          "available_at": utc(available), "value": np.asarray(values),
                          "unit": unit, "source": source, "vintage_at": utc(vintage),
                          "availability_basis": basis})
    return frame[COLS].dropna(subset=["value"])


def validate_ledger(frame):
    missing = set(COLS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing ledger fields: {sorted(missing)}")
    f = frame[COLS].copy()
    f["observation_date"] = pd.to_datetime(f["observation_date"], errors="raise").dt.normalize()
    if f["observation_date"].dt.tz is not None:
        raise ValueError("observation_date must be a timezone-free calendar date")
    for c in ("available_at", "vintage_at"):
        f[c] = utc(f[c])
    f["value"] = pd.to_numeric(f["value"], errors="raise")
    if f.isna().any().any() or not np.isfinite(f["value"]).all():
        raise ValueError("Ledger contains null or non-finite values")
    if f.duplicated(["series", "observation_date"]).any():
        raise ValueError("Duplicate series/date: resolve revisions or overlapping imports explicitly")
    for name, unit in EXPECTED_UNITS.items():
        rows = f.loc[f.series.eq(name)]
        if not rows.empty and not rows.unit.eq(unit).all():
            raise ValueError(f"{name} requires unit {unit}; convert units explicitly before import")
    positive = f.series.isin(["gold_price", "gld_price", "gld_holdings", "open_interest", "vix", "stocks", "dollar"])
    if (f.loc[positive, "value"] <= 0).any():
        raise ValueError("Price, holdings, OI and index levels must be positive")
    earliest = pd.to_datetime(f["observation_date"], utc=True)
    if (f["available_at"] < earliest).any():
        raise ValueError("Publication predates observation date")
    return f.sort_values(["series", "observation_date"]).reset_index(drop=True)


def read_ledger(path):
    return validate_ledger(pd.read_csv(path))


def quality_report(ledger):
    out = {}
    for name, f in ledger.groupby("series"):
        out[name] = {"n": len(f), "first": str(f.observation_date.min().date()),
                     "last": str(f.observation_date.max().date()), "unit": sorted(f.unit.unique()),
                     "source": sorted(f.source.unique()),
                     "availability_basis": sorted(f.availability_basis.unique()),
                     "negative_values": int((f.value < 0).sum()),
                     "max_gap_days": float(f.observation_date.diff().dt.days.max()) if len(f) > 1 else None}
    return out
