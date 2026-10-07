"""Regular period features and release-aware, bounded as-of alignment."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data.ledger import end_of_day

WEEKLY = {
    "gld_tonnes_change": ("gld_holdings", "diff", "ETF_GLD_proxy"),
    "global_etf_flow": ("global_etf_flow", "level", "ETF_Global"),
    "mm_change": ("mm_net_pct_oi", "diff", "Managed_Money"),
    "swap_change": ("swap_net_pct_oi", "diff", "Swap_Dealers"),
    "producer_change": ("producer_net_pct_oi", "diff", "Producers"),
    "nonreportable_change": ("nonreportable_net_pct_oi", "diff", "Nonreportable_proxy"),
    "oi_return": ("open_interest", "log", "Market_Activity"),
    "dollar_return": ("dollar", "log", "Dollar"),
    "real_yield_change": ("real_yield", "diff", "Real_Yield"),
    "vix_return": ("vix", "log", "Risk_Sentiment"),
    "stocks_return": ("stocks", "log", "Equities"),
}
QUARTERLY = {
    "central_banks_yoy": ("cb_purchase", "yoy", "Central_Banks"),
    "retail_yoy": ("bar_coin", "yoy", "Retail_Physical"),
    "etf_demand": ("etf_demand", "level", "ETF_Global"),
    "mine_yoy": ("mine_supply", "yoy", "Mine_Supply"),
    "recycling_yoy": ("recycling", "yoy", "Recycling"),
    "otc_demand": ("otc_demand", "level", "OTC_Estimate"),
    "dollar_return": ("dollar", "log", "Dollar"),
    "real_yield_change": ("real_yield", "diff", "Real_Yield"),
    "stocks_return": ("stocks", "log", "Equities"),
}


@dataclass
class Design:
    frame: pd.DataFrame
    features: list
    groups: dict
    audit: pd.DataFrame
    mode: str


def rule(frequency):
    return "W-FRI" if frequency == "weekly" else "QE-DEC"


def period_levels(ledger, series, index, frequency):
    f = ledger.loc[ledger.series.eq(series)].sort_values("observation_date")
    if f.empty:
        raise ValueError(f"Required series absent: {series}")
    if series in ("cb_purchase", "bar_coin", "etf_demand", "mine_supply", "recycling", "otc_demand"):
        if frequency != "quarterly" or not f.observation_date.dt.is_quarter_end.all():
            raise ValueError(f"{series} requires quarterly totals dated at quarter-end; never copy low-frequency totals into weekly rows")
    if series == "global_etf_flow" and (frequency != "weekly" or not f.observation_date.dt.dayofweek.eq(4).all()):
        raise ValueError("global_etf_flow requires genuine weekly totals dated Friday, not repeated monthly data")
    f = f.set_index("observation_date")
    # Last valid observation WITHIN each period. Empty periods remain missing.
    last = f.resample(rule(frequency)).last().reindex(index)
    return last[["value", "available_at", "vintage_at"]]


def events_for(ledger, series, transform, index, frequency):
    levels = period_levels(ledger, series, index, frequency)
    out = levels.copy()
    lag = 4 if transform == "yoy" else 1
    if transform in ("diff", "yoy", "log"):
        before = levels.shift(lag)
        if transform == "log":
            if (levels.value.dropna() <= 0).any():
                raise ValueError(f"Log transform requires positive {series}")
            out["value"] = 100 * np.log(levels.value / before.value)
        else:
            out["value"] = levels.value - before.value
        for c in ("available_at", "vintage_at"):
            out[c] = pd.concat([levels[c], before[c]], axis=1).max(axis=1)
    elif transform != "level":
        raise ValueError(f"Unknown transform {transform}")
    # A full period's level/change cannot be known before the period ends.
    completed = pd.Series(end_of_day(index), index=index)
    out["available_at"] = pd.concat([out.available_at, completed], axis=1).max(axis=1)
    return out


def make_design(ledger, cfg, mode="attribution", strict=False):
    if mode not in ("attribution", "forecast"):
        raise ValueError("mode must be attribution or forecast")
    frequency = cfg["frequency"]
    spec = WEEKLY if frequency == "weekly" else QUARTERLY
    names = cfg["features"]
    if len(names) != len(set(names)) or set(names) - set(spec):
        raise ValueError("Unknown or duplicated feature names")
    # Avoid double-counting overlapping GLD/global ETF measures by default.
    if {"gld_tonnes_change", "global_etf_flow"} <= set(names):
        raise ValueError("Choose GLD proxy OR global ETF flow, not both")
    # A forecast origin is the preceding completed period, not this period.
    index = pd.date_range(cfg["start"], cfg["end"], freq=rule(frequency))
    if len(index) < 2:
        raise ValueError("At least two complete periods required")
    ends = end_of_day(index)
    now = pd.Timestamp.now(tz="UTC")
    index, ends = index[ends <= now], ends[ends <= now]
    frame = pd.DataFrame({"date": index, "decision_at": ends})
    if mode == "forecast":
        frame["decision_at"] = pd.Series(ends).shift(1).values
        frame["decision_at"] = pd.to_datetime(frame.decision_at, utc=True)
    target = events_for(ledger, cfg.get("target_series", "gold_price"), "log", index, frequency)
    frame["actual_return"] = target.value.to_numpy()
    frame["target_available_at"] = target.available_at.to_numpy()
    if strict:
        frame["target_available_at"] = pd.concat([target.available_at, target.vintage_at], axis=1).max(axis=1).to_numpy()
    audits, groups = [], {}
    for name in names:
        source_series, transform, group = spec[name]
        groups.setdefault(group, []).append(name)
        event = events_for(ledger, source_series, transform, index, frequency)
        event["observation_date"] = index
        effective = pd.concat([event.available_at, event.vintage_at], axis=1).max(axis=1) if strict else event.available_at
        event["effective_at"] = effective
        if mode == "attribution":
            frame[name] = event.value.to_numpy()
            for date, row in event.iterrows():
                audits.append({"date": date, "feature": name, "observation_date": date,
                               "available_at": row.available_at, "vintage_at": row.vintage_at,
                               "effective_at": row.effective_at, "age_days": 0})
        else:
            values = []
            max_age = cfg.get("max_feature_age_days", 28 if frequency == "weekly" else 220)
            for date, cutoff in zip(index, frame.decision_at):
                usable = event.loc[event.effective_at.le(cutoff) & event.observation_date.lt(date)
                                   & event.value.notna()]
                # Never select an older observation merely because it was revised later.
                usable = usable.sort_values(["observation_date", "effective_at"])
                if usable.empty or pd.isna(cutoff):
                    values.append(np.nan)
                    continue
                chosen = usable.iloc[-1]
                age = (cutoff.tz_convert("America/New_York").tz_localize(None).normalize()
                       - chosen.observation_date).days
                values.append(chosen.value if age <= max_age else np.nan)
                audits.append({"date": date, "feature": name, "observation_date": chosen.observation_date,
                               "decision_at": cutoff, "available_at": chosen.available_at,
                               "vintage_at": chosen.vintage_at, "effective_at": chosen.effective_at,
                               "age_days": age, "stale": bool(age > max_age)})
            frame[name] = values
    frame["target_available_at"] = pd.to_datetime(frame.target_available_at, utc=True)
    return Design(frame, names, groups, pd.DataFrame(audits), mode)
