"""Download discovery plus explicit workbook mapping for WGC files.

Layouts change and download access can be restricted. A schema mismatch fails
instead of inferring the units from numbers or using a login bypass.
"""
from pathlib import Path
import io
import re
import pandas as pd
from .ledger import records, numeric
from .http import SourceError

PAGES = {
    "etf": "https://www.gold.org/goldhub/data/gold-etfs-holdings-and-flows",
    "demand": "https://www.gold.org/goldhub/data/gold-demand-by-country",
}


def discover(client, kind, refresh=False):
    if kind not in PAGES:
        raise ValueError("WGC kind must be etf or demand")
    return client.discover(PAGES[kind], r"\.xlsx(?:\?|$)", refresh)


def inspect_workbook(path):
    return {sheet: f.head(35).fillna("").to_string(index=True, header=True)
            for sheet, f in pd.read_excel(path, sheet_name=None, header=None).items()}


def period_end(value, frequency):
    if frequency == "quarterly":
        if isinstance(value, (pd.Timestamp,)):
            return value.to_period("Q").end_time.normalize()
        s = str(value).strip()
        match = re.fullmatch(r"(?:Q([1-4])\s*[- /]?\s*(\d{4})|(\d{4})\s*[- /]?\s*Q([1-4]))", s, re.I)
        if match:
            q, year = (match[1], match[2]) if match[1] else (match[4], match[3])
            return pd.Period(f"{year}Q{q}", freq="Q").end_time.normalize()
    return pd.Timestamp(value).normalize()


def parse_mapping(body, cfg, vintage):
    layout = cfg.get("layout", "long")
    frequency = cfg.get("frequency", "quarterly")
    if layout == "long":
        f = pd.read_excel(io.BytesIO(body), sheet_name=cfg.get("sheet", 0), header=cfg.get("header", 0))
        dates = f[cfg.get("date_column", "date")].map(lambda v: period_end(v, frequency))
        if not cfg.get("available_column"):
            raise ValueError("WGC long format requires available_column with actual publication timestamps")
        available = pd.to_datetime(f[cfg["available_column"]], utc=True)
        return pd.concat([records(series, dates, numeric(f[column]), available,
                                  cfg.get("units", {}).get(series, "tonnes"),
                                  cfg.get("source", "WGC official workbook"), vintage,
                                  "user_supplied_publication")
                          for series, column in cfg["series_columns"].items()], ignore_index=True)
    if layout != "rows":
        raise ValueError("WGC layout must be long or rows")
    raw = pd.read_excel(io.BytesIO(body), sheet_name=cfg["sheet"], header=None)
    # All row/column indices here are ZERO-based. No automatic sum of countries.
    cols = cfg["period_columns"]
    dates = [period_end(raw.iat[cfg["period_row"], c], frequency) for c in cols]
    releases = pd.read_csv(cfg["release_calendar"])
    releases["observation_date"] = pd.to_datetime(releases.observation_date)
    if releases.observation_date.duplicated().any():
        raise ValueError("Duplicate WGC release calendar observation_date")
    mapping = releases.set_index("observation_date")["available_at"]
    if not pd.Index(dates).isin(mapping.index).all():
        raise ValueError("WGC release calendar must cover every period column")
    available = [mapping[d] for d in dates]
    frames = []
    for series, spec in cfg["series_rows"].items():
        row = spec["row"]
        label = str(raw.iat[row, cfg.get("label_column", 0)]).strip()
        if label != spec["expected_label"]:
            raise ValueError(f"WGC row {row} label changed: {label!r}, expected {spec['expected_label']!r}")
        values = numeric(pd.Series([raw.iat[row, c] for c in cols]))
        if values.isna().any():
            raise ValueError(f"Missing numeric WGC value in {series}; check row/column mapping")
        frames.append(records(series, dates, values, available, spec.get("unit", "tonnes"),
                              "WGC official workbook", vintage, "user_supplied_publication"))
    return pd.concat(frames, ignore_index=True)


def load(client, cfg, start, end, refresh=False):
    if cfg.get("file"):
        body = Path(cfg["file"]).read_bytes()
        vintage = cfg.get("vintage_at", pd.Timestamp.now(tz="UTC").isoformat())
    else:
        urls = [cfg["url"]] if cfg.get("url") else discover(client, cfg.get("kind", "demand"), refresh)
        if len(urls) != 1:
            raise SourceError("Multiple WGC downloads; select URL explicitly using discover-wgc")
        body, meta = client.get(urls[0], refresh=refresh)
        vintage = meta["retrieved_at"]
    f = parse_mapping(body, cfg, vintage)
    return f.loc[f.observation_date.between(start, end)]
