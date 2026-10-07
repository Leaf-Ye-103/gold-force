import io
import re
import pandas as pd
from .http import SourceError
from .ledger import records, numeric, end_of_day

PAGE = "https://www.spdrgoldshares.com/usa/gld/"
ARCHIVE = "https://api.spdrgoldshares.com/api/v1/historical-archive?exchange=NYSE&lang=en&product=gld"


def key(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def parse_archive(body, columns=None, price_mode="benchmark"):
    """Schema-based header discovery. Ambiguous price/holdings never guessed."""
    columns = columns or {}
    if price_mode not in ("benchmark", "gld_close", "holdings_only"):
        raise ValueError("SPDR price_mode must be benchmark, gld_close or holdings_only")
    tables = pd.read_excel(io.BytesIO(body), sheet_name=None, header=None) if body.startswith(b"PK") else {
        "csv": pd.read_csv(io.BytesIO(body), header=None, on_bad_lines="error")}
    for sheet, raw in tables.items():
        for n in range(min(30, len(raw))):
            headers = [str(v).strip() for v in raw.iloc[n].values]
            canon = [key(h) for h in headers]
            if "date" not in canon:
                continue
            def select(name, predicate):
                if name in columns:
                    candidates = [j for j, h in enumerate(headers) if h == columns[name]]
                else:
                    candidates = [j for j, h in enumerate(canon) if predicate(h)]
                if len(candidates) != 1:
                    raise ValueError(f"SPDR ambiguous {name} columns {headers}; specify sources.spdr.columns")
                return candidates[0]
            idx_date = select("date", lambda h: h == "date")
            idx_tonnes = select("gld_holdings", lambda h: h == "tonnes of gold" or ("tonnes" in h and ("trust" in h or "total" in h)))
            f = raw.iloc[n+1:].copy()
            dates = pd.to_datetime(f.iloc[:, idx_date], errors="coerce")
            tonnes = numeric(f.iloc[:, idx_tonnes])
            valid = dates.notna() & tonnes.notna()
            values = {"date": dates, "gld_holdings": tonnes}
            if price_mode != "holdings_only":
                name = "gold_price" if price_mode == "benchmark" else "gld_price"
                predicate = (lambda h: ("lbma" in h and "price" in h and "am" not in h)
                             or h in ("gold price", "gold price us", "london pm fix us oz", "london pm fix")) if price_mode == "benchmark" else (
                             lambda h: h in ("closing price", "gld close", "gld closing price"))
                idx_price = select(name, predicate)
                price = numeric(f.iloc[:, idx_price])
                values[name] = price
                valid &= price.notna()
            result = pd.DataFrame(values).loc[valid]
            if result.empty or (result.drop(columns="date") <= 0).any().any():
                raise ValueError("Invalid SPDR price/holdings data")
            if result.date.duplicated().any():
                raise ValueError("Duplicate SPDR date")
            return result.sort_values("date").reset_index(drop=True)
    raise ValueError("No SPDR Date header in first 30 rows; inspect official workbook")


def load(client, cfg, start, end, refresh=False):
    if cfg.get("file"):
        from pathlib import Path
        path = Path(cfg["file"])
        body = path.read_bytes()
        vintage = cfg.get("vintage_at", pd.Timestamp.now(tz="UTC").isoformat())
    else:
        if cfg.get("url"):
            url = cfg["url"]
        else:
            try:
                links = client.discover(PAGE, r"historical-archive", refresh)
                if len(links) != 1:
                    raise SourceError("Multiple SPDR archives; set explicit URL")
                url = links[0]
            except SourceError:
                # Current official endpoint observed on the GLD product page.
                url = ARCHIVE
        body, meta = client.get(url, refresh=refresh)
        vintage = meta["retrieved_at"]
    f = parse_archive(body, cfg.get("columns"), cfg.get("price_mode", "benchmark"))
    f = f.loc[f.date.between(start, end)]
    if f.empty:
        raise ValueError("No SPDR data for dates requested")
    available = end_of_day(f.date, cfg.get("assumed_delay_days", 2))
    units = {"gold_price": "usd_per_troy_oz", "gld_price": "usd_per_share", "gld_holdings": "tonnes"}
    return pd.concat([records(name, f.date, f[name], available, units[name],
                              "SPDR GLD official archive", vintage, "assumed_spdr_delay")
                      for name in f if name != "date"], ignore_index=True)
