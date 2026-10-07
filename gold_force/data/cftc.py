import io
import json
import zipfile
import re
import pandas as pd
from .ledger import numeric, records, end_of_day
from .http import SourceError

API = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
HISTORY = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm"
PAIRS = {
    "mm": ("m_money_positions_long_all", "m_money_positions_short_all"),
    "swap": ("swap_positions_long_all", "swap_positions_short_all"),
    "producer": ("prod_merc_positions_long", "prod_merc_positions_short"),
    "other": ("other_rept_positions_long", "other_rept_positions_short"),
    "nonreportable": ("nonrept_positions_long_all", "nonrept_positions_short_all"),
}


def normalise(raw):
    f = raw.copy()
    f.columns = [re.sub(r"_+", "_", str(c).strip().lower()) for c in f.columns]
    need = {"report_date_as_yyyy_mm_dd", "cftc_contract_market_code", "open_interest_all"}
    need |= {c for pair in PAIRS.values() for c in pair}
    if need - set(f):
        raise ValueError(f"CFTC schema missing: {sorted(need-set(f))}")
    code = f.cftc_contract_market_code.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    f = f.loc[code == "088691"].copy()
    if f.empty:
        raise ValueError("No COMEX Gold contract 088691")
    if "futonly_or_combined" in f and not f.futonly_or_combined.str.lower().eq("futonly").all():
        raise ValueError("Combined futures/options rows supplied to futures-only loader")
    dates = pd.to_datetime(f.report_date_as_yyyy_mm_dd).dt.normalize()
    oi = numeric(f.open_interest_all)
    if oi.isna().any() or (oi <= 0).any():
        raise ValueError("CFTC OI must be positive")
    result = pd.DataFrame({"date": dates, "open_interest": oi})
    net_sum = pd.Series(0.0, index=f.index)
    for group, (long, short) in PAIRS.items():
        a, b = numeric(f[long]), numeric(f[short])
        if a.isna().any() or b.isna().any() or (a < 0).any() or (b < 0).any():
            raise ValueError(f"Invalid CFTC positions: {group}")
        result[group + "_net_pct_oi"] = 100 * (a-b) / oi
        net_sum += a-b
    if not (net_sum.abs() <= 1).all():
        raise ValueError("CFTC net positions fail long/short accounting identity")
    if result.date.duplicated().any():
        raise ValueError("Duplicate CFTC contract/report date")
    return result.sort_values("date").reset_index(drop=True)


def fetch_api(client, start, end, refresh=False):
    where = f"cftc_contract_market_code='088691' AND report_date_as_yyyy_mm_dd >= '{start}T00:00:00' AND report_date_as_yyyy_mm_dd <= '{end}T23:59:59'"
    rows, vintages = [], []
    for offset in range(0, 100000, 5000):
        body, meta = client.get(API, params={"$where": where, "$order": "report_date_as_yyyy_mm_dd ASC,id ASC",
                                            "$limit": 5000, "$offset": offset}, refresh=refresh)
        page = json.loads(body)
        if not isinstance(page, list):
            raise SourceError("CFTC returned a non-list response")
        rows.extend(page)
        vintages.append(meta["retrieved_at"])
        if len(page) < 5000:
            break
    else:
        raise SourceError("CFTC pagination cap exceeded")
    if not rows:
        raise SourceError("CFTC returned no Gold rows for requested dates")
    return normalise(pd.DataFrame(rows)), max(vintages)


def fetch_zip(client, start, end, refresh=False):
    frames, vintages = [], []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        # This path is the CFTC Disaggregated Futures Only text ZIP pattern.
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
        body, meta = client.get(url, refresh=refresh)
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            candidates = [n for n in z.namelist() if n.lower().endswith((".txt", ".csv"))]
            if len(candidates) != 1:
                raise SourceError("Ambiguous CFTC ZIP members")
            frames.append(pd.read_csv(z.open(candidates[0]), dtype=str))
        vintages.append(meta["retrieved_at"])
    f = normalise(pd.concat(frames, ignore_index=True))
    return f.loc[f.date.between(start, end)], max(vintages)


def load(client, cfg, start, end, refresh=False):
    method = fetch_api if cfg.get("method", "api") == "api" else fetch_zip
    f, vintage = method(client, start, end, refresh)
    # A reconstruction assumption, never promoted to an actual historical release.
    avail = end_of_day(f.date, days=cfg.get("assumed_delay_days", 10))
    basis = pd.Series("assumed_cftc_delay", index=f.index)
    if cfg.get("release_calendar"):
        calendar = pd.read_csv(cfg["release_calendar"])
        calendar["observation_date"] = pd.to_datetime(calendar["observation_date"])
        if calendar.observation_date.duplicated().any():
            raise ValueError("Duplicate CFTC release calendar dates")
        mapping = calendar.set_index("observation_date")["available_at"]
        if not f.date.isin(mapping.index).all():
            raise ValueError("Release calendar must cover every downloaded CFTC report")
        avail = pd.to_datetime(f.date.map(mapping), utc=True)
        basis[:] = "user_verified_release_calendar"
    return pd.concat([records(c, f.date, f[c], avail,
                              "contracts" if c == "open_interest" else "percent_oi",
                              "CFTC COMEX Gold futures-only", vintage, basis)
                      for c in f if c != "date"], ignore_index=True)
