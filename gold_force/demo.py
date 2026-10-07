"""Deterministic, explicitly synthetic examples. Never a downloader fallback."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .data.ledger import records, end_of_day, validate_ledger


def build_demo(root, frequency="weekly", seed=20261007):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-06", periods=420, freq="W-FRI") if frequency == "weekly" else pd.date_range("2010-03-31", periods=64, freq="QE-DEC")
    n = len(dates)
    frames = []
    avail = end_of_day(dates, 2)
    # Synthetic fixture vintages equal the simulated first-release time.
    def add(name, values, unit, delay=2):
        release = end_of_day(dates, delay)
        frames.append(records(name, dates, values, release, unit, "SYNTHETIC DEMO", release, "synthetic_first_release"))
    dollar_r = rng.normal(0, .7, n)
    stocks_r = rng.normal(.12, 1.5, n)
    yield_d = rng.normal(0, .06, n)
    vix_r = rng.normal(0, 5, n)
    add("dollar", 100*np.exp(np.cumsum(dollar_r)/100), "broad_dollar_index", 10)
    add("stocks", 3000*np.exp(np.cumsum(stocks_r)/100), "index")
    add("real_yield", 1+np.cumsum(yield_d), "percent")
    add("vix", 20*np.exp(np.cumsum(vix_r)/100), "index")
    if frequency == "weekly":
        flow = rng.normal(.1, 5, n)
        mm = 20+np.cumsum(rng.normal(0, .7, n))
        swap = -15+np.cumsum(rng.normal(0, .5, n))
        producer = -10+np.cumsum(rng.normal(0, .3, n))
        nonrep = 5+np.cumsum(rng.normal(0, .2, n))
        add("gld_holdings", 900+np.cumsum(flow), "tonnes")
        for group, values in (("mm", mm), ("swap", swap), ("producer", producer), ("nonreportable", nonrep),
                              ("other", -mm-swap-producer-nonrep)):
            add(group+"_net_pct_oi", values, "percent_oi", 4)
        add("open_interest", 500000*np.exp(np.cumsum(rng.normal(0, .01, n))), "contracts", 4)
        ret = .10+.05*flow+.2*np.r_[0, np.diff(mm)]-.6*dollar_r-2*yield_d+.03*vix_r+.1*stocks_r+rng.normal(0,.65,n)
        features = ["gld_tonnes_change", "mm_change", "swap_change", "producer_change", "nonreportable_change",
                    "oi_return", "dollar_return", "real_yield_change", "vix_return", "stocks_return"]
        window, minimum = 104, 60
    else:
        season = np.sin(np.arange(n)*np.pi/2)
        cb = 100+np.arange(n)*2+rng.normal(0,20,n)
        retail = 250+25*season+rng.normal(0,20,n)
        etf = rng.normal(0,60,n)
        mine = 800+30*season+rng.normal(0,15,n)
        recycle = 300+rng.normal(0,30,n)
        for name, values in (("cb_purchase",cb),("bar_coin",retail),("etf_demand",etf),
                              ("mine_supply",mine),("recycling",recycle)):
            add(name,values,"tonnes",35)
        ret = .5+.012*cb+.01*retail+.02*etf-.008*mine-.005*recycle-.8*dollar_r-2*yield_d+rng.normal(0,1.5,n)
        features = ["central_banks_yoy", "retail_yoy", "etf_demand", "mine_yoy", "recycling_yoy",
                    "dollar_return", "real_yield_change", "stocks_return"]
        window, minimum = 32, 24
    add("gold_price", 1300*np.exp(np.cumsum(ret)/100), "usd_per_troy_oz", 0)
    ledger = validate_ledger(pd.concat(frames, ignore_index=True))
    path = root / "synthetic_observations.csv"
    ledger.to_csv(path, index=False)
    cfg = {"root": str(root), "start": str(dates[0].date()), "end": str(dates[-1].date()),
           "frequency": frequency, "synthetic_demo": True, "features": features,
           "max_feature_age_days": 28 if frequency == "weekly" else 220,
           "model": {"window": window,"min_train":minimum,"alphas":[.1,1,10,100],"cv_splits":3},
           "bootstrap":{"iterations":100,"block_length":4,"seed":seed},"hac_maxlags":4,
           "sources":{}, "imports":[{"file":str(path)}]}
    config_path = root / "demo_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    return cfg, ledger
