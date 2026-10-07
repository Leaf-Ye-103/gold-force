import pandas as pd
import numpy as np
import pytest
from gold_force.features import make_design, events_for, rule
from gold_force.models.rolling import rolling_fit
from gold_force.demo import build_demo
from gold_force.data.ledger import end_of_day


def test_forecast_cannot_read_unreleased_features(weekly):
    cfg, ledger = weekly
    d = make_design(ledger, cfg, "forecast", strict=True)
    assert (d.audit.effective_at <= d.audit.decision_at).all()
    assert (d.audit.observation_date < d.audit.date).all()
    assert (d.audit.age_days >= 0).all()


def test_current_vintage_is_blocked_in_strict_mode(weekly):
    cfg, ledger = weekly
    ledger["vintage_at"] = pd.Timestamp("2026-10-07", tz="UTC")
    strict = make_design(ledger, cfg, "forecast", strict=True)
    assert strict.frame[strict.features].isna().all().all()
    loose = make_design(ledger, cfg, "forecast", strict=False)
    assert loose.frame[loose.features].notna().any().any()


def test_stale_sources_do_not_forward_fill_forever(weekly):
    cfg, ledger = weekly
    ledger = ledger.loc[~(ledger.series.eq("gld_holdings") & ledger.observation_date.gt("2017-07-01"))]
    cfg["max_feature_age_days"] = 21
    d = make_design(ledger, cfg, "forecast")
    assert d.frame.loc[d.frame.date.gt("2017-09-01"), "gld_tonnes_change"].isna().all()


def test_missing_week_does_not_become_multiweek_return(weekly):
    cfg, ledger = weekly
    missing = pd.Timestamp("2017-06-02")
    ledger = ledger.loc[~(ledger.series.eq("gold_price") & ledger.observation_date.eq(missing))]
    d = make_design(ledger, cfg)
    assert d.frame.loc[d.frame.date.isin([missing, missing+pd.Timedelta(days=7)]), "actual_return"].isna().all()


def test_dst_is_explicit():
    stamps = end_of_day(["2024-01-05", "2024-07-05"])
    assert stamps[0].hour == 4 and stamps[1].hour == 3


def test_future_target_values_never_used_to_fit_forecast(weekly):
    cfg, ledger = weekly
    d = make_design(ledger, cfg, "forecast")
    r, _ = rolling_fit(d, cfg)
    eligible_date = r.date.iloc[-1]
    idx = d.frame.index[d.frame.date.eq(eligible_date)][0]
    d.frame.loc[idx, "actual_return"] = 999
    changed, _ = rolling_fit(d, cfg)
    assert changed.predicted_return.iloc[-1] == r.predicted_return.iloc[-1]


def test_quarterly_design_stays_quarterly(tmp_path):
    cfg, ledger = build_demo(tmp_path, "quarterly")
    d = make_design(ledger, cfg)
    assert len(d.frame) == 64
    assert d.frame.date.dt.is_quarter_end.all()
    assert d.frame.central_banks_yoy.iloc[:4].isna().all()
    assert np.isfinite(d.frame.central_banks_yoy.iloc[4:]).all()


def test_monthly_flow_cannot_be_presented_as_weekly(weekly):
    cfg, ledger = weekly
    cfg["features"] = ["global_etf_flow"]
    extra = ledger.loc[ledger.series.eq("gld_holdings")].iloc[:4].copy()
    extra["series"] = "global_etf_flow"
    extra["observation_date"] = pd.date_range("2017-01-31", periods=4, freq="ME")
    with pytest.raises(ValueError, match="genuine weekly"):
        make_design(pd.concat([ledger,extra]), cfg)
