import numpy as np
import pandas as pd
import pytest
from gold_force.features import make_design
from gold_force.models.rolling import rolling_fit, fit_model
from gold_force.models.diagnostics import infer, evaluate
from gold_force.models.uncertainty import latest_bootstrap


def test_contribution_reconciles(weekly):
    cfg, ledger = weekly
    design = make_design(ledger, cfg)
    result, _ = rolling_fit(design, cfg)
    factors = [f+"__contrib" for f in design.features]
    groups = [g+"__group_contrib" for g in design.groups]
    assert np.allclose(result[factors].sum(axis=1)+result.intercept, result.predicted_return)
    assert np.allclose(result[groups].sum(axis=1), result[factors].sum(axis=1))
    assert np.allclose(result.predicted_return+result.residual, result.actual_return)
    assert np.allclose(result[[g+"__force" for g in design.groups]].abs().sum(axis=1), 100)
    assert (result.train_end < result.date).all()


def test_future_outlier_does_not_change_history_or_scaler(weekly):
    cfg, ledger = weekly
    d1 = make_design(ledger, cfg)
    r1, _ = rolling_fit(d1, cfg)
    date = r1.date.iloc[-1]
    revised = ledger.copy()
    revised.loc[(revised.series == "gold_price") & revised.observation_date.eq(date), "value"] *= 8
    revised.loc[(revised.series == "gld_holdings") & revised.observation_date.eq(date), "value"] *= 2
    r2, _ = rolling_fit(make_design(revised, cfg), cfg)
    pd.testing.assert_frame_equal(r1.iloc[:-1], r2.iloc[:-1])
    assert r1["gld_tonnes_change__mean"].iloc[-1] == r2["gld_tonnes_change__mean"].iloc[-1]


def test_chronological_cv_selects_only_allowed_alpha(weekly):
    cfg, ledger = weekly
    d = make_design(ledger, cfg)
    train = d.frame.dropna().iloc[:30]
    pipe, alpha = fit_model(train, d.features, [.1, 10], 3)
    assert alpha in [.1, 10]
    assert np.allclose(pipe.named_steps["scale"].mean_, train[d.features].mean())


def test_zero_force_is_defined(weekly):
    cfg, ledger = weekly
    d = make_design(ledger, cfg)
    d.frame[d.features] = 1.0
    r, _ = rolling_fit(d, cfg)
    assert (r[[g+"__force" for g in d.groups]] == 0).all().all()


def test_rank_deficient_ols_rejected(weekly):
    cfg, ledger = weekly
    d = make_design(ledger, cfg)
    d.frame["swap_change"] = d.frame["mm_change"]
    table, _, diag = infer(d)
    assert table.empty and "Rank-deficient" in diag["warning"]


def test_bootstrap_and_metrics(weekly):
    cfg, ledger = weekly
    d = make_design(ledger, cfg)
    r, _ = rolling_fit(d, cfg)
    cfg["bootstrap"]["iterations"] = 8
    ci = latest_bootstrap(d, r, cfg)
    assert len(ci) == len(d.groups)
    assert (ci.contribution_ci_low_pp <= ci.contribution_ci_high_pp).all()
    assert evaluate(r)["n"] == len(r)
