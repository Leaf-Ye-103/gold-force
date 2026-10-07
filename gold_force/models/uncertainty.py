import numpy as np
import pandas as pd
from .rolling import fit_model, contribution_row


def latest_bootstrap(design, results, cfg):
    """Moving-block pairs bootstrap of the LAST rolling regression.

    Measures coefficient/allocation instability conditional on selected alpha.
    Not causal confidence intervals, and not future-return prediction intervals.
    """
    opts = cfg.get("bootstrap", {})
    count = opts.get("iterations", 100)
    if count < 1:
        return pd.DataFrame()
    date = results.date.iloc[-1]
    idx = design.frame.index[design.frame.date.eq(date)][0]
    current = design.frame.iloc[[idx]]
    train = design.frame.iloc[max(0, idx-cfg["model"]["window"]):idx]
    train = train.dropna(subset=design.features+["actual_return"])
    if design.mode == "forecast":
        train = train.loc[train.target_available_at.le(current.decision_at.iloc[0])]
    n = len(train)
    block = min(opts.get("block_length", 4), n)
    if block < 1:
        raise ValueError("bootstrap block_length must be >=1")
    rng = np.random.default_rng(opts.get("seed", 1234))
    samples = []
    for _ in range(count):
        starts = rng.integers(0, n-block+1, size=int(np.ceil(n/block)))
        indices = np.concatenate([np.arange(s, s+block) for s in starts])[:n]
        resampled = train.iloc[indices]
        pipe, _ = fit_model(resampled, design.features, [results.alpha.iloc[-1]])
        row = contribution_row(pipe, current, design.features, design.groups)
        samples.append({g: row[g+"__group_contrib"] for g in design.groups})
    f = pd.DataFrame(samples)
    return pd.DataFrame({"group": f.columns, "contribution_ci_low_pp": f.quantile(.025).values,
                         "contribution_ci_high_pp": f.quantile(.975).values,
                         "bootstrap_probability_positive": (f > 0).mean().values,
                         "iterations": count})
