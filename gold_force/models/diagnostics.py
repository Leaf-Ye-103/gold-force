import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(results):
    f = results.dropna(subset=["actual_return", "predicted_return"])
    if len(f) < 2:
        return {"n": len(f), "warning": "Too few evaluated observations"}
    y, p = f.actual_return.to_numpy(), f.predicted_return.to_numpy()
    mse = mean_squared_error(y, p)
    mean_error = mean_squared_error(y, f.baseline_mean)
    zero_error = float(np.mean(y*y))
    nonzero = y != 0
    return {"n": len(f), "mae_log_return_pp": float(mean_absolute_error(y, p)),
            "rmse_log_return_pp": float(np.sqrt(mse)), "r2": float(r2_score(y, p)),
            "skill_vs_rolling_mean": float(1-mse/mean_error) if mean_error > 0 else None,
            "skill_vs_zero": float(1-mse/zero_error) if zero_error > 0 else None,
            "direction_accuracy": float(np.mean(np.sign(y[nonzero]) == np.sign(p[nonzero]))) if nonzero.any() else None,
            "mean_residual_pp": float((y-p).mean()),
            "label": "retrospective_out_of_window_explanation" if f['mode'].iloc[0] == "attribution" else "release_aligned_research_forecast"}


def infer(design, maxlags=4):
    """Separate full-sample OLS-HAC analysis. NOT Ridge p-values or causal inference."""
    data = design.frame.dropna(subset=design.features + ["actual_return"])
    features = design.features
    if len(data) <= len(features)+8:
        return pd.DataFrame(), pd.DataFrame(), {"warning": "Insufficient observations for OLS diagnostics"}
    raw = data[features]
    constant = raw.std(ddof=0).le(1e-12)
    use = list(constant.index[~constant])
    if not use:
        return pd.DataFrame(), pd.DataFrame(), {"warning": "All predictors constant"}
    x = (raw[use] - raw[use].mean()) / raw[use].std(ddof=0)
    X = sm.add_constant(x, has_constant="add")
    rank = np.linalg.matrix_rank(X.to_numpy())
    if rank < X.shape[1]:
        return pd.DataFrame(), pd.DataFrame(), {"warning": "Rank-deficient design: no interpretable unique OLS coefficients",
                                               "rank": int(rank), "columns": int(X.shape[1])}
    fit = sm.OLS(data.actual_return, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags}, use_t=False)
    ci = fit.conf_int()
    table = pd.DataFrame({"coef_z": fit.params, "hac_se": fit.bse, "z_stat": fit.tvalues,
                          "p_value": fit.pvalues, "ci_low": ci[0], "ci_high": ci[1]})
    table["q_value_bh"] = np.nan
    table.loc[use, "q_value_bh"] = multipletests(table.loc[use, "p_value"], method="fdr_bh")[1]
    vif = pd.DataFrame({"feature": use, "vif": [variance_inflation_factor(X.to_numpy(), X.columns.get_loc(c)) for c in use]})
    bp = het_breuschpagan(fit.resid, X)
    lb = acorr_ljungbox(fit.resid, lags=[min(maxlags, max(1, len(data)//5))], return_df=True)
    diag = {"n": len(data), "condition_number": float(np.linalg.cond(X)),
            "breusch_pagan_p": float(bp[1]), "ljung_box_p": float(lb.lb_pvalue.iloc[0]),
            "omitted_constant_features": list(constant.index[constant]),
            "inference": "Descriptive full-sample OLS with HAC; does not validate causality or ridge attributions"}
    return table, vif, diag
