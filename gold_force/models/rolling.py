import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def fit_model(train, features, alphas, splits=3):
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    if len(alphas) == 1:
        pipe.set_params(ridge__alpha=float(alphas[0]))
        pipe.fit(train[features], train.actual_return)
        return pipe, float(alphas[0])
    cv = TimeSeriesSplit(n_splits=splits, gap=1)
    search = GridSearchCV(pipe, {"ridge__alpha": list(map(float, alphas))},
                          scoring="neg_mean_squared_error", cv=cv, n_jobs=1, error_score="raise")
    # StandardScaler is fitted within each chronological CV fold.
    search.fit(train[features], train.actual_return)
    return search.best_estimator_, search.best_params_["ridge__alpha"]


def contribution_row(pipe, current, features, groups):
    scaler, model = pipe.named_steps["scale"], pipe.named_steps["ridge"]
    z = scaler.transform(current[features])[0]
    contrib = model.coef_ * z
    predicted = float(model.intercept_ + contrib.sum())
    row = {"predicted_return": predicted, "intercept": float(model.intercept_)}
    for j, name in enumerate(features):
        row[name + "__contrib"] = float(contrib[j])
        row[name + "__beta_z"] = float(model.coef_[j])
        row[name + "__beta_raw"] = float(model.coef_[j] / scaler.scale_[j])
        row[name + "__z"] = float(z[j])
        row[name + "__mean"] = float(scaler.mean_[j])
        row[name + "__scale"] = float(scaler.scale_[j])
    group_values = {g: sum(contrib[features.index(n)] for n in names) for g, names in groups.items()}
    denom = sum(abs(float(x)) for x in group_values.values())
    for group, value in group_values.items():
        row[group + "__group_contrib"] = float(value)
        row[group + "__force"] = 100 * float(value) / denom if denom > 1e-12 else 0.0
    return row


def rolling_fit(design, cfg):
    d, features = design.frame, design.features
    settings = cfg["model"]
    window, minimum = settings["window"], settings["min_train"]
    rows, skips = [], []
    for i in range(1, len(d)):
        current = d.iloc[[i]]
        if not np.isfinite(current[features].to_numpy(dtype=float)).all():
            skips.append({"date": str(d.date.iloc[i]), "reason": "missing_or_stale_current_feature"})
            continue
        train = d.iloc[max(0, i-window):i].dropna(subset=features + ["actual_return"])
        if design.mode == "forecast":
            train = train.loc[train.target_available_at.le(current.decision_at.iloc[0])]
        if len(train) < minimum:
            skips.append({"date": str(d.date.iloc[i]), "reason": "insufficient_eligible_training", "n_train": len(train)})
            continue
        pipe, alpha = fit_model(train, features, settings["alphas"], settings.get("cv_splits", 3))
        row = contribution_row(pipe, current, features, design.groups)
        actual = float(current.actual_return.iloc[0])
        row.update({"date": current.date.iloc[0], "decision_at": current.decision_at.iloc[0],
                    "actual_return": actual, "residual": actual-row["predicted_return"],
                    "baseline_mean": float(train.actual_return.mean()), "alpha": alpha,
                    "n_train": len(train), "train_start": train.date.min(), "train_end": train.date.max(),
                    "mode": design.mode})
        rows.append(row)
    if not rows:
        raise ValueError("No model estimates. Check source coverage, min_train, feature availability and strict vintage policy")
    return pd.DataFrame(rows), pd.DataFrame(skips)


