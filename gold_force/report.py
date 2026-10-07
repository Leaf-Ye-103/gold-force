from pathlib import Path
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .models.diagnostics import evaluate, infer
from .models.uncertainty import latest_bootstrap


def safe_json(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_report(design, results, skips, cfg, root, strict=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    results.to_csv(root / "force_history.csv", index=False)
    design.frame.to_csv(root / "design_matrix.csv", index=False)
    design.audit.to_csv(root / "feature_availability_audit.csv", index=False)
    skips.to_csv(root / "skipped_periods.csv", index=False)
    last = results.iloc[-1]
    latest = pd.DataFrame([{"group": g, "contribution_log_return_pp": last[g+"__group_contrib"],
                            "force_score": last[g+"__force"]} for g in design.groups])
    ci = latest_bootstrap(design, results, cfg)
    if not ci.empty:
        latest = latest.merge(ci, on="group")
    latest = latest.sort_values("contribution_log_return_pp", key=abs, ascending=False)
    latest.to_csv(root / "latest_force.csv", index=False)
    table, vif, diagnostics = infer(design, cfg.get("hac_maxlags", 4))
    table.to_csv(root / "ols_hac_descriptive.csv", index_label="feature")
    vif.to_csv(root / "vif.csv", index=False)
    metrics = evaluate(results)
    metadata = {"frequency": cfg["frequency"], "mode": design.mode, "strict_vintages": strict,
                "target_series": cfg.get("target_series", "gold_price"),
                "synthetic_demo": bool(cfg.get("synthetic_demo", False)),
                "metrics": metrics, "diagnostics": diagnostics,
                "latest": {k: last[k] for k in ("date", "actual_return", "predicted_return", "residual", "intercept", "n_train", "alpha")},
                "meaning": "Statistical contributions relative to historical feature means; not causal effects.",
                "units": "100*log(price_t/price_t-1); additive log-return percentage points",
                "forecast_limit": "Current-vintage reconstructed releases are research-only unless strict vintages is enabled.",
                "bootstrap_limit": "Conditional moving-block coefficient allocation intervals, not causal or forecast intervals."}
    (root / "metrics.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=safe_json), encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(results.date, results.actual_return, label="Actual", linewidth=1)
    axes[0].plot(results.date, results.predicted_return, label="Model", linewidth=1)
    axes[0].axhline(0, color="#888", linewidth=.5)
    axes[0].set_ylabel("Log return (%)")
    axes[0].legend()
    axes[0].set_title(("SYNTHETIC DEMO — " if cfg.get("synthetic_demo") else "") + design.mode)
    for g in design.groups:
        axes[1].plot(results.date, results[g+"__group_contrib"], label=g, linewidth=.8)
    axes[1].set_ylabel("Contribution (log-return pp)")
    axes[1].legend(ncol=3, fontsize=8)
    fig.savefig(root / "history.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    vals = latest.contribution_log_return_pp
    ax.barh(latest.group, vals, color=["#248670" if v >= 0 else "#cf604a" for v in vals])
    ax.axvline(0, color="#666", linewidth=.7)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution (log-return percentage points)")
    ax.set_title(f"{last.date.date()} — {design.mode}" + (" — SYNTHETIC" if cfg.get("synthetic_demo") else ""))
    fig.savefig(root / "latest.png", dpi=150)
    plt.close(fig)
    title = "Synthetic demonstration" if cfg.get("synthetic_demo") else "Gold driver model"
    reconciliation = pd.DataFrame({"component": ["intercept", "sum of driver contributions", "predicted", "unexplained residual", "actual"],
                                    "log_return_pp": [last.intercept, latest.contribution_log_return_pp.sum(),
                                                      last.predicted_return, last.residual, last.actual_return]})
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;background:#f5f7fa;color:#213044}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{width:100%;background:white;border-radius:12px;margin:20px 0}}.note{{padding:18px;background:#fff2d8;border-radius:10px;line-height:1.6}}code{{word-break:break-word}}</style>
<h1>{html.escape(title)} · {design.mode} · {cfg['frequency']}</h1>
<p>Target: <strong>{html.escape(cfg.get('target_series','gold_price'))}</strong>. {'GLD share-price return is a gold-market proxy, not quoted bullion USD/oz return; fees and premiums/discounts can differ.' if cfg.get('target_series') == 'gld_price' else 'Gold-price return in log percentage points.'}</p>
<p class="note">{'This report uses generated synthetic data. No current market or investor claim is implied. ' if cfg.get('synthetic_demo') else ''}
Contributions describe statistical associations relative to historical means, not independently identified buying pressure or causal effects.
Forecast inputs use publication cutoffs; {'strict vintage checks are enabled.' if strict else 'release dates may be reconstructed and historical values revised; research forecasts are not a verified trading backtest.'}
All additive values are log-return percentage points. GLD holdings are a single-fund demand proxy, not global ETF or retail demand.</p>
<h2>Latest contributions</h2>{latest.round(4).to_html(index=False, escape=True)}<img src="latest.png" alt="Latest model contributions">
<h2>Return reconciliation</h2>{reconciliation.round(4).to_html(index=False)}
<h2>Evaluation</h2><pre>{html.escape(json.dumps(metrics,indent=2))}</pre>
<img src="history.png" alt="Returns and driver contribution history">
<h2>Descriptive inference</h2><p>OLS-HAC is a separate full-sample diagnostic, not significance testing of rolling Ridge coefficients. Collinear positions can make allocations unstable.</p>
{table.round(5).to_html(escape=True)}<pre>{html.escape(json.dumps(diagnostics,indent=2))}</pre>
<p>CSV files include the full contributions, scaler parameters, raw-unit coefficients and availability audit.</p></html>'''
    (root / "report.html").write_text(text, encoding="utf-8")
    return metadata
