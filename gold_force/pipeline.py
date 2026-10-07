from pathlib import Path
import json
from .features import make_design
from .models.rolling import rolling_fit
from .report import write_report
from .data.ledger import quality_report


def analyse(ledger, cfg, modes=("attribution", "forecast"), strict=False):
    root = Path(cfg["root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "data_quality.json").write_text(json.dumps(quality_report(ledger), indent=2), encoding="utf-8")
    outputs = {}
    errors = {}
    for mode in modes:
        try:
            design = make_design(ledger, cfg, mode, strict)
            results, skips = rolling_fit(design, cfg)
            outputs[mode] = write_report(design, results, skips, cfg, root / mode, strict)
        except Exception as exc:
            errors[mode] = f"{type(exc).__name__}: {exc}"
    status = {"outputs": outputs, "errors": errors}
    (root / "analysis_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    if errors:
        raise RuntimeError(f"Analysis incomplete: {errors}; inspect {root/'analysis_status.json'}")
    return outputs
