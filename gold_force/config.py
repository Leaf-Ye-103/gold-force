from pathlib import Path
import copy
import yaml


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    for key in ("start", "end"):
        cfg[key] = str(cfg[key])
    cfg["root"] = str(Path(cfg.get("root", "runs/live")).resolve())
    if cfg["frequency"] not in ("weekly", "quarterly"):
        raise ValueError("frequency must be weekly or quarterly")
    if cfg["start"] > cfg["end"]:
        raise ValueError("start must precede end")
    model = cfg["model"]
    if model["window"] < model["min_train"] or model["min_train"] < 10:
        raise ValueError("Require window >= min_train >= 10")
    if any(float(a) <= 0 for a in model["alphas"]):
        raise ValueError("Ridge alphas must be positive")
    return cfg
