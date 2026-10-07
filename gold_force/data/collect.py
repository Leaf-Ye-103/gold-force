from pathlib import Path
import json
import pandas as pd
from .http import HttpClient
from .ledger import validate_ledger, quality_report, COLS
from . import cftc, spdr, fred, wgc, imports


def collect(cfg, refresh=False):
    root = Path(cfg["root"])
    root.mkdir(parents=True, exist_ok=True)
    client = HttpClient(root / "data/raw", **cfg.get("http", {}))
    frames, status = [], []
    loaders = {"cftc": cftc.load, "spdr": spdr.load, "fred": fred.load, "wgc": wgc.load}
    for name, loader in loaders.items():
        options = cfg.get("sources", {}).get(name, {})
        if not options.get("enabled", False):
            continue
        try:
            f = loader(client, options, cfg["start"], cfg["end"], refresh)
            frames.append(f)
            status.append({"source": name, "ok": True, "rows": len(f)})
        except Exception as exc:
            status.append({"source": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    for item in cfg.get("imports", []):
        try:
            f = imports.load(item)
            frames.append(f.loc[f.observation_date.between(cfg["start"], cfg["end"])])
            status.append({"source": item["file"], "ok": True, "rows": len(f)})
        except Exception as exc:
            status.append({"source": item["file"], "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    client.save_manifest(root / "download_manifest.json")
    (root / "source_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    if frames:
        ledger = validate_ledger(pd.concat(frames, ignore_index=True))
        dest = root / "data/processed"
        dest.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(dest / "observations.csv", index=False)
        (root / "data_quality.json").write_text(json.dumps(quality_report(ledger), indent=2), encoding="utf-8")
    else:
        ledger = pd.DataFrame(columns=COLS)
    failed = [s["source"] for s in status if not s["ok"]]
    if failed:
        raise RuntimeError(f"Sources failed: {failed}. Successful data retained; inspect {root/'source_status.json'}. No synthetic substitution.")
    if ledger.empty:
        raise ValueError("No sources or imports produced observations")
    return ledger
