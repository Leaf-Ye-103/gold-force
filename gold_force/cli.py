import argparse
from pathlib import Path
import json
from .config import load_config
from .data.collect import collect
from .data.ledger import read_ledger
from .data.http import HttpClient
from .data import wgc
from .demo import build_demo
from .pipeline import analyse


def main():
    parser = argparse.ArgumentParser(description="Gold driver attribution and release-aware research forecasts")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "run", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/weekly.yaml")
        p.add_argument("--refresh", action="store_true")
        p.add_argument("--mode", choices=("both", "attribution", "forecast"), default="both")
        p.add_argument("--strict-vintages", action="store_true")
        if name == "analyse":
            p.add_argument("--ledger", required=True)
    p = sub.add_parser("demo")
    p.add_argument("--root", default="runs/demo_weekly")
    p.add_argument("--frequency", choices=("weekly", "quarterly"), default="weekly")
    p.add_argument("--mode", choices=("both", "attribution", "forecast"), default="both")
    p.add_argument("--strict-vintages", action="store_true")
    p = sub.add_parser("discover-wgc")
    p.add_argument("--kind", choices=("etf", "demand"), required=True)
    p.add_argument("--root", default="runs/discovery")
    p = sub.add_parser("download")
    p.add_argument("--url", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--refresh", action="store_true")
    p = sub.add_parser("inspect-xlsx")
    p.add_argument("file")
    args = parser.parse_args()
    try:
        if args.command == "discover-wgc":
            client = HttpClient(Path(args.root) / "raw")
            print(json.dumps(wgc.discover(client, args.kind), indent=2))
            client.save_manifest(Path(args.root) / "manifest.json")
            return
        if args.command == "download":
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            client = HttpClient(out.parent / "raw")
            body, _ = client.get(args.url, refresh=args.refresh)
            out.write_bytes(body)
            print(out.resolve())
            return
        if args.command == "inspect-xlsx":
            for sheet, contents in wgc.inspect_workbook(args.file).items():
                print(f"SHEET: {sheet}\n{contents}\n")
            return
        if args.command == "demo":
            cfg, ledger = build_demo(args.root, args.frequency)
        else:
            cfg = load_config(args.config)
            ledger = read_ledger(args.ledger) if args.command == "analyse" else collect(cfg, args.refresh)
        if args.command == "fetch":
            print(f"Collected {len(ledger)} observations into {cfg['root']}")
            return
        modes = ("attribution", "forecast") if args.mode == "both" else (args.mode,)
        outputs = analyse(ledger, cfg, modes, args.strict_vintages)
        for mode, meta in outputs.items():
            print(json.dumps({"mode":mode,"metrics":meta["metrics"],"report":str(Path(cfg['root'])/mode/'report.html')},indent=2))
    except Exception as exc:
        parser.exit(1, f"ERROR: {exc}\n")
