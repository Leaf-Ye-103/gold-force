from pathlib import Path
import sqlite3
import pandas as pd
from .ledger import validate_ledger


def load_table(path, sheet=0, query=None):
    """Read user-controlled files. SQL is read-only SQLite, never arbitrary SQL writes."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".csv", ".txt", ".tsv"):
        return pd.read_csv(p, sep="\t" if ext == ".tsv" else None, engine="python")
    if ext in (".xlsx", ".xlsm"):
        return pd.read_excel(p, sheet_name=sheet)
    if ext == ".json":
        return pd.read_json(p)
    if ext in (".sqlite", ".db", ".sqlite3"):
        if not query or not query.lstrip().lower().startswith("select"):
            raise ValueError("SQLite import requires a SELECT query")
        with sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            conn.execute("PRAGMA query_only=ON")
            return pd.read_sql_query(query, conn)
    raise ValueError(f"Unsupported file type {ext}; supported: CSV, TSV, TXT, XLSX, JSON, SQLite")


def load(cfg):
    f = load_table(cfg["file"], cfg.get("sheet", 0), cfg.get("query"))
    return validate_ledger(f.rename(columns=cfg.get("rename", {})))
