from pathlib import Path
from urllib.parse import urlsplit, urljoin
from hashlib import sha256
from datetime import datetime, timezone
import json
import time
import requests
from bs4 import BeautifulSoup


class SourceError(RuntimeError):
    pass


class HttpClient:
    """Bounded retries, immutable raw snapshots, SHA256 and retrieval manifest.

    Does not retry access-denied pages or attempt to bypass authentication.
    Cached data never silently substitutes for a failed refresh.
    """
    def __init__(self, root, timeout=30, retries=2, cache_hours=24):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout, self.retries, self.cache_hours = timeout, retries, cache_hours
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "GoldForceResearch/1.0", "Accept": "*/*"})
        self.manifest = []

    def get(self, url, params=None, refresh=False):
        if urlsplit(url).scheme != "https":
            raise SourceError("Only HTTPS download URLs are accepted")
        req = requests.Request("GET", url, params=params).prepare()
        key = sha256(req.url.encode()).hexdigest()
        meta_path = self.root / (key + ".json")
        if not refresh and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if time.time() - meta["retrieved_epoch"] < self.cache_hours * 3600:
                blob = self.root / meta["blob"]
                if blob.exists() and sha256(blob.read_bytes()).hexdigest() == meta["sha256"]:
                    self.manifest.append({**meta, "cache_hit": True})
                    return blob.read_bytes(), meta
        for attempt in range(self.retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code in (401, 403, 404):
                    raise SourceError(f"HTTP {r.status_code}: {url}; use an authorized local import")
                if r.status_code == 429 or r.status_code >= 500:
                    r.raise_for_status()
                r.raise_for_status()
                if len(r.content) > 100 * 1024 * 1024:
                    raise SourceError("Download exceeds 100 MiB limit")
                digest = sha256(r.content).hexdigest()
                blob = key + "_" + digest[:16] + ".bin"
                (self.root / blob).write_bytes(r.content)
                meta = {"url": req.url, "final_url": r.url, "sha256": digest,
                        "content_type": r.headers.get("Content-Type", ""), "blob": blob,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "retrieved_epoch": time.time()}
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                self.manifest.append({**meta, "cache_hit": False})
                return r.content, meta
            except SourceError:
                raise
            except requests.RequestException as exc:
                if attempt == self.retries:
                    raise SourceError(f"Download failed: {url}: {type(exc).__name__}") from exc
                time.sleep(min(2 ** attempt, 4))

    def discover(self, page, pattern, refresh=False):
        import re
        body, meta = self.get(page, refresh=refresh)
        soup = BeautifulSoup(body, "html.parser")
        matches = []
        for a in soup.find_all("a", href=True):
            href = urljoin(meta["final_url"], a["href"])
            if re.search(pattern, href + " " + a.get_text(" ", strip=True), re.I):
                matches.append(href)
        matches = list(dict.fromkeys(matches))
        if not matches:
            raise SourceError(f"No download matched {pattern!r} on {page}; inspect current official page")
        return matches

    def save_manifest(self, path):
        Path(path).write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
