"""Ship the built data (search index + financials DB) to the deployed app.

The index is built locally (download, parse, embed takes about 20 minutes) and is too
large for git, so it is published as one tarball on a GitHub Release. On a fresh
deployment the app downloads it once, checks its SHA-256 (so a corrupted or tampered
download is never used), and extracts it with tarfile's "data" filter, which refuses
absolute paths, "..", links and device files.

    finrag-package-data   ->  dist/finrag-data.tar.gz  (+ prints its SHA-256)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import tarfile
import tempfile
from pathlib import Path

import requests

from finrag.config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

BUNDLE_NAME = "finrag-data.tar.gz"
# Paths inside data/ that the deployed app needs: the section index and the XBRL table.
BUNDLE_MEMBERS = (
    "index/section/chunks.jsonl",
    "index/section/embeddings.npy",
    "index/section/meta.json",
    "financials.db",
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package(data_dir: Path, out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for member in BUNDLE_MEMBERS:
            tar.add(data_dir / member, arcname=member)
    return sha256_of(out)


def is_ready(data_dir: Path) -> bool:
    return all((data_dir / m).exists() for m in BUNDLE_MEMBERS)


def install(bundle: Path, data_dir: Path, expected_sha256: str | None) -> None:
    actual = sha256_of(bundle)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"data bundle checksum mismatch: expected {expected_sha256}, got {actual}")
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        names = set(tar.getnames())
        missing = [m for m in BUNDLE_MEMBERS if m not in names]
        if missing:
            raise ValueError(f"data bundle is missing {missing}")
        tar.extractall(data_dir, filter="data")


def ensure_data(data_dir: Path, url: str | None, expected_sha256: str | None) -> None:
    """Download and install the bundle if the data is not already present."""
    if is_ready(data_dir):
        return
    if not url:
        raise RuntimeError("data not built and no DATA_BUNDLE_URL configured")
    logger.info("downloading data bundle from %s", url)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / BUNDLE_NAME
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with bundle.open("wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
        install(bundle, data_dir, expected_sha256)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Package the built data for deployment.")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "dist" / BUNDLE_NAME)
    a = p.parse_args(argv)
    data_dir = get_settings().data_dir
    if not is_ready(data_dir):
        raise SystemExit("data not built yet: run `make data` first")
    digest = package(data_dir, a.out)
    size_mb = a.out.stat().st_size / 1e6
    print(f"Wrote {a.out} ({size_mb:.1f} MB)\nSHA-256: {digest}")


if __name__ == "__main__":
    main()
