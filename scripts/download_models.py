#!/usr/bin/env python3
"""Download the trained models (ONNX) from this repository's GitHub release and check their SHA-256 checksums.

    python scripts/download_models.py                 # all versions into ./models
    python scripts/download_models.py --out my_models

The release is `models-v1.0`. While the repository is private, set GITHUB_TOKEN to a token with read access.
Standard library only.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO = "Maz2580/urban-heat-tmrt-surrogate"
TAG = "models-v1.0"


def get(url, accept="application/vnd.github+json"):
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "download-models"})
    if os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    return urllib.request.urlopen(req)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="models", help="folder for the model files (default: models)")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    assets = {a["name"]: a["url"] for a in json.load(get(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"))["assets"]}
    sums = get(assets["SHA256SUMS.txt"], "application/octet-stream").read().decode()
    expected = dict(reversed(line.split()) for line in sums.splitlines() if line.strip())
    bad = []
    for name, digest in expected.items():
        path = os.path.join(out, name)
        if not (os.path.exists(path) and hashlib.sha256(open(path, "rb").read()).hexdigest() == digest):
            print(f"downloading {name} ...", flush=True)
            with get(assets[name], "application/octet-stream") as r, open(path, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        ok = hashlib.sha256(open(path, "rb").read()).hexdigest() == digest
        print(f"  {'OK ' if ok else 'BAD'} {name}")
        if not ok:
            bad.append(name)
    if bad:
        sys.exit(f"checksum mismatch: {bad}")
    print(f"all {len(expected)} files present and verified in {out}/")


if __name__ == "__main__":
    main()
