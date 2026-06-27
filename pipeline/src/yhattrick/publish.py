"""Publish the heavy per-game timeline JSON to object storage (Cloudflare R2).

Small data (the games index, player ratings + cards) ships inside the web deploy; the per-game
files (web/public/data/game/*.json, ~850 MB) are synced here to R2, which the production site
reads via NEXT_PUBLIC_GAME_DATA_BASE=<public-url>/game.

S3-compatible credentials are read from the environment (never commit them):
  R2_ENDPOINT            https://<accountid>.r2.cloudflarestorage.com
  R2_BUCKET              e.g. yhattrick-data
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY

Usage:
  uv run python -m yhattrick.publish               # upload game/*.json (skip unchanged by size)
  uv run python -m yhattrick.publish --force       # re-upload everything
  uv run python -m yhattrick.publish --set-cors    # also (re)apply a public GET CORS policy
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config as C

PREFIX = "game"  # objects live at game/<id>.json to match NEXT_PUBLIC_GAME_DATA_BASE=.../game
_ENV = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")


def _client():
    try:
        import boto3
    except ImportError:
        raise SystemExit("boto3 not installed — run `uv add boto3` in pipeline/")
    missing = [k for k in _ENV if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing env var(s): {', '.join(missing)} (see module docstring)")
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    return s3, os.environ["R2_BUCKET"]


def _remote_sizes(s3, bucket: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": f"{PREFIX}/"}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            sizes[o["Key"]] = o["Size"]
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            return sizes


def _set_cors(s3, bucket: str) -> None:
    """Allow browsers on any origin to GET the public JSON (read-only data CDN)."""
    s3.put_bucket_cors(Bucket=bucket, CORSConfiguration={
        "CORSRules": [{"AllowedMethods": ["GET", "HEAD"], "AllowedOrigins": ["*"],
                       "AllowedHeaders": ["*"], "MaxAgeSeconds": 3600}]
    })
    print(f"[publish] applied public GET CORS to {bucket}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Publish per-game JSON to Cloudflare R2")
    p.add_argument("--force", action="store_true", help="re-upload even if a same-size object exists")
    p.add_argument("--set-cors", action="store_true", help="also (re)apply the public GET CORS policy")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args(argv)

    src = C.WEB_DATA / "game"
    files = sorted(src.glob("*.json"))
    if not files:
        raise SystemExit(f"no game JSON in {src} — run `make games` first")

    s3, bucket = _client()
    if args.set_cors:
        _set_cors(s3, bucket)

    remote = {} if args.force else _remote_sizes(s3, bucket)
    todo = [f for f in files if remote.get(f"{PREFIX}/{f.name}") != f.stat().st_size]
    print(f"[publish] {len(files):,} local · {len(remote):,} remote · uploading {len(todo):,}")

    def up(f):
        s3.upload_file(str(f), bucket, f"{PREFIX}/{f.name}",
                       ExtraArgs={"ContentType": "application/json"})

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed(ex.submit(up, f) for f in todo):
            fut.result()
            done += 1
            if done % 500 == 0:
                print(f"  {done:,}/{len(todo):,}")
    print(f"[publish] uploaded {done:,} files -> r2://{bucket}/{PREFIX}/")


if __name__ == "__main__":
    main()
