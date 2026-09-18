#!/usr/bin/env python3
"""Download MaleCNS SWC skeletons and synapse tables from Janelia public GCS.

Official location: gs://flyem-male-cns/v1.0/
HTTPS: https://storage.googleapis.com/flyem-male-cns/...

Does not download EM imagery, segmentation volumes, or tbar predictions.
Resumes partial files. Safe to re-run.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "malecns_v1"
BUCKET = "https://storage.googleapis.com/flyem-male-cns"
FLAT = f"{BUCKET}/v1.0/connectome-data/flat-connectome"
SWC_PREFIX = "v1.0/segmentation/skeletons-malecns/skeletons-swc"
LIST_API = "https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o"

FEATHERS = (
    (
        "syn-points-male-cns-v1.0-minconf-0.5.feather",
        13_061_489_098,
        "c69d08758de07582035cc8843574493a",
    ),
    (
        "syn-partners-male-cns-v1.0-minconf-0.5.feather",
        6_777_179_098,
        "58efcf712f8c4d4de5f2ad51e97def76",
    ),
)

ANN_CANDIDATES = (
    ROOT / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    DEST / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _md5_file(path: Path) -> str:
    import hashlib

    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(url: str, dest: Path, expected: int | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if dest.exists() and expected and dest.stat().st_size == expected:
        _log(f"already complete: {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    if dest.exists() and expected and dest.stat().st_size != expected:
        _log(f"size mismatch, replacing {dest.name}")
        dest.unlink()
    if dest.exists() and expected is None:
        _log(f"already present: {dest.name}")
        return dest

    existing = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "fly-learns-shiri/malecns-download"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        _log(f"resume {dest.name} from {existing:,}")
    else:
        _log(f"start {dest.name}")

    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    last = t0
    written = existing
    with urllib.request.urlopen(req, timeout=120) as resp, part.open("ab" if existing else "wb") as out:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            now = time.time()
            if now - last >= 5:
                dt = max(now - t0, 1e-6)
                speed = (written - existing) / dt / (1024 * 1024)
                pct = f"{100 * written / expected:.1f}%" if expected else "?"
                _log(f"  {dest.name}  {written:,}  {pct}  {speed:.1f} MB/s")
                last = now
    size = part.stat().st_size
    if expected and size != expected:
        raise RuntimeError(f"{dest.name} size {size} != expected {expected}")
    part.replace(dest)
    _log(f"wrote {dest} ({size:,} bytes)")
    return dest


def list_swc_objects() -> list[dict]:
    items: list[dict] = []
    token = None
    prefix = f"{SWC_PREFIX}/"
    while True:
        url = f"{LIST_API}?prefix={prefix}&maxResults=1000&fields=items(name,size,md5Hash),nextPageToken"
        if token:
            url += f"&pageToken={token}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.load(resp)
        batch = payload.get("items") or []
        items.extend(batch)
        token = payload.get("nextPageToken")
        _log(f"listed {len(items):,} SWC objects")
        if not token:
            break
    return items


def retained_body_ids() -> set[int] | None:
    path = next((p for p in ANN_CANDIDATES if p.exists()), None)
    if path is None:
        return None
    import pyarrow.feather as feather

    table = feather.read_table(path, columns=["bodyId"])
    ids = set(int(v) for v in table.column("bodyId").to_pylist() if v is not None)
    _log(f"retained body IDs from annotations: {len(ids):,}")
    return ids


def _fetch_one(url: str, dest: Path, expected: int | None) -> str:
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_err = "error"
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "fly-learns-shiri/malecns-download"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if expected and part.stat().st_size != expected:
                part.unlink(missing_ok=True)
                last_err = "size-mismatch"
                time.sleep(0.4 * (attempt + 1))
                continue
            part.replace(dest)
            return "ok"
        except urllib.error.HTTPError as err:
            part.unlink(missing_ok=True)
            if err.code == 404:
                return "missing"
            last_err = f"http-{err.code}"
        except Exception as err:
            part.unlink(missing_ok=True)
            last_err = f"error:{type(err).__name__}"
        time.sleep(0.4 * (attempt + 1))
    return last_err


def download_skeletons(swc_dir: Path, workers: int = 24) -> dict:
    swc_dir.mkdir(parents=True, exist_ok=True)
    objects = list_swc_objects()
    wanted = retained_body_ids()
    selected = []
    for obj in objects:
        name = str(obj.get("name") or "")
        if not name.endswith(".swc"):
            continue
        stem = Path(name).stem
        try:
            body_id = int(stem)
        except ValueError:
            continue
        if wanted is not None and body_id not in wanted:
            continue
        selected.append(obj)
    _log(f"SWC files to fetch: {len(selected):,} (of {len(objects):,} in bucket)")

    counts = {"ok": 0, "skip": 0, "missing": 0, "size-mismatch": 0, "error": 0}
    errors: list[str] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for obj in selected:
            name = str(obj["name"])
            dest = swc_dir / Path(name).name
            url = f"{BUCKET}/{name}"
            size = int(obj.get("size") or 0) or None
            futures[pool.submit(_fetch_one, url, dest, size)] = Path(name).name
        for fut in as_completed(futures):
            status = fut.result()
            done += 1
            key = status if status in counts else "error"
            counts[key] = counts.get(key, 0) + 1
            if key == "error":
                errors.append(f"{futures[fut]}:{status}")
            if done % 2000 == 0 or done == len(selected):
                dt = max(time.time() - t0, 1e-6)
                _log(
                    f"  skeletons {done:,}/{len(selected):,}  "
                    f"{done / dt:.0f}/s  {counts}"
                )
    return {"counts": counts, "n_selected": len(selected), "n_listed": len(objects), "errors": errors[:40]}


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    report: dict = {"feathers": {}, "skeletons": {}}
    swc_dir = DEST / "skeletons-swc"
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {
            pool.submit(download_resumable, f"{FLAT}/{name}", DEST / name, size): (name, size, md5_hex)
            for name, size, md5_hex in FEATHERS
        }
        skel_fut = pool.submit(download_skeletons, swc_dir)
        bad = False
        for fut in as_completed(jobs):
            name, size, md5_hex = jobs[fut]
            path = fut.result()
            got = _md5_file(path)
            ok = got == md5_hex
            report["feathers"][name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "md5": got,
                "ok": ok,
            }
            _log(f"md5 {name}: {got} {'OK' if ok else 'MISMATCH expected ' + md5_hex}")
            if not ok:
                bad = True
        report["skeletons"] = skel_fut.result()
    n_files = len(list(swc_dir.glob("*.swc")))
    report["skeletons"]["files_on_disk"] = n_files
    out = DEST / "anatomy_download.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    _log(f"report {out}")
    _log(f"SWC on disk: {n_files:,}")
    if bad:
        return 1
    return 0 if report["skeletons"]["counts"].get("error", 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
