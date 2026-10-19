#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torch

from topofusion_v3.baselines.multitask_utils import feature_cache_path_for_row, select_manifest_rows


def parse_expected_counts(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, val = item.split(":")
        out[key.strip()] = int(val)
    return out


def split_counts(rows: list[dict]) -> dict[str, int]:
    out = {"train": 0, "val": 0, "EXTERNAL_COHORT_A_external": 0}
    for row in rows:
        split = row.get("split", "")
        if split in out:
            out[split] += 1
    return out


def cache_one(task: tuple[dict, str, bool]) -> dict:
    row, cache_dir, overwrite = task
    out_path = feature_cache_path_for_row(cache_dir, row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return {
            "status": "exists",
            "slide_id": row.get("slide_id", ""),
            "split": row.get("split", ""),
            "dataset": row.get("dataset", ""),
            "cache_path": str(out_path),
            "source_h5": row.get("h5_path", ""),
            "n_patches": "",
            "dim": "",
            "seconds": 0.0,
        }

    t0 = time.time()
    h5_path = row["h5_path"]
    with h5py.File(h5_path, "r") as f:
        feats = np.asarray(f["features"][:], dtype=np.float32)

    payload = {
        "features": torch.from_numpy(feats),
        "slide_id": row.get("slide_id", ""),
        "sample_id": row.get("sample_id", ""),
        "patient_id": row.get("patient_id", ""),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "source_h5": h5_path,
        "n_patches": int(feats.shape[0]),
        "dim": int(feats.shape[1]) if feats.ndim == 2 else None,
    }
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    return {
        "status": "written",
        "slide_id": row.get("slide_id", ""),
        "split": row.get("split", ""),
        "dataset": row.get("dataset", ""),
        "cache_path": str(out_path),
        "source_h5": h5_path,
        "n_patches": int(feats.shape[0]),
        "dim": int(feats.shape[1]) if feats.ndim == 2 else None,
        "seconds": round(time.time() - t0, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="${PROJECT_ROOT}/manifests/topofusion_v3_connectfix_manifest.csv")
    parser.add_argument("--clinical-dir", default="${DATA_ROOT}/gnn-crc/data/clinical")
    parser.add_argument("--cache-dir", default="${PROJECT_ROOT}/cache/baseline_features_40x")
    parser.add_argument("--splits", default="train,val,EXTERNAL_COHORT_A_external")
    parser.add_argument("--expected-counts", default="train:385,val:96,EXTERNAL_COHORT_A_external:98")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    rows = select_manifest_rows(
        args.manifest,
        splits,
        args.clinical_dir,
        require_h5=True,
        require_graph=False,
        deduplicate_patients=True,
    )
    counts = split_counts(rows)
    expected = parse_expected_counts(args.expected_counts)
    actual = {key: counts.get(key, 0) for key in expected}
    if actual != expected:
        raise RuntimeError(f"split count mismatch: expected={expected} actual={counts}")

    out_csv = Path(args.out_csv) if args.out_csv else Path(args.cache_dir) / "cache_manifest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "cache_start", "cache_dir": args.cache_dir, "counts": counts, "workers": args.workers}, sort_keys=True), flush=True)

    tasks = [(row, args.cache_dir, bool(args.overwrite)) for row in rows]
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(cache_one, task) for task in tasks]
        for done, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if done == 1 or done % 25 == 0 or done == len(futures):
                written = sum(1 for item in results if item["status"] == "written")
                exists = sum(1 for item in results if item["status"] == "exists")
                print(json.dumps({"event": "cache_progress", "done": done, "total": len(futures), "written": written, "exists": exists, "last": res}, sort_keys=True), flush=True)

    fieldnames = ["status", "dataset", "split", "slide_id", "n_patches", "dim", "seconds", "cache_path", "source_h5"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "event": "cache_done",
        "cache_dir": args.cache_dir,
        "manifest": str(out_csv),
        "total": len(results),
        "written": sum(1 for item in results if item["status"] == "written"),
        "exists": sum(1 for item in results if item["status"] == "exists"),
        "counts": counts,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
