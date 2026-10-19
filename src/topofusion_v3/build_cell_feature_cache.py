from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import h5py

from topofusion_v3.cell_features import load_or_compute_cell_features_for_patches
from topofusion_v3.config import deep_get, load_config


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-csv", default="")
    parser.add_argument("--datasets", default="EXTERNAL_COHORT_B")
    parser.add_argument("--splits", default="EXTERNAL_COHORT_B_external")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_csv = Path(args.manifest_csv or deep_get(cfg, "manifest.out_csv"))
    cache_cfg = dict(deep_get(cfg, "cache", {}))
    root = Path(cache_cfg["cell_feature_cache_root"])
    datasets = {x.strip().upper() for x in args.datasets.split(",") if x.strip()}
    splits = {x.strip() for x in args.splits.split(",") if x.strip()}

    rows = [
        r for r in read_manifest(manifest_csv)
        if r["dataset"].upper() in datasets and r["split"] in splits
    ]
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_index]
    if args.limit > 0:
        rows = rows[: args.limit]

    summary = {"attempted": 0, "built_or_existing": 0, "failed": 0, "failures": []}
    for row in rows:
        slide_id = row["slide_id"]
        out_path = root / row["dataset"] / f"{slide_id}.npz"
        summary["attempted"] += 1
        try:
            with h5py.File(row["h5_path"], "r") as f:
                coords = f["coords_mpp025"][:]
            counts, topo, meta = load_or_compute_cell_features_for_patches(
                row["cellvit_path"],
                coords,
                cache_path=out_path,
                use_cache=True,
                write_cache=True,
                overwrite=args.overwrite,
                patch_size=float(cache_cfg.get("patch_size_mpp025", 256.0)),
                buffer_px=float(cache_cfg.get("cell_buffer_px", 64.0)),
                radius_px=float(cache_cfg.get("cell_radius_px", 50.0)),
                cap_ext=int(cache_cfg.get("cell_cap_ext", 1000)),
                seed=int(deep_get(cfg, "manifest.seed", 42)),
            )
            summary["built_or_existing"] += 1
            print(
                f"cell_cache {row['dataset']} {slide_id} patches={counts.shape[0]} topo_dim={topo.shape[1]} source={meta.get('source')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            summary["failed"] += 1
            summary["failures"].append({"dataset": row["dataset"], "slide_id": slide_id, "error": repr(exc)})
            print(f"FAILED cell_cache {row['dataset']} {slide_id}: {exc}", flush=True)

    summary["num_shards"] = args.num_shards
    summary["shard_index"] = args.shard_index
    out_summary = Path(args.summary_json) if args.summary_json else Path(deep_get(cfg, "project_root")) / "outputs" / "cell_cache_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

