from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch

from topofusion_v3.config import deep_get, load_config
from topofusion_v3.graph_building import build_graph_from_h5_and_cellvit


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _scalar(data, name: str, default: int | float = 0) -> int | float:
    value = getattr(data, name, default)
    if torch.is_tensor(value):
        return value.view(-1)[0].item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-csv", default="")
    parser.add_argument("--datasets", default="TRAIN_COHORT,EXTERNAL_COHORT_A,EXTERNAL_COHORT_B")
    parser.add_argument("--splits", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_csv = Path(args.manifest_csv or deep_get(cfg, "manifest.out_csv"))
    cache_cfg = dict(deep_get(cfg, "cache", {}))
    cache_cfg["seed"] = int(deep_get(cfg, "manifest.seed", 42))
    rows = read_manifest(manifest_csv)
    datasets = {x.strip().upper() for x in args.datasets.split(",") if x.strip()}
    splits = {x.strip() for x in args.splits.split(",") if x.strip()} if args.splits else None
    rows = [r for r in rows if r["dataset"].upper() in datasets and (splits is None or r["split"] in splits)]
    if int(args.num_shards) > 1:
        n_shards = int(args.num_shards)
        shard_idx = int(args.shard_index)
        if shard_idx < 0 or shard_idx >= n_shards:
            raise ValueError(f"shard-index must be in [0,{n_shards - 1}], got {shard_idx}")
        rows = [r for i, r in enumerate(rows) if (i % n_shards) == shard_idx]
    if args.limit > 0:
        rows = rows[: args.limit]

    summary = {
        "attempted": 0,
        "built": 0,
        "skipped": 0,
        "failed": 0,
        "failures": [],
        "component_count_final": [],
        "lym_dominant_fraction": [],
        "bridge_edges_added": [],
        "nodes_after": [],
        "downsample_nodes_before": [],
        "downsample_nodes_after": [],
        "downsample_nodes_dropped": [],
    }
    for row in rows:
        out_path = Path(row["cache_path"])
        if out_path.exists() and not args.overwrite:
            summary["skipped"] += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary["attempted"] += 1
        try:
            data = build_graph_from_h5_and_cellvit(
                h5_path=row["h5_path"],
                cellvit_path=row["cellvit_path"],
                label_id=int(row["label_id"]),
                dataset=row["dataset"],
                slide_id=row["slide_id"],
                patient_id=row["patient_id"],
                cfg=cache_cfg,
            )
            torch.save(data, out_path)
            summary["built"] += 1
            comps = int(_scalar(data, "component_count_final", -1))
            bridges = int(_scalar(data, "component_bridge_edges_added", 0))
            lym_frac = float(data.tissue_dom_frac[2].item()) if hasattr(data, "tissue_dom_frac") else float("nan")
            summary["component_count_final"].append(comps)
            summary["lym_dominant_fraction"].append(lym_frac)
            ds_before = int(_scalar(data, "downsample_nodes_before", int(data.num_nodes)))
            ds_after = int(_scalar(data, "downsample_nodes_after", int(data.num_nodes)))
            ds_dropped = int(_scalar(data, "downsample_nodes_dropped", 0))
            summary["bridge_edges_added"].append(bridges)
            summary["nodes_after"].append(int(data.num_nodes))
            summary["downsample_nodes_before"].append(ds_before)
            summary["downsample_nodes_after"].append(ds_after)
            summary["downsample_nodes_dropped"].append(ds_dropped)
            print(
                "built "
                f"{row['dataset']} {row['slide_id']} "
                f"nodes={data.num_nodes} ds={ds_before}->{ds_after} drop={ds_dropped} "
                f"edges={data.edge_index.size(1)} "
                f"components={comps} lym_dom={lym_frac:.4f} bridges={bridges}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            summary["failed"] += 1
            item = {"slide_id": row["slide_id"], "dataset": row["dataset"], "error": repr(exc)}
            summary["failures"].append(item)
            print(f"FAILED {row['dataset']} {row['slide_id']}: {exc}", flush=True)

    if summary["component_count_final"]:
        comp_arr = torch.tensor(summary["component_count_final"], dtype=torch.float32)
        lym_arr = torch.tensor(summary["lym_dominant_fraction"], dtype=torch.float32)
        summary["mean_components_final"] = float(comp_arr.mean().item())
        summary["median_components_final"] = float(comp_arr.median().item())
        summary["mean_lym_dominant_fraction"] = float(lym_arr.mean().item())
        summary["min_lym_dominant_fraction"] = float(lym_arr.min().item())
    if summary["nodes_after"]:
        node_arr = torch.tensor(summary["nodes_after"], dtype=torch.float32)
        before_arr = torch.tensor(summary["downsample_nodes_before"], dtype=torch.float32)
        drop_arr = torch.tensor(summary["downsample_nodes_dropped"], dtype=torch.float32)
        summary["mean_nodes_after"] = float(node_arr.mean().item())
        summary["median_nodes_after"] = float(node_arr.median().item())
        summary["max_nodes_after"] = int(node_arr.max().item())
        summary["mean_downsample_nodes_before"] = float(before_arr.mean().item())
        summary["mean_downsample_nodes_dropped"] = float(drop_arr.mean().item())
    out_summary = (
        Path(args.summary_json)
        if args.summary_json
        else Path(deep_get(cfg, "project_root")) / "outputs" / "cache_build_summary.json"
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary["num_shards"] = int(args.num_shards)
    summary["shard_index"] = int(args.shard_index)
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
