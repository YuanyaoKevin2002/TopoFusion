from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from topofusion_v3.config import deep_get, load_config


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _scalar(data, name: str, default: int | float = 0) -> int | float:
    value = getattr(data, name, default)
    if torch.is_tensor(value):
        return value.view(-1)[0].item()
    return value


def connected_components(n: int, edge_index: torch.Tensor) -> Tuple[np.ndarray, int, float]:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64), 0, 0.0
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra = find(int(a))
        rb = find(int(b))
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    if edge_index.numel():
        src = edge_index[0].detach().cpu().numpy()
        dst = edge_index[1].detach().cpu().numpy()
        for a, b in zip(src, dst):
            union(int(a), int(b))
    roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    _, comp_ids, counts = np.unique(roots, return_inverse=True, return_counts=True)
    largest_frac = float(counts.max() / n) if counts.size else 0.0
    return comp_ids.astype(np.int64), int(counts.size), largest_frac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-csv", default="")
    parser.add_argument("--datasets", default="TRAIN_COHORT")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_csv = Path(args.manifest_csv or deep_get(cfg, "manifest.out_csv"))
    rows = read_manifest(manifest_csv)
    datasets = {x.strip().upper() for x in args.datasets.split(",") if x.strip()}
    splits = {x.strip() for x in args.splits.split(",") if x.strip()} if args.splits else None
    rows = [r for r in rows if r["dataset"].upper() in datasets and (splits is None or r["split"] in splits)]
    rows = [r for r in rows if Path(r["cache_path"]).exists()]
    if args.limit > 0:
        rows = rows[: args.limit]

    out_rows: List[Dict[str, object]] = []
    for row in rows:
        data = torch.load(row["cache_path"], map_location="cpu", weights_only=False)
        n = int(data.num_nodes)
        _, comp_count, largest_frac = connected_components(n, data.edge_index)
        tissue = data.tissue_probs4.detach().cpu().numpy()
        dom = tissue.argmax(axis=1) if tissue.size else np.zeros((0,), dtype=np.int64)
        denom = max(n, 1)
        dom_frac = [(dom == i).sum() / denom for i in range(4)]
        probs_mean = tissue.mean(axis=0).tolist() if tissue.size else [0.0, 0.0, 0.0, 0.0]
        out_rows.append(
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "slide_id": row["slide_id"],
                "cms_label": row["cms_label"],
                "label_id": row["label_id"],
                "nodes_raw": int(_scalar(data, "n_nodes_raw", n)),
                "nodes_core": int(_scalar(data, "n_nodes_core", n)),
                "nodes_kept": n,
                "nodes_lym_rescued": int(_scalar(data, "n_nodes_lym_rescued", 0)),
                "edges": int(data.edge_index.size(1)),
                "edge_attr_dim": int(data.edge_attr_raw.size(1)) if data.edge_attr_raw.numel() else 0,
                "component_count_attr_initial": int(_scalar(data, "component_count_initial", comp_count)),
                "component_count_attr_final": int(_scalar(data, "component_count_final", comp_count)),
                "component_count_verified": comp_count,
                "largest_component_frac": largest_frac,
                "bridge_edges_added": int(_scalar(data, "component_bridge_edges_added", 0)),
                "bridge_pairs_added": int(_scalar(data, "component_bridge_pairs_added", 0)),
                "dom_TUM_frac": dom_frac[0],
                "dom_STR_frac": dom_frac[1],
                "dom_LYM_frac": dom_frac[2],
                "dom_MUC_frac": dom_frac[3],
                "mean_TUM_prob": probs_mean[0],
                "mean_STR_prob": probs_mean[1],
                "mean_LYM_prob": probs_mean[2],
                "mean_MUC_prob": probs_mean[3],
            }
        )

    fields = [
        "dataset",
        "split",
        "slide_id",
        "cms_label",
        "label_id",
        "nodes_raw",
        "nodes_core",
        "nodes_kept",
        "nodes_lym_rescued",
        "edges",
        "edge_attr_dim",
        "component_count_attr_initial",
        "component_count_attr_final",
        "component_count_verified",
        "largest_component_frac",
        "bridge_edges_added",
        "bridge_pairs_added",
        "dom_TUM_frac",
        "dom_STR_frac",
        "dom_LYM_frac",
        "dom_MUC_frac",
        "mean_TUM_prob",
        "mean_STR_prob",
        "mean_LYM_prob",
        "mean_MUC_prob",
    ]
    write_csv(Path(args.out_csv), out_rows, fields)

    summary: Dict[str, object] = {"n": len(out_rows)}
    if out_rows:
        for split in sorted({str(r["split"]) for r in out_rows}):
            subset = [r for r in out_rows if str(r["split"]) == split]
            comps = np.asarray([float(r["component_count_verified"]) for r in subset], dtype=np.float32)
            lym = np.asarray([float(r["dom_LYM_frac"]) for r in subset], dtype=np.float32)
            largest = np.asarray([float(r["largest_component_frac"]) for r in subset], dtype=np.float32)
            bridges = np.asarray([float(r["bridge_edges_added"]) for r in subset], dtype=np.float32)
            summary[split] = {
                "n": len(subset),
                "mean_components": float(comps.mean()),
                "median_components": float(np.median(comps)),
                "max_components": float(comps.max()),
                "pct_components_le5": float((comps <= 5).mean()),
                "mean_dom_LYM_frac": float(lym.mean()),
                "min_dom_LYM_frac": float(lym.min()),
                "mean_largest_component_frac": float(largest.mean()),
                "mean_bridge_edges_added": float(bridges.mean()),
            }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
