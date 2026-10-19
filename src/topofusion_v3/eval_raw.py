from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from topofusion_v3.config import deep_get, load_config
from topofusion_v3.graph_building import build_graph_from_h5_and_cellvit
from topofusion_v3.metrics import classification_metrics
from topofusion_v3.model import CMSGraphModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="EXTERNAL_COHORT_B_external")
    parser.add_argument("--dataset", default="EXTERNAL_COHORT_B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = list(csv.DictReader(open(deep_get(cfg, "manifest.out_csv"), newline="", encoding="utf-8")))
    rows = [r for r in rows if r["split"] == args.split and r["dataset"].upper() == args.dataset.upper()]
    if args.limit > 0:
        rows = rows[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMSGraphModel(dict(deep_get(cfg, "model", {}))).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    cache_cfg = dict(deep_get(cfg, "cache", {}))
    y_true, logits_all, pred_rows = [], [], []
    with torch.no_grad():
        for idx, row in enumerate(rows, start=1):
            data = build_graph_from_h5_and_cellvit(
                h5_path=row["h5_path"],
                cellvit_path=row["cellvit_path"],
                label_id=int(row["label_id"]),
                dataset=row["dataset"],
                slide_id=row["slide_id"],
                patient_id=row["patient_id"],
                cfg=cache_cfg,
            ).to(device)
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            logits = model(data)
            logits_np = logits.detach().cpu().numpy()[0]
            y = int(row["label_id"])
            y_true.append(y)
            logits_all.append(logits_np)
            pred_rows.append(
                {
                    "slide_id": row["slide_id"],
                    "dataset": row["dataset"],
                    "label_id": y,
                    "pred_id": int(np.argmax(logits_np)),
                    "n_nodes": int(data.num_nodes),
                }
            )
            if idx % 10 == 0:
                print(f"raw_eval processed {idx}/{len(rows)}", flush=True)

    logits_arr = np.vstack(logits_all) if logits_all else np.zeros((0, 4), dtype=np.float32)
    metrics = classification_metrics(np.asarray(y_true, dtype=np.int64), logits_arr) if len(y_true) else {}
    payload = {"split": args.split, "dataset": args.dataset, "n": len(y_true), "metrics": metrics}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["slide_id", "dataset", "label_id", "pred_id", "n_nodes"])
            writer.writeheader()
            writer.writerows(pred_rows)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

