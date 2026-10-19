from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from topofusion_v3.config import deep_get, load_config
from topofusion_v3.constants import CMS_TO_ID


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def make_TRAIN_COHORT_patient_split(rows: List[Dict[str, object]], val_fraction: float, seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    patient_to_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["dataset"] == "TRAIN_COHORT":
            patient_to_rows[str(row["patient_id"])].append(row)

    label_to_patients: Dict[str, List[str]] = defaultdict(list)
    for patient, prows in patient_to_rows.items():
        labels = [str(r["cms_label"]) for r in prows]
        label = max(set(labels), key=labels.count)
        label_to_patients[label].append(patient)

    val_patients = set()
    for label, patients in sorted(label_to_patients.items()):
        patients = list(patients)
        rng.shuffle(patients)
        n_val = max(1, int(round(len(patients) * val_fraction))) if len(patients) >= 4 else 1
        val_patients.update(patients[:n_val])

    splits = {"train": [], "val": [], "EXTERNAL_COHORT_A_external": [], "EXTERNAL_COHORT_B_external": []}
    for row in rows:
        sid = str(row["slide_id"])
        dataset = str(row["dataset"])
        if dataset == "TRAIN_COHORT":
            split = "val" if str(row["patient_id"]) in val_patients else "train"
        elif dataset == "EXTERNAL_COHORT_A":
            split = "EXTERNAL_COHORT_A_external"
        elif dataset == "EXTERNAL_COHORT_B":
            split = "EXTERNAL_COHORT_B_external"
        else:
            continue
        row["split"] = split
        splits[split].append(sid)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-csv", default="")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--split-json", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    audit_csv = Path(args.audit_csv or deep_get(cfg, "audit_csv"))
    out_csv = Path(args.out_csv or deep_get(cfg, "manifest.out_csv"))
    split_json = Path(args.split_json or deep_get(cfg, "manifest.split_json"))
    seed = int(args.seed if args.seed is not None else deep_get(cfg, "manifest.seed", 42))
    val_fraction = float(
        args.val_fraction if args.val_fraction is not None else deep_get(cfg, "manifest.val_fraction", 0.2)
    )

    raw_rows = read_csv(audit_csv)
    rows: List[Dict[str, object]] = []
    for row in raw_rows:
        if not normalize_bool(row.get("usable_all")):
            continue
        cms = str(row.get("cms_label", "")).strip()
        if cms not in CMS_TO_ID:
            continue
        slide_id = Path(str(row["resolved_h5_path"])).stem
        cache_root_key = "TRAIN_COHORT_root" if row["dataset"] == "TRAIN_COHORT" else "external_root"
        cache_root = Path(deep_get(cfg, f"cache.{cache_root_key}"))
        rows.append(
            {
                "dataset": row["dataset"],
                "slide_id": slide_id,
                "sample_id": row.get("sample_id", ""),
                "patient_id": row.get("patient_id", ""),
                "cms_label": cms,
                "label_id": CMS_TO_ID[cms],
                "h5_path": row["resolved_h5_path"],
                "cellvit_path": row["cellvit_path"],
                "cache_path": str(cache_root / str(row["dataset"]) / f"{slide_id}.pt"),
                "split": "",
            }
        )

    splits = make_TRAIN_COHORT_patient_split(rows, val_fraction=val_fraction, seed=seed)
    fields = [
        "dataset",
        "slide_id",
        "sample_id",
        "patient_id",
        "cms_label",
        "label_id",
        "split",
        "h5_path",
        "cellvit_path",
        "cache_path",
    ]
    write_csv(out_csv, rows, fields)
    split_json.parent.mkdir(parents=True, exist_ok=True)
    split_json.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")

    summary = defaultdict(int)
    for row in rows:
        summary[(row["dataset"], row["split"], row["cms_label"])] += 1
    print(f"wrote manifest: {out_csv}")
    print(f"wrote splits: {split_json}")
    for key, value in sorted(summary.items()):
        print(",".join(map(str, key)), value)


if __name__ == "__main__":
    main()

