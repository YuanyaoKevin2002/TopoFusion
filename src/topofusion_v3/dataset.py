from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class GraphCacheDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, split: str, limit: Optional[int] = None):
        self.manifest_csv = Path(manifest_csv)
        with self.manifest_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.rows: List[Dict[str, str]] = [r for r in rows if r["split"] == split and Path(r["cache_path"]).exists()]
        if limit is not None and int(limit) > 0:
            self.rows = self.rows[: int(limit)]
        if len(self.rows) == 0:
            raise RuntimeError(f"No cached graphs found for split={split} in {manifest_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        data = torch.load(row["cache_path"], weights_only=False)
        data.slide_id = row["slide_id"]
        data.dataset = row["dataset"]
        return data

    @property
    def labels(self) -> List[int]:
        return [int(r["label_id"]) for r in self.rows]

    @property
    def slide_ids(self) -> List[str]:
        return [r["slide_id"] for r in self.rows]

