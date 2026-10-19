from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


CLASSIFICATION_TASKS = ("cms", "msi", "pn", "cin")
SURVIVAL_TASKS = ("os", "dfs")
ALL_TASKS = CLASSIFICATION_TASKS + SURVIVAL_TASKS
CMS_TO_ID = {"CMS1": 0, "CMS2": 1, "CMS3": 2, "CMS4": 3}


def _clean(value: object) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if value.lower() in {"", "nan", "none", "na", "n/a", "null", "not reported", "unknown"}:
        return ""
    return value


def _float(value: object) -> Optional[float]:
    value = _clean(value)
    if not value:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _int_binary(value: object) -> Optional[int]:
    x = _float(value)
    if x is None:
        return None
    if int(x) == 1:
        return 1
    if int(x) == 0:
        return 0
    return None


def _cms_label(row: Dict[str, str]) -> Optional[int]:
    for key in ("label", "cms_from_rna", "cms_raw", "cms_label"):
        value = _clean(row.get(key)).upper()
        if value in CMS_TO_ID:
            return CMS_TO_ID[value]
    return None


def _msi_label(dataset: str, row: Dict[str, str]) -> Optional[int]:
    if dataset == "TRAIN_COHORT":
        status = _clean(row.get("MSI_status") or row.get("msi_status")).upper()
        if status:
            if "MSI-H" in status or status in {"MSI_H", "MSIH"}:
                return 1
            if "MSS" in status or "MSI-L" in status or "MSIL" in status:
                return 0
        sensor = _float(row.get("msisensor_score"))
        if sensor is not None:
            return int(sensor >= 10.0)
        mantis = _float(row.get("msi_mantis_score"))
        if mantis is not None:
            return int(mantis >= 0.4)
        return None

    if dataset == "EXTERNAL_COHORT_A":
        status = _clean(row.get("msi_status")).upper()
        if "MSI-H" in status or status in {"MSIH", "MSI_H"}:
            return 1
        if "MSS" in status or "MSI-L" in status or "MSIL" in status:
            return 0
        return None

    if dataset == "EXTERNAL_COHORT_B":
        status = _clean(row.get("mmr_status")).lower()
        if not status:
            return None
        if status in {"1", "1.0"} or "dmmr" in status or "deficient" in status:
            return 1
        if status in {"0", "0.0"} or "pmmr" in status or "proficient" in status:
            return 0
        return None
    return None


def _pn_label(row: Dict[str, str]) -> Optional[int]:
    value = _clean(row.get("pN") or row.get("pathologic_N") or row.get("pathologic_n")).upper()
    if not value or value in {"NX", "N X"}:
        return None
    value = value.replace(" ", "")
    if value.startswith("N0"):
        return 0
    if value.startswith(("N1", "N2", "N3")):
        return 1
    return None


def _survival_pair(time_value: object, event_value: object) -> tuple[Optional[float], Optional[int]]:
    time = _float(time_value)
    event = _int_binary(event_value)
    if time is None or time <= 0 or event is None:
        return None, None
    return float(time), int(event)


def _os_label(dataset: str, row: Dict[str, str]) -> tuple[Optional[float], Optional[int]]:
    if dataset == "TRAIN_COHORT":
        return _survival_pair(row.get("OS.time"), row.get("OS"))
    if dataset == "EXTERNAL_COHORT_A":
        return _survival_pair(row.get("os_days"), row.get("os_event"))
    if dataset == "EXTERNAL_COHORT_B":
        return _survival_pair(row.get("os"), row.get("os.event"))
    return None, None


def _dfs_label(dataset: str, row: Dict[str, str]) -> tuple[Optional[float], Optional[int]]:
    if dataset == "TRAIN_COHORT":
        return _survival_pair(row.get("PFI.time"), row.get("PFI"))
    if dataset == "EXTERNAL_COHORT_A":
        return _survival_pair(row.get("dfs_days"), row.get("dfs_event"))
    if dataset == "EXTERNAL_COHORT_B":
        return _survival_pair(row.get("dfs"), row.get("dfs.event"))
    return None, None


def _cin_label(dataset: str, row: Dict[str, str]) -> Optional[int]:
    if dataset == "TRAIN_COHORT":
        subtype = _clean(row.get("molecular_subtype")).upper()
        if not subtype:
            return None
        if "CIN" in subtype:
            return 1
        if any(token in subtype for token in ("MSI", "GS", "POLE")):
            return 0
        return None
    if dataset == "EXTERNAL_COHORT_A":
        cin = _clean(row.get("CIN")).lower()
        if "high" in cin:
            return 1
        if "low" in cin:
            return 0
        return None
    return None


def _labels_from_clinical_row(dataset: str, row: Dict[str, str]) -> Dict[str, Optional[float | int]]:
    os_time, os_event = _os_label(dataset, row)
    dfs_time, dfs_event = _dfs_label(dataset, row)
    return {
        "cms": _cms_label(row),
        "msi": _msi_label(dataset, row),
        "pn": _pn_label(row) if dataset in {"TRAIN_COHORT", "EXTERNAL_COHORT_A"} else None,
        "os_time": os_time,
        "os_event": os_event,
        "dfs_time": dfs_time,
        "dfs_event": dfs_event,
        "cin": _cin_label(dataset, row),
    }


def _clinical_files(clinical_dir: str | Path) -> Dict[str, Path]:
    root = Path(clinical_dir)
    return {
        "TRAIN_COHORT": root / "TRAIN_COHORT_clinical_aligned.csv",
        "EXTERNAL_COHORT_A": root / "EXTERNAL_COHORT_A_clinical_aligned.csv",
        "EXTERNAL_COHORT_B": root / "EXTERNAL_COHORT_B_clinical_aligned.csv",
    }


def load_multitask_label_index(clinical_dir: str | Path) -> Dict[tuple[str, str, str], Dict[str, Optional[float | int]]]:
    """Return labels indexed by (dataset, key_type, key_value)."""

    index: Dict[tuple[str, str, str], Dict[str, Optional[float | int]]] = {}
    for dataset, path in _clinical_files(clinical_dir).items():
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels = _labels_from_clinical_row(dataset, row)
                for key_type, col in (
                    ("sample_id", "sample_id"),
                    ("patient_id", "patient_id"),
                    ("patient_id", "hospital_id"),
                    ("slide_id", "slide_id"),
                    ("path_id", "path_id"),
                ):
                    key = _clean(row.get(col))
                    if key:
                        index[(dataset, key_type, key)] = labels
    return index


def labels_for_manifest_row(
    row: Dict[str, str],
    label_index: Dict[tuple[str, str, str], Dict[str, Optional[float | int]]],
) -> Dict[str, Optional[float | int]]:
    dataset = row["dataset"]
    labels: Dict[str, Optional[float | int]] = {
        "cms": int(row["label_id"]) if _clean(row.get("label_id")) else None,
        "msi": None,
        "pn": None,
        "os_time": None,
        "os_event": None,
        "dfs_time": None,
        "dfs_event": None,
        "cin": None,
    }
    for key_type, col in (
        ("sample_id", "sample_id"),
        ("patient_id", "patient_id"),
        ("slide_id", "slide_id"),
    ):
        key = _clean(row.get(col))
        if not key:
            continue
        found = label_index.get((dataset, key_type, key))
        if found:
            labels.update({k: v for k, v in found.items() if v is not None})
            break
    if labels["cms"] is None and _clean(row.get("cms_label")).upper() in CMS_TO_ID:
        labels["cms"] = CMS_TO_ID[_clean(row.get("cms_label")).upper()]
    return labels


@dataclass
class DatasetSummary:
    n_rows: int
    n_patients: int
    task_counts: Dict[str, Dict[str, int]]


class MultiTaskGraphCacheDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        splits: str | Sequence[str],
        clinical_dir: str | Path,
        include_datasets: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        deduplicate_patients: bool = True,
        allow_empty: bool = False,
    ):
        self.manifest_csv = Path(manifest_csv)
        if isinstance(splits, str):
            splits = [s.strip() for s in splits.split(",") if s.strip()]
        self.splits = set(splits)
        self.include_datasets = set(include_datasets) if include_datasets else None
        self.label_index = load_multitask_label_index(clinical_dir)

        with self.manifest_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        selected: List[Dict[str, str]] = []
        seen_patients: set[tuple[str, str]] = set()
        for row in rows:
            if row.get("split") not in self.splits:
                continue
            if self.include_datasets and row.get("dataset") not in self.include_datasets:
                continue
            if not Path(row.get("cache_path", "")).exists():
                continue
            patient_key = (row.get("dataset", ""), row.get("patient_id") or row.get("sample_id") or row.get("slide_id", ""))
            if deduplicate_patients and patient_key[1]:
                if patient_key in seen_patients:
                    continue
                seen_patients.add(patient_key)
            row = dict(row)
            row["_labels"] = labels_for_manifest_row(row, self.label_index)  # type: ignore[assignment]
            selected.append(row)
            if limit is not None and int(limit) > 0 and len(selected) >= int(limit):
                break
        if not selected and not allow_empty:
            raise RuntimeError(f"No cached graphs found for splits={sorted(self.splits)} in {manifest_csv}")
        self.rows = selected

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        data = torch.load(row["cache_path"], weights_only=False)
        data.slide_id = row["slide_id"]
        data.dataset = row["dataset"]
        data.patient_id = row.get("patient_id", "")
        labels = row["_labels"]  # type: ignore[index]
        self._attach_labels(data, labels)
        return data

    @staticmethod
    def _attach_binary_or_multiclass(data, task: str, value: Optional[int]) -> None:
        has_value = value is not None
        setattr(data, task, torch.tensor([int(value) if has_value else -1], dtype=torch.long))
        setattr(data, f"{task}_mask", torch.tensor([has_value], dtype=torch.bool))

    @staticmethod
    def _attach_survival(data, task: str, time: Optional[float], event: Optional[int]) -> None:
        has_value = time is not None and event is not None
        setattr(data, f"{task}_time", torch.tensor([float(time) if has_value else -1.0], dtype=torch.float32))
        setattr(data, f"{task}_event", torch.tensor([int(event) if has_value else -1], dtype=torch.long))
        setattr(data, f"{task}_mask", torch.tensor([has_value], dtype=torch.bool))

    def _attach_labels(self, data, labels: Dict[str, Optional[float | int]]) -> None:
        for task in CLASSIFICATION_TASKS:
            self._attach_binary_or_multiclass(data, task, labels.get(task))  # type: ignore[arg-type]
        self._attach_survival(data, "os", labels.get("os_time"), labels.get("os_event"))  # type: ignore[arg-type]
        self._attach_survival(data, "dfs", labels.get("dfs_time"), labels.get("dfs_event"))  # type: ignore[arg-type]

    @property
    def cms_labels(self) -> List[int]:
        vals = []
        for row in self.rows:
            value = row["_labels"].get("cms")  # type: ignore[index]
            if value is not None:
                vals.append(int(value))
        return vals

    def label_values(self, task: str) -> List[int]:
        vals = []
        for row in self.rows:
            value = row["_labels"].get(task)  # type: ignore[index]
            if value is not None:
                vals.append(int(value))
        return vals

    def summary(self) -> DatasetSummary:
        patients = {(r.get("dataset", ""), r.get("patient_id") or r.get("sample_id") or r.get("slide_id", "")) for r in self.rows}
        task_counts: Dict[str, Dict[str, int]] = {}
        for task in CLASSIFICATION_TASKS:
            task_counts[task] = {str(k): int(v) for k, v in Counter(self.label_values(task)).items()}
        for task in SURVIVAL_TASKS:
            events = []
            for row in self.rows:
                labels = row["_labels"]  # type: ignore[index]
                if labels.get(f"{task}_time") is not None and labels.get(f"{task}_event") is not None:
                    events.append(int(labels[f"{task}_event"]))  # type: ignore[arg-type]
            task_counts[task] = {f"event={k}": int(v) for k, v in Counter(events).items()}
        return DatasetSummary(n_rows=len(self.rows), n_patients=len(patients), task_counts=task_counts)

