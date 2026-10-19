from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from topofusion_v3.constants import CELL_COUNT_NAMES, CELL_TOPO_DIM, CELL_TYPE_TO_COUNT_COL

EPS = 1e-6


def _resolve_cells_json(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        candidates = [p / "cell_detection" / "cells.json", p / "cells.json"]
        for cand in candidates:
            if cand.exists():
                return cand
    return p


def _load_cells(cells_json: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    cells_path = _resolve_cells_json(cells_json)
    with cells_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    type_map_raw = obj.get("type_map", {})
    type_map = {}
    for name, idx in type_map_raw.items():
        try:
            type_map[int(idx)] = str(name)
        except Exception:
            continue
    cells = obj.get("cells", [])
    if not cells:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64), type_map
    centroids = np.asarray([c.get("centroid", [np.nan, np.nan]) for c in cells], dtype=np.float32)
    types = np.asarray([int(c.get("type", 0)) for c in cells], dtype=np.int64)
    valid = np.isfinite(centroids).all(axis=1)
    valid &= types >= 0
    return centroids[valid], types[valid], type_map


def _stratified_cap(indices: np.ndarray, types: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(indices) <= cap:
        return indices
    rng = np.random.default_rng(seed)
    keep = []
    remaining = []
    for t in sorted(set(types[indices].tolist())):
        idx_t = indices[types[indices] == t]
        take = min(10, len(idx_t))
        keep.extend(idx_t[:take].tolist())
        if len(idx_t) > take:
            remaining.extend(idx_t[take:].tolist())
    remaining = np.asarray(remaining, dtype=np.int64)
    quota = max(0, cap - len(keep))
    if quota > 0 and len(remaining) > 0:
        sampled = rng.choice(remaining, size=min(quota, len(remaining)), replace=False)
        keep.extend(sampled.tolist())
    return np.asarray(keep[:cap], dtype=np.int64)


def _topo_from_cells(pos: np.ndarray, types: np.ndarray, radius: float) -> np.ndarray:
    z = np.zeros((CELL_TOPO_DIM,), dtype=np.float32)
    n = int(len(types))
    if n <= 0:
        return z
    neo, inf, conn = 1, 2, 3
    if n >= 2:
        tree = cKDTree(pos)
        pairs = np.asarray(list(tree.query_pairs(r=radius)), dtype=np.int64)
    else:
        pairs = np.zeros((0, 2), dtype=np.int64)
    if pairs.size > 0:
        s = types[pairs[:, 0]]
        t = types[pairs[:, 1]]
        e = float(len(pairs))
        z[0] = float((((s == neo) & (t == inf)) | ((s == inf) & (t == neo))).sum()) / (e + EPS)
        z[1] = float((((s == neo) & (t == conn)) | ((s == conn) & (t == neo))).sum()) / (e + EPS)
        z[2] = float((s != t).sum()) / (e + EPS)

        neigh: List[List[int]] = [[] for _ in range(n)]
        for u, v in pairs.tolist():
            neigh[u].append(v)
            neigh[v].append(u)
        neo_idx = np.where(types == neo)[0]
        mix_inf, mix_conn, neo_nonneo = [], [], []
        for u in neo_idx.tolist():
            nu = neigh[u]
            if not nu:
                continue
            nt = types[np.asarray(nu, dtype=np.int64)]
            deg = float(len(nu))
            mix_inf.append(float((nt == inf).sum()) / (deg + EPS))
            mix_conn.append(float((nt == conn).sum()) / (deg + EPS))
            neo_nonneo.append(float((nt != neo).sum()) / (deg + EPS))
        z[3] = float(np.mean(mix_inf)) if mix_inf else 0.0
        z[4] = float(np.mean(mix_conn)) if mix_conn else 0.0
        z[6] = float(np.mean(neo_nonneo)) if neo_nonneo else 0.0

    idx_inf = np.where(types == inf)[0]
    idx_neo = np.where(types == neo)[0]
    if len(idx_inf) > 0 and len(idx_neo) > 0:
        d = np.sqrt(((pos[idx_inf, None, :] - pos[idx_neo][None, :, :]) ** 2).sum(axis=2))
        z[5] = float(np.median(d.min(axis=1))) if d.size else 0.0

    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def compute_cell_features_for_patches(
    cells_json: str | Path,
    patch_coords_mpp025: np.ndarray,
    patch_size: float = 256.0,
    buffer_px: float = 64.0,
    radius_px: float = 50.0,
    cap_ext: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Compute 5-D cell counts and 7-D topology per H5 patch.

    Cell centroids are already in the same native/Mpp0.25 coordinate scale as
    `coords_mpp025` for the current integrated H5 files.
    """
    coords = np.asarray(patch_coords_mpp025, dtype=np.float32)
    n_patches = int(coords.shape[0])
    counts = np.zeros((n_patches, len(CELL_COUNT_NAMES)), dtype=np.float32)
    topo = np.zeros((n_patches, CELL_TOPO_DIM), dtype=np.float32)
    centroids, cell_types, type_map = _load_cells(cells_json)
    meta: Dict[str, object] = {
        "n_cells": int(len(cell_types)),
        "cell_type_map": {str(k): v for k, v in type_map.items()},
        "n_patches": n_patches,
        "empty_patch_ratio": 1.0,
    }
    if n_patches == 0 or len(cell_types) == 0:
        return counts, topo, meta

    centers = coords + float(patch_size) / 2.0
    patch_tree = cKDTree(centers)
    d, patch_idx = patch_tree.query(centroids, k=1)
    assign_radius = (2.0 ** 0.5) * float(patch_size) / 2.0 + 2.0
    assigned = d <= assign_radius
    for t, col in CELL_TYPE_TO_COUNT_COL.items():
        mask = assigned & (cell_types == int(t))
        if mask.any():
            np.add.at(counts[:, col], patch_idx[mask], 1.0)

    cell_tree = cKDTree(centroids)
    query_radius = (2.0 ** 0.5) * (float(patch_size) / 2.0 + float(buffer_px)) + 2.0
    empty = 0
    for i, (x0, y0) in enumerate(coords):
        cand = np.asarray(cell_tree.query_ball_point([x0 + patch_size / 2.0, y0 + patch_size / 2.0], r=query_radius), dtype=np.int64)
        if cand.size == 0:
            empty += 1
            continue
        cpos = centroids[cand]
        in_box = (
            (cpos[:, 0] >= x0 - buffer_px)
            & (cpos[:, 0] < x0 + patch_size + buffer_px)
            & (cpos[:, 1] >= y0 - buffer_px)
            & (cpos[:, 1] < y0 + patch_size + buffer_px)
        )
        cand = cand[in_box]
        cand = cand[cell_types[cand] != 0]
        if cand.size == 0:
            empty += 1
            continue
        cand = _stratified_cap(cand, cell_types, cap=int(cap_ext), seed=int(seed) + i)
        topo[i] = _topo_from_cells(centroids[cand], cell_types[cand], radius=float(radius_px))

    meta["empty_patch_ratio"] = float(empty) / float(max(1, n_patches))
    return counts, topo, meta


def load_or_compute_cell_features_for_patches(
    cells_json: str | Path,
    patch_coords_mpp025: np.ndarray,
    cache_path: Optional[str | Path] = None,
    use_cache: bool = True,
    write_cache: bool = True,
    overwrite: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    cp = Path(cache_path) if cache_path else None
    n_patches = int(np.asarray(patch_coords_mpp025).shape[0])
    if use_cache and cp is not None and cp.exists() and not overwrite:
        try:
            obj = np.load(cp, allow_pickle=False)
            counts = obj["cell_counts"].astype(np.float32)
            topo = obj["cell_topo"].astype(np.float32)
            if counts.shape[0] == n_patches and topo.shape[0] == n_patches:
                meta = {
                    "source": "cell_feature_cache",
                    "cache_path": str(cp),
                    "n_patches": n_patches,
                }
                return counts, topo, meta
        except Exception:
            pass

    counts, topo, meta = compute_cell_features_for_patches(cells_json, patch_coords_mpp025, **kwargs)
    if write_cache and cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        # Float16 is enough for counts/topology cache and keeps EXTERNAL_COHORT_B storage low.
        np.savez_compressed(
            cp,
            cell_counts=counts.astype(np.float16),
            cell_topo=topo.astype(np.float16),
            n_patches=np.asarray([n_patches], dtype=np.int64),
        )
        meta["cache_path"] = str(cp)
        meta["source"] = "computed_and_cached"
    return counts.astype(np.float32), topo.astype(np.float32), meta
