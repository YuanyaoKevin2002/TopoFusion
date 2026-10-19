from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from topofusion_v3.cell_features import load_or_compute_cell_features_for_patches
from topofusion_v3.constants import LYM4, MUC4, STR4, TISSUE4_IDXS, TUM4


def _edge_attr_for_pair(
    coords: np.ndarray,
    tissue4: np.ndarray,
    i: int,
    j: int,
    max_edge_dist: float,
    dist_norm_cap: float,
) -> List[float]:
    dx = float(coords[j, 0] - coords[i, 0])
    dy = float(coords[j, 1] - coords[i, 1])
    dist = float(np.hypot(dx, dy))
    pi = tissue4[i]
    pj = tissue4[j]
    p_tl = float(pi[TUM4] * pj[LYM4] + pj[TUM4] * pi[LYM4])
    p_ts = float(pi[TUM4] * pj[STR4] + pj[TUM4] * pi[STR4])
    p_sl = float(pi[STR4] * pj[LYM4] + pj[STR4] * pi[LYM4])
    dist_norm = min(dist / max(float(max_edge_dist), 1e-6), float(dist_norm_cap))
    return [dx, dy, dist_norm, p_tl, p_ts, p_sl]


def _connected_components(n: int, edge_index: np.ndarray) -> Tuple[np.ndarray, int]:
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

    if edge_index.size:
        src = edge_index[0]
        dst = edge_index[1]
        for a, b in zip(src, dst):
            union(int(a), int(b))

    roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    _, comp_ids = np.unique(roots, return_inverse=True)
    return comp_ids.astype(np.int64), int(comp_ids.max() + 1) if n else 0


def _build_knn_edges(
    coords: np.ndarray,
    tissue4: np.ndarray,
    k: int,
    max_edge_dist: float,
    dist_norm_cap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(coords.shape[0])
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 6), dtype=np.float32)
    k_eff = min(int(k), n - 1)
    tree = cKDTree(coords)
    dists, nbrs = tree.query(coords, k=k_eff + 1)
    edge_src: List[int] = []
    edge_dst: List[int] = []
    attrs: List[List[float]] = []
    seen = set()
    for i in range(n):
        for jj in range(1, k_eff + 1):
            j = int(nbrs[i, jj])
            dist = float(dists[i, jj])
            if j < 0 or j >= n or dist >= float(max_edge_dist):
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            edge_src.append(i)
            edge_dst.append(j)
            attrs.append(_edge_attr_for_pair(coords, tissue4, i, j, max_edge_dist, dist_norm_cap))
    if not edge_src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 6), dtype=np.float32)
    return (
        np.vstack([np.asarray(edge_src, dtype=np.int64), np.asarray(edge_dst, dtype=np.int64)]),
        np.asarray(attrs, dtype=np.float32),
    )


def _stable_slide_seed(base_seed: int, slide_id: str) -> int:
    digest = hashlib.md5(str(slide_id).encode("utf-8")).hexdigest()[:8]
    return (int(base_seed) + int(digest, 16)) % (2**32 - 1)


def _purity_guided_downsample_mask(
    tissue4: np.ndarray,
    mask: np.ndarray,
    max_per_class: int,
    purity_threshold: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, int | float]]:
    max_per_class = int(max_per_class)
    if max_per_class <= 0:
        n = int(mask.sum())
        return mask, {
            "downsample_enabled": 0,
            "downsample_nodes_before": n,
            "downsample_nodes_after": n,
            "downsample_nodes_dropped": 0,
        }

    filtered_idx = np.flatnonzero(mask)
    n_before = int(filtered_idx.size)
    if n_before == 0:
        return mask, {
            "downsample_enabled": 1,
            "downsample_nodes_before": 0,
            "downsample_nodes_after": 0,
            "downsample_nodes_dropped": 0,
        }

    tissue_f = tissue4[filtered_idx]
    max_prob = tissue_f.max(axis=1)
    dominant = tissue_f.argmax(axis=1)
    keep_f = np.zeros(n_before, dtype=bool)
    rng = np.random.default_rng(int(seed))
    class_before = []
    class_after = []
    class_low = []

    for cls in range(4):
        cls_local = np.flatnonzero(dominant == cls)
        class_before.append(int(cls_local.size))
        if cls_local.size <= max_per_class:
            keep_f[cls_local] = True
            class_low.append(int((max_prob[cls_local] <= float(purity_threshold)).sum()))
            class_after.append(int(cls_local.size))
            continue
        low_local = cls_local[max_prob[cls_local] <= float(purity_threshold)]
        high_local = cls_local[max_prob[cls_local] > float(purity_threshold)]
        if low_local.size >= max_per_class:
            # Hard cap: low-purity/interface nodes remain highest priority, but
            # extremely large interface regions must still respect the per-class
            # node budget to keep slide graphs comparable.
            selected = rng.choice(low_local, size=max_per_class, replace=False)
            keep_f[selected] = True
        else:
            keep_f[low_local] = True
            need = max_per_class - int(low_local.size)
            if need > 0 and high_local.size > 0:
                selected = rng.choice(high_local, size=min(need, int(high_local.size)), replace=False)
                keep_f[selected] = True
        class_low.append(int(low_local.size))
        class_after.append(int(keep_f[cls_local].sum()))

    out = np.zeros_like(mask, dtype=bool)
    out[filtered_idx[keep_f]] = True
    n_after = int(out.sum())
    return out, {
        "downsample_enabled": 1,
        "downsample_max_per_class": int(max_per_class),
        "downsample_purity_threshold": float(purity_threshold),
        "downsample_nodes_before": n_before,
        "downsample_nodes_after": n_after,
        "downsample_nodes_dropped": int(n_before - n_after),
        "downsample_class0_before": class_before[0],
        "downsample_class1_before": class_before[1],
        "downsample_class2_before": class_before[2],
        "downsample_class3_before": class_before[3],
        "downsample_class0_after": class_after[0],
        "downsample_class1_after": class_after[1],
        "downsample_class2_after": class_after[2],
        "downsample_class3_after": class_after[3],
        "downsample_class0_low_purity": class_low[0],
        "downsample_class1_low_purity": class_low[1],
        "downsample_class2_low_purity": class_low[2],
        "downsample_class3_low_purity": class_low[3],
    }


def _stitch_components(
    coords: np.ndarray,
    tissue4: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    max_edge_dist: float,
    target_max_components: int,
    bridge_max_dist: float,
    bridge_neighbor_k: int,
    bidirectional: bool,
    dist_norm_cap: float,
    force_target_components: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int | float]]:
    n = int(coords.shape[0])
    comp_ids, n_comp = _connected_components(n, edge_index)
    initial = int(n_comp)
    if n <= 1 or target_max_components <= 0 or n_comp <= target_max_components:
        return edge_index, edge_attr, {
            "component_count_initial": initial,
            "component_count_final": initial,
            "component_bridge_edges_added": 0,
            "component_bridge_pairs_added": 0,
            "component_bridge_forced_pairs_added": 0,
        }

    query_k = min(max(int(bridge_neighbor_k), 2), n)
    tree = cKDTree(coords)
    dists, nbrs = tree.query(coords, k=query_k)
    if query_k == 1:
        dists = dists[:, None]
        nbrs = nbrs[:, None]

    candidates: List[Tuple[float, int, int, int, int]] = []
    best_by_pair: Dict[Tuple[int, int], Tuple[float, int, int, int, int]] = {}
    for i in range(n):
        ci = int(comp_ids[i])
        for jj in range(1, query_k):
            j = int(nbrs[i, jj])
            if j < 0 or j >= n:
                continue
            cj = int(comp_ids[j])
            if ci == cj:
                continue
            dist = float(dists[i, jj])
            if dist > float(bridge_max_dist):
                continue
            key = (ci, cj) if ci < cj else (cj, ci)
            item = (dist, ci, cj, i, j)
            if key not in best_by_pair or dist < best_by_pair[key][0]:
                best_by_pair[key] = item
    candidates = sorted(best_by_pair.values(), key=lambda x: x[0])

    parent = np.arange(initial, dtype=np.int64)
    rank = np.zeros(initial, dtype=np.int8)

    def find_comp(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union_comp(a: int, b: int) -> bool:
        ra = find_comp(int(a))
        rb = find_comp(int(b))
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    src_new: List[int] = []
    dst_new: List[int] = []
    attr_new: List[List[float]] = []
    current_comp = initial
    pairs_added = 0
    forced_pairs_added = 0
    for _, ci, cj, i, j in candidates:
        if current_comp <= int(target_max_components):
            break
        if not union_comp(ci, cj):
            continue
        src_new.append(i)
        dst_new.append(j)
        attr_new.append(_edge_attr_for_pair(coords, tissue4, i, j, max_edge_dist, dist_norm_cap))
        if bidirectional:
            src_new.append(j)
            dst_new.append(i)
            attr_new.append(_edge_attr_for_pair(coords, tissue4, j, i, max_edge_dist, dist_norm_cap))
        current_comp -= 1
        pairs_added += 1

    # Some WSIs have tissue islands farther apart than the biological bridge
    # distance. For graph-batch stability we still force a small number of
    # long-range component bridges, with the long distance explicitly encoded.
    while force_target_components and current_comp > int(target_max_components):
        root_to_nodes: Dict[int, List[int]] = {}
        for node_idx, comp_id in enumerate(comp_ids.tolist()):
            root = find_comp(int(comp_id))
            if root not in root_to_nodes:
                root_to_nodes[root] = []
            root_to_nodes[root].append(node_idx)
        if len(root_to_nodes) <= int(target_max_components):
            break
        roots = sorted(root_to_nodes)
        reps: List[int] = []
        centroids: List[np.ndarray] = []
        for root in roots:
            nodes = np.asarray(root_to_nodes[root], dtype=np.int64)
            c = coords[nodes].mean(axis=0)
            rep = int(nodes[np.argmin(((coords[nodes] - c) ** 2).sum(axis=1))])
            reps.append(rep)
            centroids.append(c)
        centroids_np = np.asarray(centroids, dtype=np.float32)
        tree = cKDTree(centroids_np)
        query_k = min(len(roots), max(2, min(32, len(roots))))
        best = None
        for a, root_a in enumerate(roots):
            dists, nbrs = tree.query(centroids_np[a], k=query_k)
            dists = np.atleast_1d(dists)
            nbrs = np.atleast_1d(nbrs)
            for dist, b in zip(dists.tolist(), nbrs.tolist()):
                b = int(b)
                if b == a:
                    continue
                root_b = roots[b]
                if find_comp(root_a) == find_comp(root_b):
                    continue
                item = (float(dist), int(root_a), int(root_b), reps[a], reps[b])
                if best is None or item[0] < best[0]:
                    best = item
                break
        if best is None:
            break
        _, root_a, root_b, i, j = best
        if not union_comp(root_a, root_b):
            break
        src_new.append(i)
        dst_new.append(j)
        attr_new.append(_edge_attr_for_pair(coords, tissue4, i, j, max_edge_dist, dist_norm_cap))
        if bidirectional:
            src_new.append(j)
            dst_new.append(i)
            attr_new.append(_edge_attr_for_pair(coords, tissue4, j, i, max_edge_dist, dist_norm_cap))
        current_comp -= 1
        forced_pairs_added += 1

    if src_new:
        bridge_index = np.vstack([np.asarray(src_new, dtype=np.int64), np.asarray(dst_new, dtype=np.int64)])
        bridge_attr = np.asarray(attr_new, dtype=np.float32)
        edge_index = np.concatenate([edge_index, bridge_index], axis=1)
        edge_attr = np.concatenate([edge_attr, bridge_attr], axis=0)

    _, final_comp = _connected_components(n, edge_index)
    return edge_index, edge_attr, {
        "component_count_initial": initial,
        "component_count_final": int(final_comp),
        "component_bridge_edges_added": int(len(src_new)),
        "component_bridge_pairs_added": int(pairs_added),
        "component_bridge_forced_pairs_added": int(forced_pairs_added),
    }


def build_graph_from_h5_and_cellvit(
    h5_path: str,
    cellvit_path: str,
    label_id: int,
    dataset: str,
    slide_id: str,
    patient_id: str,
    cfg: Dict[str, object],
) -> Data:
    with h5py.File(h5_path, "r") as f:
        patch_feats = f["features"][:].astype(np.float32)
        coords025 = f["coords_mpp025"][:].astype(np.float32)
        tissue_prob = f["tissue_prob"][:].astype(np.float32)
        target_mpp = str(f.attrs.get("target_mpp", ""))
        tissue_names = str(f.attrs.get("tissue_class_names", ""))

    tissue4 = tissue_prob[:, TISSUE4_IDXS].astype(np.float32)
    p_core = tissue4.sum(axis=1)
    thr = float(cfg.get("tissue_core_threshold", 0.5))
    lym_thr = float(cfg.get("lym_rescue_threshold", -1.0))
    muc_thr = float(cfg.get("muc_rescue_threshold", -1.0))
    core_mask = p_core > thr
    lym_mask = tissue4[:, LYM4] > lym_thr if lym_thr >= 0.0 else np.zeros_like(core_mask)
    muc_mask = tissue4[:, MUC4] > muc_thr if muc_thr >= 0.0 else np.zeros_like(core_mask)
    mask = core_mask | lym_mask | muc_mask
    rescued_lym = int(np.logical_and(~core_mask, lym_mask).sum())
    rescued_muc = int(np.logical_and(~core_mask, muc_mask).sum())

    min_nodes = int(cfg.get("min_nodes_after_filter", 8))
    if int(mask.sum()) < min_nodes:
        keep = np.argsort(-p_core)[: min(min_nodes, len(p_core))]
        mask = np.zeros_like(p_core, dtype=bool)
        mask[keep] = True

    downsample_stats: Dict[str, int | float] = {
        "downsample_enabled": 0,
        "downsample_nodes_before": int(mask.sum()),
        "downsample_nodes_after": int(mask.sum()),
        "downsample_nodes_dropped": 0,
    }
    max_per_class = int(cfg.get("purity_downsample_max_per_class", 0))
    if max_per_class > 0:
        mask, downsample_stats = _purity_guided_downsample_mask(
            tissue4,
            mask,
            max_per_class=max_per_class,
            purity_threshold=float(cfg.get("purity_downsample_threshold", 0.8)),
            seed=_stable_slide_seed(int(cfg.get("seed", 42)), slide_id),
        )

    cell_cache_root = str(cfg.get("cell_feature_cache_root", "") or "").strip()
    cell_cache_datasets = {
        x.strip().upper()
        for x in str(cfg.get("cell_feature_cache_datasets", "EXTERNAL_COHORT_B")).split(",")
        if x.strip()
    }
    cell_cache_path = None
    use_dataset_cell_cache = str(dataset).upper() in cell_cache_datasets
    if cell_cache_root and use_dataset_cell_cache:
        cell_cache_path = Path(cell_cache_root) / str(dataset) / f"{slide_id}.npz"

    cell_counts_all, cell_topo_all, cell_meta = load_or_compute_cell_features_for_patches(
        cellvit_path,
        coords025,
        cache_path=cell_cache_path,
        use_cache=bool(cfg.get("use_cell_feature_cache", True)) and use_dataset_cell_cache,
        write_cache=bool(cfg.get("write_cell_feature_cache", True)) and use_dataset_cell_cache,
        patch_size=float(cfg.get("patch_size_mpp025", 256.0)),
        buffer_px=float(cfg.get("cell_buffer_px", 64.0)),
        radius_px=float(cfg.get("cell_radius_px", 50.0)),
        cap_ext=int(cfg.get("cell_cap_ext", 1000)),
        seed=int(cfg.get("seed", 42)),
    )

    patch_feats = patch_feats[mask]
    coords025_f = coords025[mask]
    tissue4_f = tissue4[mask]
    cell_counts = cell_counts_all[mask]
    cell_topo = cell_topo_all[mask]
    coords_model = coords025_f * float(cfg.get("coord_scale_to_model", 0.5))

    max_edge_dist = float(cfg.get("max_edge_dist", 300.0))
    dist_norm_cap = float(cfg.get("component_bridge_dist_norm_cap", 5.0))
    edge_index_np, edge_attr_np = _build_knn_edges(
        coords_model,
        tissue4_f,
        k=int(cfg.get("knn_k", 8)),
        max_edge_dist=max_edge_dist,
        dist_norm_cap=dist_norm_cap,
    )
    edge_index_np, edge_attr_np, comp_stats = _stitch_components(
        coords_model,
        tissue4_f,
        edge_index_np,
        edge_attr_np,
        max_edge_dist=max_edge_dist,
        target_max_components=int(cfg.get("target_max_components", 5)),
        bridge_max_dist=float(cfg.get("component_bridge_max_dist", 1500.0)),
        bridge_neighbor_k=int(cfg.get("component_bridge_neighbor_k", 64)),
        bidirectional=bool(cfg.get("component_bridge_bidirectional", True)),
        dist_norm_cap=dist_norm_cap,
        force_target_components=bool(cfg.get("component_bridge_force_target", True)),
    )

    dominant = tissue4_f.argmax(axis=1) if tissue4_f.size else np.zeros((0,), dtype=np.int64)
    denom = max(int(tissue4_f.shape[0]), 1)
    tissue_dom_frac = np.asarray([(dominant == i).sum() / denom for i in range(4)], dtype=np.float32)

    data = Data()
    data.patch_feats = torch.from_numpy(patch_feats)
    data.coords = torch.from_numpy(coords_model.astype(np.float32))
    data.coords_mpp025 = torch.from_numpy(coords025_f.astype(np.float32))
    data.tissue_probs4 = torch.from_numpy(tissue4_f.astype(np.float32))
    data.cell_counts = torch.from_numpy(cell_counts.astype(np.float32))
    data.cell_topo = torch.from_numpy(cell_topo.astype(np.float32))
    data.edge_index = torch.from_numpy(edge_index_np).long()
    data.edge_attr_raw = torch.from_numpy(edge_attr_np.astype(np.float32))
    data.y = torch.tensor([int(label_id)], dtype=torch.long)
    data.num_nodes = int(patch_feats.shape[0])
    data.slide_id = str(slide_id)
    data.dataset = str(dataset)
    data.patient_id = str(patient_id)
    data.target_mpp = str(target_mpp)
    data.tissue_class_names = str(tissue_names)
    data.n_nodes_raw = torch.tensor([int(len(mask))], dtype=torch.long)
    data.n_nodes_kept = torch.tensor([int(mask.sum())], dtype=torch.long)
    data.n_nodes_core = torch.tensor([int(core_mask.sum())], dtype=torch.long)
    data.n_nodes_lym_rescued = torch.tensor([rescued_lym], dtype=torch.long)
    data.n_nodes_muc_rescued = torch.tensor([rescued_muc], dtype=torch.long)
    data.downsample_enabled = torch.tensor([int(downsample_stats.get("downsample_enabled", 0))], dtype=torch.long)
    data.downsample_nodes_before = torch.tensor([int(downsample_stats.get("downsample_nodes_before", int(mask.sum())))], dtype=torch.long)
    data.downsample_nodes_after = torch.tensor([int(downsample_stats.get("downsample_nodes_after", int(mask.sum())))], dtype=torch.long)
    data.downsample_nodes_dropped = torch.tensor([int(downsample_stats.get("downsample_nodes_dropped", 0))], dtype=torch.long)
    data.downsample_max_per_class = torch.tensor([int(downsample_stats.get("downsample_max_per_class", 0))], dtype=torch.long)
    data.downsample_purity_threshold = torch.tensor([float(downsample_stats.get("downsample_purity_threshold", 0.0))], dtype=torch.float32)
    data.component_count_initial = torch.tensor([int(comp_stats["component_count_initial"])], dtype=torch.long)
    data.component_count_final = torch.tensor([int(comp_stats["component_count_final"])], dtype=torch.long)
    data.component_bridge_edges_added = torch.tensor([int(comp_stats["component_bridge_edges_added"])], dtype=torch.long)
    data.component_bridge_pairs_added = torch.tensor([int(comp_stats["component_bridge_pairs_added"])], dtype=torch.long)
    data.component_bridge_forced_pairs_added = torch.tensor([int(comp_stats["component_bridge_forced_pairs_added"])], dtype=torch.long)
    data.tissue_dom_frac = torch.from_numpy(tissue_dom_frac)
    data.cell_empty_patch_ratio = torch.tensor([float(cell_meta.get("empty_patch_ratio", 0.0))], dtype=torch.float32)
    return data
