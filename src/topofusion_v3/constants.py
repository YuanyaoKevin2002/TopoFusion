from __future__ import annotations

CMS_TO_ID = {"CMS1": 0, "CMS2": 1, "CMS3": 2, "CMS4": 3}
ID_TO_CMS = {v: k for k, v in CMS_TO_ID.items()}

TISSUE_CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TISSUE_NAME_TO_IDX = {name: idx for idx, name in enumerate(TISSUE_CLASS_NAMES)}
TISSUE4_NAMES = ["TUM", "STR", "LYM", "MUC"]
TISSUE4_IDXS = [TISSUE_NAME_TO_IDX[name] for name in TISSUE4_NAMES]
TUM4, STR4, LYM4, MUC4 = 0, 1, 2, 3

CELL_COUNT_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
CELL_TYPE_TO_COUNT_COL = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
CELL_TOPO_DIM = 7

