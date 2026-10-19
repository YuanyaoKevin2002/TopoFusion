# Example Data Format

This repository expects de-identified slide-level feature files. No real data are included.

## Manifest CSV

A manifest row should describe one slide:

| column | description |
|---|---|
| `slide_id` | anonymized slide key, e.g. `slide_000001` |
| `patient_id` | anonymized patient key, e.g. `patient_000001` |
| `cohort` | generic cohort name, e.g. `train`, `validation`, `external_a` |
| `feature_path` | path to patch feature H5 file |
| `label_cms` | optional integer label 0--3 |
| `label_msi` | optional binary label 0/1 |
| `label_pn` | optional binary label 0/1 |
| `label_cin` | optional binary label 0/1 |
| `os_time`, `os_event` | optional survival time and event indicator |
| `dfs_time`, `dfs_event` | optional disease-free survival time and event indicator |

## Patch Feature H5

Each de-identified H5 file should contain:

- `patch_feats`: float array of shape `(N, 1536)` for patch embeddings.
- `coords`: float array of shape `(N, 2)` for patch centers in a consistent coordinate system.
- `tissue_probs`: float array of shape `(N, 9)` or `(N, 4)`. The 4-class order used by TopoFusion is `[TUM, STR, LYM, MUC]`.
- `cell_counts`: optional float array of shape `(N, 5)`.
- `cell_topology`: optional float array of shape `(N, 7)`.

Missing labels should be represented as empty strings or NaN in the manifest and handled by task masks.
