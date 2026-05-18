from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Batch, Data
except ImportError as exc:
    raise ImportError(
        "torch-geometric is required for the checkpointed GNN model. "
        "Install a CPU or CUDA build that matches your PyTorch version."
    ) from exc

from .config import PredictorConfig
from .features import clean_seq, seq_feats_tensor
from .pdb_utils import parse_pdb_coords_raw


def norm_id(pid: str) -> str:
    return str(pid).strip().split()[0]


class FileIndexer:
    """Index PDB and SASA files by leading protein id."""

    def __init__(self, folder: str | Path, kind: str):
        self.folder = Path(folder)
        self.kind = kind
        self.map: dict[str, list[Path]] = {}
        if not self.folder.exists():
            return

        if kind == "pdb":
            suffix = re.compile(r"(?:\.pdb(?:\.gz)?)$", re.I)
            stripers = [r"\.gz$", r"\.pdb$"]
        elif kind == "exposed":
            suffix = re.compile(r"_exposed_only\.csv(?:\.gz)?$", re.I)
            stripers = [r"\.gz$", r"_exposed_only\.csv$"]
        elif kind == "semi":
            suffix = re.compile(r"_semi_only\.csv(?:\.gz)?$", re.I)
            stripers = [r"\.gz$", r"_semi_only\.csv$"]
        elif kind == "internal":
            suffix = re.compile(r"_internal_only\.csv(?:\.gz)?$", re.I)
            stripers = [r"\.gz$", r"_internal_only\.csv$"]
        else:
            raise ValueError(f"Unsupported index kind: {kind}")

        for path in self.folder.iterdir():
            if not path.is_file() or not suffix.search(path.name):
                continue
            stem = path.name
            for pat in stripers:
                stem = re.sub(pat, "", stem, flags=re.I)
            lead = re.match(r"^([A-Za-z0-9]+)", stem)
            if lead:
                self.map.setdefault(lead.group(1), []).append(path)

    def get(self, pid: str) -> Path | None:
        hits = self.map.get(norm_id(pid))
        if not hits:
            return None
        return sorted(hits, key=lambda p: len(p.name))[0]


def find_best_by_id(folder: str | Path, pid: str, suffix: str) -> Path | None:
    root = Path(folder)
    if not root.exists():
        return None
    pid_norm = norm_id(pid).lower()
    suffix_norm = suffix.lower()
    hits = [
        p for p in root.iterdir()
        if p.is_file() and p.name.lower().startswith(pid_norm) and p.name.lower().endswith(suffix_norm)
    ]
    if not hits and suffix_norm == ".pdb":
        hits = [
            p for p in root.iterdir()
            if p.is_file() and p.name.lower().startswith(pid_norm) and p.name.lower().endswith(".pdb.gz")
        ]
    return sorted(hits, key=lambda p: len(p.name))[0] if hits else None


def parse_surface_positions_from_csv(path: str | Path) -> tuple[list[int], pd.DataFrame]:
    df = pd.read_csv(path)
    lower = {str(c).lower(): c for c in df.columns}
    col = None
    for candidate in ["residue_number", "residue", "position", "pos", "resid", "residue_index"]:
        if candidate in lower:
            col = lower[candidate]
            break
    if col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return [], df
        col = numeric_cols[0]
    pos = []
    for value in df[col].tolist():
        try:
            if pd.notna(value):
                pos.append(int(value))
        except (TypeError, ValueError):
            continue
    return pos, df


def positions_to_mask(positions: list[int], seq_len: int) -> tuple[list[int], int]:
    mask = [0] * max(0, int(seq_len))
    used = 0
    for p in positions:
        idx = int(p) - 1
        if 0 <= idx < len(mask) and mask[idx] == 0:
            mask[idx] = 1
            used += 1
    return mask, used


def read_table_any(table_path: str | Path) -> pd.DataFrame:
    path = Path(table_path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.attrs["_lower_map"] = {str(c).lower(): c for c in df.columns}
    return df


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_items_from_table(table_path: str | Path, config: PredictorConfig) -> list[dict]:
    df = read_table_any(table_path)
    low = df.attrs.get("_lower_map", {str(c).lower(): c for c in df.columns})

    def get_c(keys: list[str]) -> str | None:
        for key in keys:
            if key.lower() in low:
                return low[key.lower()]
        return None

    id_col = get_c(["uniprotkb_ac", "id", "entry", "protein_id", "accession"])
    seq_col = get_c(["sequence", "seq", "aa_seq"])
    pi_col = get_c(["pi", "target", "label"])
    ph_col = get_c(["phopt", "ph", "ph_value", "optimal_ph", "optimum_ph"])
    temp_col = get_c(["topt", "temp", "tm", "temperature", "optimal_temp", "optimal_temperature", "optimum_temp"])

    if not id_col or not seq_col:
        raise ValueError("Input table must contain id/accession and sequence columns.")

    items: list[dict] = []
    for _, row in df.iterrows():
        pid = norm_id(str(row[id_col]))
        seq = clean_seq(str(row[seq_col]).strip() if pd.notna(row[seq_col]) else "")
        if not seq:
            continue
        if len(seq) > config.max_seq_len:
            seq = seq[: config.max_seq_len]
        items.append({
            "id": pid,
            "sequence": seq,
            "seq_len": len(seq),
            "pi": _as_float(row.get(pi_col)) if pi_col else None,
            "phopt": _as_float(row.get(ph_col)) if ph_col else None,
            "topt": _as_float(row.get(temp_col)) if temp_col else None,
        })
    return items


def build_dataset_from_table(
    table_path: str | Path,
    pdb_dir: str | Path,
    sasa_dir: str | Path,
    config: PredictorConfig,
) -> list[dict]:
    rows = normalize_items_from_table(table_path, config)
    pdb_idx = FileIndexer(pdb_dir, "pdb")
    ex_idx = FileIndexer(sasa_dir, "exposed")
    se_idx = FileIndexer(sasa_dir, "semi")
    in_idx = FileIndexer(sasa_dir, "internal")

    dataset: list[dict] = []
    misses = 0
    for item in rows:
        pid = item["id"]
        length = item["seq_len"]
        pdb_p = pdb_idx.get(pid) or find_best_by_id(pdb_dir, pid, ".pdb")
        ex_p = ex_idx.get(pid) or find_best_by_id(sasa_dir, pid, "_exposed_only.csv")
        se_p = se_idx.get(pid) or find_best_by_id(sasa_dir, pid, "_semi_only.csv")
        in_p = in_idx.get(pid) or find_best_by_id(sasa_dir, pid, "_internal_only.csv")

        if config.strict_files and (not pdb_p or not ex_p or not se_p or not in_p):
            misses += 1
            continue

        def mask_from_csv(path: Optional[Path]) -> list[int]:
            if path is None:
                return [0] * length
            positions, _ = parse_surface_positions_from_csv(path)
            mask, _ = positions_to_mask(positions, length)
            return mask

        item["pdb_path"] = str(pdb_p) if pdb_p else None
        item["exposed_csv"] = str(ex_p) if ex_p else None
        item["semi_csv"] = str(se_p) if se_p else None
        item["internal_csv"] = str(in_p) if in_p else None
        item["exposed_mask"] = mask_from_csv(ex_p)
        item["semi_mask"] = mask_from_csv(se_p)
        item["internal_mask"] = mask_from_csv(in_p)
        item["surface_mask"] = item["exposed_mask"]
        dataset.append(item)

    print(f"[{Path(table_path).name}] Loaded {len(dataset)} items. (Skipped {misses} due to missing files)")
    return dataset


def make_graph_from_coords(coords: torch.Tensor, seq_len: int, k: int = 20) -> Data:
    if coords.shape[0] > seq_len:
        coords = coords[:seq_len]
    n = int(coords.shape[0])
    if n <= 1:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 17), dtype=torch.float32)
        return Data(pos=coords, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)

    kk = min(max(1, int(k)), n - 1)
    dist_matrix = torch.cdist(coords, coords)
    dist_matrix.fill_diagonal_(math.inf)
    dst = torch.topk(dist_matrix, k=kk, largest=False).indices.reshape(-1)
    src = torch.arange(n, dtype=torch.long).repeat_interleave(kk)
    edge_index = torch.stack([src, dst], dim=0)

    dist = (coords[src] - coords[dst]).norm(dim=1)
    centers = torch.linspace(0, 30, 16, device=coords.device)
    step = centers[1] - centers[0]
    edge_attr_dist = torch.exp(-((dist.unsqueeze(-1) - centers) / step) ** 2)
    seq_dist = (src - dst).float().unsqueeze(-1) / 100.0
    edge_attr = torch.cat([edge_attr_dist, seq_dist], dim=1).float()
    return Data(pos=coords, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


class EnzymeDataset(Dataset):
    def __init__(self, items: list[dict], config: PredictorConfig):
        self.items = items
        self.config = config

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        seq = item["sequence"]
        y_value = item.get(self.config.target)
        y = float(y_value) if y_value is not None else float("nan")
        graph_data = None

        if self.config.use_gnn and item.get("pdb_path"):
            coords = parse_pdb_coords_raw(item["pdb_path"])
            if coords is not None:
                graph_data = make_graph_from_coords(coords, len(seq), k=self.config.gnn_k_neighbors)

        if self.config.use_gnn and graph_data is None:
            length = len(seq)
            dummy = torch.zeros(length, 3)
            if length:
                dummy[:, 0] = torch.arange(length)
            graph_data = make_graph_from_coords(dummy, length, k=min(5, max(1, length - 1)))
            graph_data.is_dummy = torch.tensor([True])
        elif self.config.use_gnn:
            graph_data.is_dummy = torch.tensor([False])

        return {
            "id": item["id"],
            "seq": seq,
            "y": y,
            "masks_dict": {
                "exposed": item.get("exposed_mask", item.get("surface_mask", [0] * len(seq))),
                "semi": item.get("semi_mask", [0] * len(seq)),
                "internal": item.get("internal_mask", [0] * len(seq)),
            },
            "seq_feat": seq_feats_tensor(seq, expected_dim=self.config.d_seqfeat),
            "graph_data": graph_data,
        }


def collate_fn(batch: list[dict]) -> dict:
    ids = [b["id"] for b in batch]
    seqs = [b["seq"] for b in batch]
    ys = torch.tensor([b["y"] for b in batch], dtype=torch.float32)
    masks = {
        "exposed": [b["masks_dict"]["exposed"] for b in batch],
        "semi": [b["masks_dict"]["semi"] for b in batch],
        "internal": [b["masks_dict"]["internal"] for b in batch],
    }
    seq_feats = torch.stack([b["seq_feat"] for b in batch])
    graph_items = [b["graph_data"] for b in batch if b["graph_data"] is not None]
    batch_graph = Batch.from_data_list(graph_items) if len(graph_items) == len(batch) else None
    return {
        "ids": ids,
        "seqs": seqs,
        "y": ys,
        "masks": masks,
        "seq_feats": seq_feats,
        "batch_graph": batch_graph,
    }
