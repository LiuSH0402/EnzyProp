from __future__ import annotations

import gzip
from pathlib import Path

import torch


def parse_pdb_coords_raw(pdb_path: str | Path) -> torch.Tensor | None:
    """Parse CA atom coordinates from a PDB or PDB.GZ file."""
    coords: list[list[float]] = []
    path = str(pdb_path)
    try:
        open_fn = gzip.open if path.endswith(".gz") else open
        mode = "rt" if path.endswith(".gz") else "r"
        with open_fn(path, mode) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                try:
                    coords.append([
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ])
                except ValueError:
                    continue
    except OSError:
        return None
    if not coords:
        return None
    return torch.tensor(coords, dtype=torch.float32)
