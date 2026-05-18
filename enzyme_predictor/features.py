from __future__ import annotations

import math
import statistics as stats
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch


AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA20)
AA_ALL = AA20 + ["X"]

KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    "X": 0.0,
}

AA_MASS = {
    "A": 71.0788, "C": 103.1388, "D": 115.0886, "E": 129.1155, "F": 147.1766,
    "G": 57.0519, "H": 137.1411, "I": 113.1594, "K": 128.1741, "L": 113.1594,
    "M": 131.1926, "N": 114.1038, "P": 97.1167, "Q": 128.1307, "R": 156.1875,
    "S": 87.0782, "T": 101.1051, "V": 99.1326, "W": 186.2132, "Y": 163.1760,
    "X": 0.0,
}

GROUPS = {
    "acidic": set("DE"),
    "basic": set("KRH"),
    "charged": set("DEKRH"),
    "polar_uncharged": set("STNQCY"),
    "nonpolar": set("GAVLIMPFW"),
    "hydrophobic": set("AVILMFYW"),
    "aromatic": set("FYW"),
    "aliphatic": set("ILV"),
    "tiny": set("ACGST"),
    "small": set("ACDGNPSTV"),
    "proline": set("P"),
    "glycine": set("G"),
}

GROUP_ORDER = [
    "acidic", "basic", "charged", "polar_uncharged", "nonpolar",
    "hydrophobic", "aromatic", "aliphatic", "tiny", "small", "proline", "glycine",
]


def clean_seq(seq: str) -> str:
    seq = "".join(ch for ch in str(seq).upper() if not ch.isspace())
    return "".join(ch if ch in AA_SET else "X" for ch in seq)


def length_stats(seq: str) -> dict[str, float]:
    s = clean_seq(seq)
    length = len(s)
    return {
        "length": float(length),
        "sqrt_len": math.sqrt(max(length, 1)),
        "log_len": math.log(max(length, 1.0)),
    }


def aac(seq: str) -> dict[str, float]:
    s = clean_seq(seq)
    length = max(len(s), 1)
    counts = {aa: 0 for aa in AA_ALL}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    return {f"aac_{aa}": counts[aa] / length for aa in AA_ALL}


def group_fractions(seq: str) -> dict[str, float]:
    s = clean_seq(seq)
    length = max(len(s), 1)
    return {f"frac_{name}": sum(1 for ch in s if ch in group) / length for name, group in GROUPS.items()}


def kd_stats(seq: str) -> dict[str, float]:
    s = clean_seq(seq)
    vals = [KD[ch] for ch in s if ch in KD and ch != "X"] or [0.0]
    return {
        "kd_mean": float(sum(vals) / len(vals)),
        "kd_std": float(stats.pstdev(vals)),
        "kd_min": float(min(vals)),
        "kd_max": float(max(vals)),
    }


def molecular_weight(seq: str) -> dict[str, float]:
    s = clean_seq(seq)
    if not s:
        return {"mw_residue_sum": 0.0, "mw_peptide_est": 0.0}
    residue_sum = sum(AA_MASS.get(ch, 0.0) for ch in s)
    peptide_est = residue_sum - max(len(s) - 1, 0) * 18.01528
    return {"mw_residue_sum": float(residue_sum), "mw_peptide_est": float(peptide_est)}


def ctd_hydrophobicity(seq: str) -> dict[str, float]:
    s = clean_seq(seq)
    length = max(len(s), 1)
    hydrophob = sum(1 for ch in s if KD.get(ch, 0.0) > 0)
    hydrophil = sum(1 for ch in s if KD.get(ch, 0.0) < 0)
    neutral = length - hydrophob - hydrophil
    return {
        "ctd_hydrophob_frac": hydrophob / length,
        "ctd_hydrophil_frac": hydrophil / length,
        "ctd_hydroneut_frac": neutral / length,
    }


def extract_traditional_features(seq: str) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(length_stats(seq))
    feats.update(aac(seq))
    feats.update(group_fractions(seq))
    feats.update(kd_stats(seq))
    feats.update(molecular_weight(seq))
    feats.update(ctd_hydrophobicity(seq))
    return feats


def seq_feats_tensor(seq: str, expected_dim: int = 45) -> torch.Tensor:
    feats = extract_traditional_features(seq)
    vec: list[float] = []
    vec += [feats["length"], feats["sqrt_len"], feats["log_len"]]
    vec += [feats[f"aac_{aa}"] for aa in AA_ALL]
    vec += [feats[f"frac_{g}"] for g in GROUP_ORDER]
    vec += [feats["kd_mean"], feats["kd_std"], feats["kd_min"], feats["kd_max"]]
    vec += [feats["mw_residue_sum"], feats["mw_peptide_est"]]
    vec += [feats["ctd_hydrophob_frac"], feats["ctd_hydrophil_frac"], feats["ctd_hydroneut_frac"]]
    x = torch.tensor(vec, dtype=torch.float32)
    if x.numel() != expected_dim:
        raise ValueError(f"Sequence feature dim mismatch: got {x.numel()}, expected {expected_dim}")
    return x


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    ids: list[str] = []
    seqs: list[str] = []
    cur_id: str | None = None
    cur: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(cur))
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)
    if cur_id is not None:
        ids.append(cur_id)
        seqs.append("".join(cur))
    return list(zip(ids, seqs))


def featurize_sequences(pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for pid, seq in pairs:
        rows.append({"id": pid, "sequence": clean_seq(seq), **extract_traditional_features(seq)})
    return pd.DataFrame(rows)
