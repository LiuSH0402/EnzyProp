from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import pandas as pd


MAX_SASA = {
    "ALA": 121, "ARG": 265, "ASN": 187, "ASP": 187, "CYS": 148,
    "GLN": 214, "GLU": 214, "GLY": 97, "HIS": 216, "ILE": 195,
    "LEU": 191, "LYS": 230, "MET": 203, "PHE": 228, "PRO": 154,
    "SER": 143, "THR": 163, "TRP": 264, "TYR": 255, "VAL": 165,
}
STD_AA = set(MAX_SASA)


def _load_freesasa():
    try:
        import freesasa
    except ImportError as exc:
        raise ImportError(
            "FreeSASA is required when EnzyProp needs to generate sasa_three files. "
            "Install it in the active environment or provide --sasa-dir with precomputed files."
        ) from exc
    return freesasa


def pdb_stem(path: str | Path) -> str:
    name = Path(path).name
    if name.lower().endswith(".pdb.gz"):
        return name[:-7]
    if name.lower().endswith(".pdb"):
        return name[:-4]
    return Path(name).stem


def find_pdb_by_id(pdb_dir: str | Path, pid: str) -> Path | None:
    root = Path(pdb_dir)
    if not root.exists():
        return None
    pid_lower = str(pid).strip().lower()
    hits = [
        p for p in root.iterdir()
        if p.is_file()
        and p.name.lower().startswith(pid_lower)
        and (p.name.lower().endswith(".pdb") or p.name.lower().endswith(".pdb.gz"))
    ]
    return sorted(hits, key=lambda p: len(p.name))[0] if hits else None


def expected_sasa_files(out_dir: str | Path, pid: str) -> dict[str, Path]:
    root = Path(out_dir)
    return {
        "detail": root / f"{pid}_sasa_detail_3class.csv",
        "internal": root / f"{pid}_internal_only.csv",
        "semi": root / f"{pid}_semi_only.csv",
        "exposed": root / f"{pid}_exposed_only.csv",
    }


def has_three_sasa_files(out_dir: str | Path, pid: str) -> bool:
    files = expected_sasa_files(out_dir, pid)
    return files["internal"].exists() and files["semi"].exists() and files["exposed"].exists()


def analyze_pdb_3class(
    pdb_path: str | Path,
    out_dir: str | Path,
    *,
    output_id: str | None = None,
    rsasa_internal: float = 5.0,
    rsasa_exposed: float = 20.0,
    abs_sasa_thresh: float | None = None,
    chains_to_include: Iterable[str] | None = None,
) -> dict[str, Path]:
    """Calculate residue SASA/rSASA and write internal/semi/exposed CSV files."""
    freesasa = _load_freesasa()
    pdb_path = Path(pdb_path)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    name = output_id or pdb_stem(pdb_path)
    chain_filter = set(chains_to_include) if chains_to_include is not None else None

    calc_path = pdb_path
    temp_dir = None
    try:
        str(pdb_path).encode("ascii")
    except UnicodeEncodeError:
        temp_dir = tempfile.TemporaryDirectory(prefix="enzyprop_sasa_")
        calc_path = Path(temp_dir.name) / pdb_path.name.encode("ascii", "ignore").decode("ascii")
        if not calc_path.name:
            calc_path = Path(temp_dir.name) / "input.pdb"
        shutil.copy2(pdb_path, calc_path)

    structure = freesasa.Structure(str(calc_path))
    result = freesasa.calc(structure)

    residue_sasa: dict[tuple[str, str, str], float] = {}
    for i in range(structure.nAtoms()):
        chain = structure.chainLabel(i)
        if chain_filter is not None and chain not in chain_filter:
            continue
        resn = structure.residueNumber(i)
        resname = structure.residueName(i)
        key = (chain, resn, resname)
        residue_sasa[key] = residue_sasa.get(key, 0.0) + float(result.atomArea(i))

    if temp_dir is not None:
        temp_dir.cleanup()

    rows = []
    for (chain, resn, resname), sasa in residue_sasa.items():
        if resname not in STD_AA:
            continue
        rsasa = sasa / float(MAX_SASA[resname]) * 100.0
        if abs_sasa_thresh is not None and sasa < float(abs_sasa_thresh):
            cls = "Internal"
        elif rsasa < rsasa_internal:
            cls = "Internal"
        elif rsasa < rsasa_exposed:
            cls = "Semi"
        else:
            cls = "Exposed"
        try:
            residue_number = int(resn)
        except ValueError:
            residue_number = int("".join(ch for ch in str(resn) if ch.isdigit()) or 0)
        rows.append((chain, residue_number, resname, float(sasa), float(rsasa), cls))

    df = pd.DataFrame(rows, columns=["Chain", "Residue_Number", "Residue_Name", "SASA", "Relative_SASA", "Class"])
    files = expected_sasa_files(out_root, name)
    df.to_csv(files["detail"], index=False, encoding="utf-8-sig")
    df[df["Class"] == "Internal"].copy().to_csv(files["internal"], index=False, encoding="utf-8-sig")
    df[df["Class"] == "Semi"].copy().to_csv(files["semi"], index=False, encoding="utf-8-sig")
    df[df["Class"] == "Exposed"].copy().to_csv(files["exposed"], index=False, encoding="utf-8-sig")
    return files


def ensure_sasa_three_for_ids(
    ids: Iterable[str],
    pdb_dir: str | Path,
    sasa_dir: str | Path,
    *,
    overwrite: bool = False,
    rsasa_internal: float = 5.0,
    rsasa_exposed: float = 20.0,
    abs_sasa_thresh: float | None = None,
) -> dict[str, int]:
    """Generate missing sasa_three CSV files for the given protein ids."""
    out_root = Path(sasa_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    made = 0
    reused = 0
    missing_pdb = 0
    failed = 0
    for pid in ids:
        pid = str(pid).strip()
        if not overwrite and has_three_sasa_files(out_root, pid):
            reused += 1
            continue
        pdb_path = find_pdb_by_id(pdb_dir, pid)
        if pdb_path is None:
            missing_pdb += 1
            continue
        try:
            analyze_pdb_3class(
                pdb_path,
                out_root,
                output_id=pid,
                rsasa_internal=rsasa_internal,
                rsasa_exposed=rsasa_exposed,
                abs_sasa_thresh=abs_sasa_thresh,
            )
            made += 1
        except Exception as exc:
            failed += 1
            print(f"[SASA][WARN] Failed {pid}: {exc}")
    print(f"[SASA] generated={made} reused={reused} missing_pdb={missing_pdb} failed={failed} -> {out_root}")
    return {"generated": made, "reused": reused, "missing_pdb": missing_pdb, "failed": failed}
