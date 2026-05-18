from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import PredictorConfig, default_hf_cache_dir, default_models_dir, get_config, normalize_target, resolve_checkpoint
from .data import EnzymeDataset, build_dataset_from_table, collate_fn, normalize_items_from_table
from .model import ZScoreTarget, build_model, normalize_state_dict_keys
from .sasa import ensure_sasa_three_for_ids


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        requested = device.lower()
        if requested == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpointed_model(
    config: PredictorConfig,
    checkpoint_path: str | Path,
    device: torch.device,
    strict_load: bool = True,
) -> tuple[torch.nn.Module, ZScoreTarget, dict]:
    model = build_model(config, device=device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model", checkpoint)
    model_state = normalize_state_dict_keys(model_state)
    missing, unexpected = model.load_state_dict(model_state, strict=strict_load)
    if not strict_load and (missing or unexpected):
        print(f"[WARN] Non-strict load: missing={len(missing)} unexpected={len(unexpected)}")
    scaler_state = checkpoint.get("scaler")
    if scaler_state is None:
        raise KeyError("Checkpoint does not contain a 'scaler' entry; cannot inverse-transform predictions.")
    scaler = ZScoreTarget()
    scaler.load_state_dict(scaler_state)
    model.eval()
    return model, scaler, checkpoint


def collect_predictions(
    model: torch.nn.Module,
    scaler: ZScoreTarget,
    loader: DataLoader,
    device: torch.device,
    z: float,
    uncertainty_low_sd: float,
    uncertainty_high_sd: float,
) -> list[dict]:
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            seq_feats = batch["seq_feats"].to(device)
            batch_graph = batch.get("batch_graph")
            if batch_graph is not None:
                batch_graph = batch_graph.to(device)
            out = model(
                seqs=batch["seqs"],
                masks=batch["masks"],
                seq_feats=seq_feats,
                batch_graph=batch_graph,
            )
            mu_z = out["mu_z"].float()
            log_var_z = out["log_var_z"].float().clamp(-6, 2)
            mu = scaler.inverse(mu_z)
            if scaler.std is None:
                raise RuntimeError("Scaler std is not loaded.")
            sd = (0.5 * log_var_z).exp() * scaler.std.to(device)

            ids = batch["ids"]
            for i, pid in enumerate(ids):
                pred_mu = float(mu[i].cpu())
                pred_sd = float(sd[i].cpu())
                if pred_sd < uncertainty_low_sd:
                    uncertainty = "low"
                elif pred_sd <= uncertainty_high_sd:
                    uncertainty = "medium"
                else:
                    uncertainty = "high"
                rows.append({
                    "id": pid,
                    "pred_mu": pred_mu,
                    "pred_sd": pred_sd,
                    "pred_var": pred_sd * pred_sd,
                    "lower95": pred_mu - z * pred_sd,
                    "upper95": pred_mu + z * pred_sd,
                    "uncertainty": uncertainty,
                })
    return rows


def result_csv_paths(out_root: str | Path, config: PredictorConfig) -> tuple[Path, Path | None]:
    out_root = Path(out_root)
    result_path = out_root / config.result_filename
    legacy_result_path = None
    if config.legacy_result_filename and config.legacy_result_filename != config.result_filename:
        legacy_result_path = out_root / config.legacy_result_filename
    return result_path, legacy_result_path


def resolve_existing_result_csv(out_root: str | Path, config: PredictorConfig) -> Path:
    result_path, legacy_result_path = result_csv_paths(out_root, config)
    if result_path.exists():
        return result_path
    if legacy_result_path is not None and legacy_result_path.exists():
        return legacy_result_path
    return result_path


def predict_target(
    target: str,
    input_csv: str | Path,
    pdb_file: str | Path | None = None,
    pdb_dir: str | Path | None = None,
    sasa_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    out_dir: str | Path = "outputs",
    model_path: str | Path | None = None,
    device_name: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    strict_files: bool | None = None,
    strict_load: bool = True,
    hf_cache: str | Path | None = None,
    auto_sasa: bool = True,
    overwrite_sasa: bool = False,
) -> dict:
    input_csv = Path(input_csv)
    temp_dirs: list[tempfile.TemporaryDirectory] = []
    table_items_for_paths = None

    if pdb_file is not None:
        table_items_for_paths = normalize_items_from_table(input_csv, get_config(target))
        if len(table_items_for_paths) != 1:
            raise ValueError("--pdb-file can only be used when --input-csv contains exactly one sequence row.")
        temp_pdb_dir = tempfile.TemporaryDirectory(prefix="enzyprop_pdb_")
        temp_dirs.append(temp_pdb_dir)
        source_pdb = Path(pdb_file)
        if not source_pdb.exists():
            raise FileNotFoundError(f"PDB file not found: {source_pdb}")
        suffix = ".pdb.gz" if source_pdb.name.lower().endswith(".pdb.gz") else ".pdb"
        copied_pdb = Path(temp_pdb_dir.name) / f"{table_items_for_paths[0]['id']}{suffix}"
        shutil.copy2(source_pdb, copied_pdb)
        pdb_dir = Path(temp_pdb_dir.name)

    if pdb_dir is None:
        pdb_dir = input_csv.parent / "PDB"
    if sasa_dir is None:
        temp_sasa_dir = tempfile.TemporaryDirectory(prefix="enzyprop_sasa_three_")
        temp_dirs.append(temp_sasa_dir)
        sasa_dir = Path(temp_sasa_dir.name)
    if models_dir is None:
        models_dir = default_models_dir()
    if hf_cache is None:
        hf_cache = default_hf_cache_dir()

    overrides = {}
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if num_workers is not None:
        overrides["num_workers"] = num_workers
    if strict_files is not None:
        overrides["strict_files"] = strict_files
    overrides["esm2_cache"] = str(hf_cache)
    overrides["protbert_cache"] = str(hf_cache)
    config = get_config(target, **overrides)
    checkpoint_path = resolve_checkpoint(config, models_dir=models_dir, model_path=model_path)
    device = resolve_device(device_name)
    print(f"[Device] {device}")
    print(f"[Checkpoint] {checkpoint_path}")
    print(f"[HF cache] {hf_cache}")
    print(f"[PDB dir] {pdb_dir}")
    print(f"[SASA dir] {sasa_dir}")

    try:
        if auto_sasa:
            table_items = table_items_for_paths or normalize_items_from_table(input_csv, config)
            ensure_sasa_three_for_ids(
                [item["id"] for item in table_items],
                pdb_dir,
                sasa_dir,
                overwrite=overwrite_sasa,
            )

        items = build_dataset_from_table(input_csv, pdb_dir, sasa_dir, config)
        if not items:
            raise RuntimeError("No usable samples were built from the input table and file folders.")
        dataset = EnzymeDataset(items, config)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(device.type == "cuda" and config.pin_memory),
            drop_last=config.drop_last,
            collate_fn=collate_fn,
        )

        model, scaler, checkpoint = load_checkpointed_model(config, checkpoint_path, device, strict_load=strict_load)
        rows = collect_predictions(
            model,
            scaler,
            loader,
            device,
            z=config.interval_z,
            uncertainty_low_sd=config.uncertainty_low_sd,
            uncertainty_high_sd=config.uncertainty_high_sd,
        )
        result_df = pd.DataFrame(rows, columns=["id", "pred_mu", "pred_var", "lower95", "upper95", "uncertainty"])

        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        result_path, _ = result_csv_paths(out_root, config)
        result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
        print(f"[Saved] {len(result_df)} -> {result_path}")
        return {"target": config.target, "n": int(len(result_df)), "csv": str(result_path)}
    finally:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

def predict_all_targets(
    targets: Iterable[str],
    input_csv: str | Path,
    pdb_file: str | Path | None = None,
    pdb_dir: str | Path | None = None,
    sasa_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    out_dir: str | Path = "outputs",
    device_name: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    strict_files: bool | None = None,
    strict_load: bool = True,
    hf_cache: str | Path | None = None,
    auto_sasa: bool = True,
    overwrite_sasa: bool = False,
) -> dict:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    reports = {}
    merged: pd.DataFrame | None = None
    for target in targets:
        normalized_target = normalize_target(target)
        report = predict_target(
            target=target,
            input_csv=input_csv,
            pdb_file=pdb_file,
            pdb_dir=pdb_dir,
            sasa_dir=sasa_dir,
            models_dir=models_dir,
            out_dir=out_root,
            device_name=device_name,
            batch_size=batch_size,
            num_workers=num_workers,
            strict_files=strict_files,
            strict_load=strict_load,
            hf_cache=hf_cache,
            auto_sasa=auto_sasa,
            overwrite_sasa=overwrite_sasa,
        )
        reports[target] = report
        config = get_config(normalized_target)
        df = pd.read_csv(resolve_existing_result_csv(out_root, config))
        if "pred_var" not in df.columns and "pred_sd" in df.columns:
            df["pred_var"] = df["pred_sd"] * df["pred_sd"]
        keep = df[["id", "pred_mu", "pred_var", "lower95", "upper95", "uncertainty"]].copy()
        keep = keep.rename(columns={
            "pred_mu": f"{normalized_target}_pred",
            "pred_var": f"{normalized_target}_variance",
            "lower95": f"{normalized_target}_lower95",
            "upper95": f"{normalized_target}_upper95",
            "uncertainty": f"{normalized_target}_uncertainty",
        })
        merged = keep if merged is None else merged.merge(keep, on="id", how="outer")
    merged_path = out_root / "all_targets_predictions.csv"
    if merged is not None:
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        print(f"[Saved] merged predictions -> {merged_path}")
    return {"reports": reports, "merged_csv": str(merged_path)}
