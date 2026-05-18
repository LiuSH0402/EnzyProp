from __future__ import annotations

import argparse


def add_common_predict_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-csv", required=True, help="CSV/XLSX table with id/accession and sequence columns.")
    parser.add_argument("--pdb-file", default=None, help="Single PDB/PDB.GZ file. Use when --input-csv has exactly one sequence row.")
    parser.add_argument("--pdb-dir", default=None, help="Folder containing PDB or PDB.GZ files. Default: PDB next to --input-csv.")
    parser.add_argument("--sasa-dir", default=None, help="Folder containing or receiving SASA CSV files. Default: temporary folder deleted after prediction.")
    parser.add_argument("--models-dir", default=None, help="Folder containing trained checkpoints. Default: project models folder.")
    parser.add_argument("--out-dir", default="outputs", help="Folder for prediction CSV outputs.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto-detect.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from the target config.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count. Default config uses 0 for portability.")
    parser.add_argument("--hf-cache", default=None, help="Optional HuggingFace cache folder for ESM2 and ProtBert.")
    parser.add_argument("--skip-sasa", action="store_true", help="Use existing sasa_three files and do not calculate missing SASA files.")
    parser.add_argument("--overwrite-sasa", action="store_true", help="Recalculate SASA files even when existing files are present.")
    parser.add_argument("--allow-missing-files", action="store_true", help="Do not skip rows when PDB/SASA files are missing; missing parts become dummy inputs.")
    parser.add_argument("--non-strict-load", action="store_true", help="Load checkpoint with strict=False. Use only for debugging architecture mismatches.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="EnzyProp",
        description="Predict enzyme optimal pH, pI, or optimal temperature from sequence, PDB, and SASA files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="Predict one target: phopt, pi, or topt.")
    predict.add_argument("--target", required=True, choices=["phopt", "pi", "topt", "ph", "temp"])
    predict.add_argument("--model-path", default=None, help="Optional explicit checkpoint path.")
    add_common_predict_args(predict)

    predict_all = sub.add_parser("predict-all", help="Predict phopt, pi, and topt sequentially.")
    predict_all.add_argument("--targets", nargs="+", default=["phopt", "pi", "topt"], choices=["phopt", "pi", "topt", "ph", "temp"])
    add_common_predict_args(predict_all)

    sasa = sub.add_parser("sasa", help="Generate sasa_three files from a sequence table and PDB folder.")
    sasa.add_argument("--input-csv", required=True, help="CSV/XLSX table with id/accession and sequence columns.")
    sasa.add_argument("--pdb-file", default=None, help="Single PDB/PDB.GZ file. Use when --input-csv has exactly one sequence row.")
    sasa.add_argument("--pdb-dir", default=None, help="Folder containing PDB or PDB.GZ files. Default: PDB next to --input-csv.")
    sasa.add_argument("--sasa-dir", default=None, help="Output folder for sasa_three CSV files. Default: sasa_three next to --input-csv.")
    sasa.add_argument("--overwrite", action="store_true", help="Recalculate files even when outputs already exist.")

    train = sub.add_parser("train", help="Train an EnzyProp model from a YAML/JSON config file.")
    train.add_argument("--config", required=True, help="Training config YAML/JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        from .predict import predict_target

        strict_files = not args.allow_missing_files
        strict_load = not args.non_strict_load
        predict_target(
            target=args.target,
            input_csv=args.input_csv,
            pdb_file=args.pdb_file,
            pdb_dir=args.pdb_dir,
            sasa_dir=args.sasa_dir,
            models_dir=args.models_dir,
            out_dir=args.out_dir,
            model_path=args.model_path,
            device_name=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            strict_files=strict_files,
            strict_load=strict_load,
            hf_cache=args.hf_cache,
            auto_sasa=not args.skip_sasa,
            overwrite_sasa=args.overwrite_sasa,
        )
        return 0

    if args.command == "predict-all":
        from .predict import predict_all_targets

        strict_files = not args.allow_missing_files
        strict_load = not args.non_strict_load
        predict_all_targets(
            targets=args.targets,
            input_csv=args.input_csv,
            pdb_file=args.pdb_file,
            pdb_dir=args.pdb_dir,
            sasa_dir=args.sasa_dir,
            models_dir=args.models_dir,
            out_dir=args.out_dir,
            device_name=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            strict_files=strict_files,
            strict_load=strict_load,
            hf_cache=args.hf_cache,
            auto_sasa=not args.skip_sasa,
            overwrite_sasa=args.overwrite_sasa,
        )
        return 0

    if args.command == "sasa":
        import shutil
        import tempfile
        from pathlib import Path

        from .config import get_config
        from .data import normalize_items_from_table
        from .sasa import ensure_sasa_three_for_ids

        input_csv = Path(args.input_csv)
        pdb_dir = Path(args.pdb_dir) if args.pdb_dir else input_csv.parent / "PDB"
        sasa_dir = Path(args.sasa_dir) if args.sasa_dir else input_csv.parent / "sasa_three"
        items = normalize_items_from_table(input_csv, get_config("topt"))
        temp_dir = None
        if args.pdb_file:
            if len(items) != 1:
                raise ValueError("--pdb-file can only be used when --input-csv contains exactly one sequence row.")
            source_pdb = Path(args.pdb_file)
            if not source_pdb.exists():
                raise FileNotFoundError(f"PDB file not found: {source_pdb}")
            temp_dir = tempfile.TemporaryDirectory(prefix="enzyprop_pdb_")
            suffix = ".pdb.gz" if source_pdb.name.lower().endswith(".pdb.gz") else ".pdb"
            copied_pdb = Path(temp_dir.name) / f"{items[0]['id']}{suffix}"
            shutil.copy2(source_pdb, copied_pdb)
            pdb_dir = Path(temp_dir.name)
        try:
            ensure_sasa_three_for_ids([item["id"] for item in items], pdb_dir, sasa_dir, overwrite=args.overwrite)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
        return 0

    if args.command == "train":
        from .train import train_from_config

        train_from_config(args.config)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
