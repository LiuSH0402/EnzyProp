from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal


TargetName = Literal["phopt", "pi", "topt"]


@dataclass(frozen=True)
class PredictorConfig:
    target: TargetName
    checkpoint_name: str
    result_filename: str
    legacy_result_filename: str | None
    uncertainty_low_sd: float
    uncertainty_high_sd: float

    max_seq_len: int = 1800
    d_seqfeat: int = 45
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    strict_files: bool = True

    gnn_k_neighbors: int = 20
    use_gnn: bool = True
    gnn_hidden: int = 128
    gnn_layers: int = 3
    d_struct_graph: int = 128

    use_protbert: bool = True
    protbert_model: str = "Rostlab/prot_bert"
    protbert_cache: str | None = None
    protbert_hidden: int = 1024
    protbert_finetune: bool = True
    protbert_pool: str = "mean"

    use_esm2: bool = True
    esm2_model: str = "facebook/esm2_t33_650M_UR50D"
    esm2_cache: str | None = None
    esm2_hidden: int = 1280
    esm2_finetune: bool = True

    fusion_hidden: int = 1024
    dropout: float = 0.15
    interval_z: float = 1.96


TARGET_CONFIGS: dict[str, PredictorConfig] = {
    "topt": PredictorConfig(
        target="topt",
        checkpoint_name="topt_best_model.pt",
        result_filename="topt_result.csv",
        legacy_result_filename="topt_relust.csv",
        uncertainty_low_sd=4.338927222,
        uncertainty_high_sd=5.140176075,
    ),
    "phopt": PredictorConfig(
        target="phopt",
        checkpoint_name="phopt_best_model.pt",
        result_filename="phopt_result.csv",
        legacy_result_filename="phopt_relust.csv",
        uncertainty_low_sd=0.359165809,
        uncertainty_high_sd=0.428786247,
    ),
    "pi": PredictorConfig(
        target="pi",
        checkpoint_name="pi_best_model.pt",
        result_filename="pi_result.csv",
        legacy_result_filename="pi_relust.csv",
        uncertainty_low_sd=0.539304525,
        uncertainty_high_sd=0.628773868,
    ),
}

TARGET_ALIASES = {
    "temp": "topt",
    "temperature": "topt",
    "ph": "phopt",
}


def normalize_target(target: str) -> str:
    key = target.lower()
    return TARGET_ALIASES.get(key, key)


def get_config(target: str, **overrides) -> PredictorConfig:
    key = normalize_target(target)
    if key not in TARGET_CONFIGS:
        valid = ", ".join(sorted(TARGET_CONFIGS))
        raise ValueError(f"Unknown target '{target}'. Expected one of: {valid}")
    cfg = TARGET_CONFIGS[key]
    if overrides:
        cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    return cfg


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_models_dir() -> Path:
    return project_root() / "models"


def default_hf_cache_dir() -> Path:
    return project_root() / "hf_cache"


def resolve_checkpoint(config: PredictorConfig, models_dir: str | Path, model_path: str | Path | None = None) -> Path:
    if model_path:
        path = Path(model_path)
    else:
        root = Path(models_dir) if models_dir else default_models_dir()
        roots = [root]
        if not root.is_absolute():
            roots.append(project_root() / root)
        candidates = []
        for candidate_root in roots:
            candidates.extend([
                candidate_root / config.checkpoint_name,
                candidate_root / config.target / config.checkpoint_name,
                candidate_root / f"{config.target}_best_on_extval.pt",
                candidate_root / f"{config.target}_full_best_on_extval.pt",
                candidate_root / "temp_full_best_on_extval.pt" if config.target == "topt" else candidate_root / "__never__",
                candidate_root / "ph_full_best_on_extval.pt" if config.target == "phopt" else candidate_root / "__never__",
                candidate_root / "pi_best_on_extval.pt" if config.target == "pi" else candidate_root / "__never__",
            ])
        path = next((p for p in candidates if p.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Put {config.checkpoint_name} in the models folder "
            "or pass --model-path explicitly."
        )
    return path
