from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import PredictorConfig, default_hf_cache_dir, get_config, normalize_target
from .data import EnzymeDataset, build_dataset_from_table, collate_fn, normalize_items_from_table
from .model import ZScoreTarget, build_model
from .predict import collect_predictions, resolve_device
from .sasa import ensure_sasa_three_for_ids


def load_train_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    return data


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_from_tensors(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    y_true = y_true.view(-1).float()
    y_pred = y_pred.view(-1).float()
    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = math.sqrt(max(mse, 0.0))
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    ybar = torch.mean(y_true)
    ss_tot = torch.sum((y_true - ybar) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = float((1.0 - ss_res / (ss_tot + 1e-12)).item())
    if y_true.numel() >= 2:
        cov = torch.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
        var_t = torch.var(y_true, unbiased=False)
        var_p = torch.var(y_pred, unbiased=False)
        pearson = float((cov / (torch.sqrt(var_t * var_p) + 1e-12)).item())
        ccc = float((2 * cov / (var_t + var_p + (y_true.mean() - y_pred.mean()) ** 2 + 1e-12)).item())
    else:
        pearson = float("nan")
        ccc = float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson, "ccc": ccc}


def gaussian_nll(mu_z: torch.Tensor, log_var_z: torch.Tensor, y_z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (log_var_z + (y_z - mu_z) ** 2 * torch.exp(-log_var_z))


def append_log_csv(path: str | Path, row: dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_backbone_trainable(model: torch.nn.Module, trainable: bool):
    bb_kw = ("protbert", "pb_enc", "esm", "esm2", "backbone", "encoder", "transformer", "gnn")
    for name, param in model.named_parameters():
        if any(kw in name.lower() for kw in bb_kw):
            param.requires_grad = trainable


def set_grad_checkpointing(model: torch.nn.Module, enabled: bool):
    for attr in ("esm2_enc", "pb_enc"):
        module = getattr(model, attr, None)
        if module is None:
            continue
        if hasattr(module, "config") and hasattr(module.config, "use_cache"):
            module.config.use_cache = False
        if enabled and hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
        if (not enabled) and hasattr(module, "gradient_checkpointing_disable"):
            module.gradient_checkpointing_disable()


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    *,
    updates_per_epoch: int,
    epochs: int,
    lr_head: float,
    lr_backbone: float | None,
    weight_decay: float,
    warmup_ratio: float,
    lr_min_factor: float,
):
    bb_kw = ("protbert", "pb_enc", "esm", "esm2", "backbone", "encoder", "transformer", "gnn")

    def is_backbone(name: str) -> bool:
        return any(kw in name.lower() for kw in bb_kw)

    def is_no_decay(name: str) -> bool:
        ln = name.lower()
        return ln.endswith(".bias") or "layernorm.weight" in ln or "layer_norm.weight" in ln or ".ln" in ln

    groups: dict[str, dict[str, Any]] = {}

    def add_param(key: str, param, lr: float, wd: float):
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": wd})
        groups[key]["params"].append(param)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = lr_backbone if (lr_backbone is not None and is_backbone(name)) else lr_head
        key = ("bb" if is_backbone(name) else "head") + ("_no_decay" if is_no_decay(name) else "_decay")
        add_param(key, param, float(lr), 0.0 if is_no_decay(name) else float(weight_decay))

    if not groups:
        raise RuntimeError("No trainable parameters found.")
    optimizer = torch.optim.AdamW(list(groups.values()))
    total_steps = max(1, int(epochs) * max(1, int(updates_per_epoch)))
    warmup_steps = int(round(total_steps * max(0.0, float(warmup_ratio))))

    def lr_lambda(step: int):
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        denom = max(1, total_steps - warmup_steps - 1)
        t = min(max((step - warmup_steps) / denom, 0.0), 1.0)
        return lr_min_factor + (1 - lr_min_factor) * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    scaler: ZScoreTarget,
    *,
    is_train: bool,
    optimizer=None,
    scheduler=None,
    amp_scaler=None,
    accum_steps: int = 1,
    loss_alpha: float = 0.2,
    grad_clip: float | None = None,
    target_noise: float = 0.0,
    y_std: float | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    device = next(model.parameters()).device
    alpha = min(max(float(loss_alpha), 0.0), 1.0)
    beta = 1.0 - alpha
    model.train(is_train)
    if is_train:
        for sub_name in ["esm2_enc", "pb_enc", "gnn"]:
            sub = getattr(model, sub_name, None)
            if sub is not None and not any(p.requires_grad for p in sub.parameters()):
                sub.eval()

    ys, preds, sds = [], [], []
    loss_sum = nll_sum = huber_sum = 0.0
    n_total = 0
    accum_steps = max(1, int(accum_steps))
    context = torch.enable_grad if is_train else torch.no_grad

    with context():
        for bi, batch in enumerate(loader):
            y = batch["y"].to(device).float().view(-1)
            if y.numel() == 0:
                continue
            if not torch.isfinite(y).all():
                continue
            n_total += int(y.numel())
            y_obj = y
            if is_train and target_noise and y_std:
                y_obj = y + float(target_noise) * float(y_std) * torch.randn_like(y)
            y_z_obj = scaler.transform(y_obj)
            x_seqfeat = batch["seq_feats"].to(device)
            batch_graph = batch.get("batch_graph")
            if batch_graph is not None:
                batch_graph = batch_graph.to(device)
            if is_train and optimizer and bi % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(seqs=batch["seqs"], masks=batch["masks"], seq_feats=x_seqfeat, batch_graph=batch_graph)
                mu_z = out["mu_z"].view(-1).float()
                log_var_z = out["log_var_z"].view(-1).float().clamp(-10, 5)
                mu_orig = scaler.inverse(mu_z)
                sd_orig = (0.5 * log_var_z).exp() * scaler.std.to(device)
                huber = F.smooth_l1_loss(mu_orig, y, reduction="mean")
                nll = gaussian_nll(mu_z, log_var_z, y_z_obj).mean()
                loss = alpha * nll + beta * huber

            if is_train and optimizer:
                if amp_scaler is not None and use_amp:
                    amp_scaler.scale(loss / accum_steps).backward()
                    do_step = ((bi + 1) % accum_steps == 0) or (bi + 1 == len(loader))
                    if do_step:
                        amp_scaler.unscale_(optimizer)
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                        amp_scaler.step(optimizer)
                        amp_scaler.update()
                        if scheduler:
                            scheduler.step()
                else:
                    (loss / accum_steps).backward()
                    do_step = ((bi + 1) % accum_steps == 0) or (bi + 1 == len(loader))
                    if do_step:
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

            bsz = int(y.numel())
            loss_sum += float(loss.detach().cpu()) * bsz
            nll_sum += float(nll.detach().cpu()) * bsz
            huber_sum += float(huber.detach().cpu()) * bsz
            ys.append(y.detach().cpu())
            preds.append(mu_orig.detach().cpu())
            sds.append(sd_orig.detach().cpu())

    if n_total == 0:
        return {"loss": float("nan"), "n": 0}
    ycat = torch.cat(ys)
    pcat = torch.cat(preds)
    sdcat = torch.cat(sds)
    metrics = metrics_from_tensors(ycat, pcat)
    metrics.update({
        "loss": loss_sum / n_total,
        "nll_z": nll_sum / n_total,
        "huber": huber_sum / n_total,
        "mean_sd": float(sdcat.mean().item()),
        "n": n_total,
    })
    return metrics


def filter_labeled(items: list[dict], target: str) -> list[dict]:
    out = []
    for item in items:
        value = item.get(target)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            out.append(item)
    return out


def make_loader(items: list[dict], config: PredictorConfig, *, shuffle: bool) -> DataLoader:
    ds = EnzymeDataset(items, config)
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=collate_fn,
    )


def prepare_sasa_for_tables(tables: list[str | Path], pdb_dir: str | Path, sasa_dir: str | Path, config: PredictorConfig):
    ids = []
    for table in tables:
        if table:
            ids.extend(item["id"] for item in normalize_items_from_table(table, config))
    ensure_sasa_three_for_ids(ids, pdb_dir, sasa_dir, overwrite=False)


def write_predictions_csv(model, loader, scaler: ZScoreTarget, config: PredictorConfig, device, out_path: str | Path):
    rows = collect_predictions(
        model,
        scaler,
        loader,
        device,
        z=config.interval_z,
        uncertainty_low_sd=config.uncertainty_low_sd,
        uncertainty_high_sd=config.uncertainty_high_sd,
    )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    raw = load_train_config(config_path)
    config_root = config_path.resolve().parent

    def resolve_path(value: str | Path | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        return str(path if path.is_absolute() else config_root / path)

    target = normalize_target(str(raw["target"]))
    out_dir = Path(resolve_path(raw.get("out_dir", Path("runs") / target)))
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_cache = resolve_path(raw.get("hf_cache")) or str(default_hf_cache_dir())
    model_overrides = {
        "batch_size": raw.get("batch_size"),
        "num_workers": raw.get("num_workers", 0),
        "pin_memory": bool(raw.get("pin_memory", False)),
        "drop_last": bool(raw.get("drop_last", False)),
        "max_seq_len": raw.get("max_seq_len"),
        "dropout": raw.get("dropout"),
        "fusion_hidden": raw.get("fusion_hidden"),
        "use_esm2": raw.get("use_esm2"),
        "use_protbert": raw.get("use_protbert"),
        "use_gnn": raw.get("use_gnn"),
        "esm2_model": raw.get("esm2_model"),
        "protbert_model": raw.get("protbert_model"),
        "esm2_cache": hf_cache,
        "protbert_cache": hf_cache,
        "esm2_finetune": raw.get("esm2_finetune"),
        "protbert_finetune": raw.get("protbert_finetune"),
        "protbert_pool": raw.get("protbert_pool"),
        "gnn_hidden": raw.get("gnn_hidden"),
        "gnn_layers": raw.get("gnn_layers"),
        "gnn_k_neighbors": raw.get("gnn_k_neighbors"),
        "strict_files": True,
    }
    pred_config = get_config(target, **model_overrides)
    seed = int(raw.get("seed", 42))
    set_seed(seed)
    device = resolve_device(raw.get("device"))
    use_amp = bool(raw.get("use_amp", True)) and device.type == "cuda"
    print(f"[Train] target={target} device={device} amp={use_amp}")

    train_table = resolve_path(raw["train_table"])
    val_table = resolve_path(raw["val_table"])
    test_table = resolve_path(raw.get("test_table"))
    pdb_dir = resolve_path(raw["pdb_dir"])
    sasa_dir = out_dir / "sasa_three"
    prepare_sasa_for_tables([train_table, val_table, test_table], pdb_dir, sasa_dir, pred_config)

    train_items = filter_labeled(build_dataset_from_table(train_table, pdb_dir, sasa_dir, pred_config), target)
    val_items = filter_labeled(build_dataset_from_table(val_table, pdb_dir, sasa_dir, pred_config), target)
    test_items = filter_labeled(build_dataset_from_table(test_table, pdb_dir, sasa_dir, pred_config), target) if test_table else []
    print(f"[Count] train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    if not train_items or not val_items:
        raise RuntimeError("Training and validation sets must both contain labeled samples.")

    y_train = torch.tensor([float(item[target]) for item in train_items], dtype=torch.float32, device=device)
    scaler = ZScoreTarget()
    scaler.fit(y_train)
    y_std = float(scaler.std.cpu()) if scaler.std is not None else None
    print(f"[Scaler] mu={float(scaler.mu):.4f} std={float(scaler.std):.4f}")

    train_loader = make_loader(train_items, pred_config, shuffle=True)
    val_loader = make_loader(val_items, pred_config, shuffle=False)
    test_loader = make_loader(test_items, pred_config, shuffle=False) if test_items else None

    model = build_model(pred_config, device=device).to(device)
    epochs = int(raw.get("epochs", 20))
    freeze_epochs = int(raw.get("freeze_epochs", 0))
    patience = int(raw.get("patience", 8))
    accum_steps = int(raw.get("accum_steps", 1))
    grad_clip = raw.get("grad_clip", 1.0)
    target_noise = float(raw.get("target_noise", 0.0))
    warmup_ratio = float(raw.get("warmup_ratio", 0.02))
    lr_min_factor = float(raw.get("lr_min_factor", 0.1))
    weight_decay = float(raw.get("weight_decay", 1e-4))
    alpha_a = float(raw.get("loss_alpha_A", 0.0))
    alpha_b = float(raw.get("loss_alpha_B", 0.2))
    lr_head_a = float(raw.get("lr_head_A", raw.get("lr_head", 1e-4)))
    lr_head_b = float(raw.get("lr_head_B", raw.get("lr_head", 1e-4)))
    lr_backbone_b = raw.get("lr_backbone_B", raw.get("lr_backbone", 1e-5))
    lr_backbone_b = None if lr_backbone_b is None else float(lr_backbone_b)
    early_metric = str(raw.get("early_metric", "rmse"))
    ckpt_path = out_dir / pred_config.checkpoint_name
    log_path = out_dir / "train_log.csv"

    def score(metrics: dict[str, float]) -> float:
        if early_metric in {"rmse", "mae", "loss"}:
            return float(metrics[early_metric])
        if early_metric in {"r2", "pearson", "ccc"}:
            return -float(metrics[early_metric])
        raise ValueError(f"Unsupported early_metric: {early_metric}")

    best_score = float("inf")
    best_epoch = -1
    best_phase = None
    bad = 0
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    def run_phase(phase: str, start_ep: int, end_ep: int, alpha: float, freeze: bool, lr_head: float, lr_backbone: float | None):
        nonlocal best_score, best_epoch, best_phase, bad
        if end_ep < start_ep:
            return
        bad = 0
        set_backbone_trainable(model, trainable=not freeze)
        set_grad_checkpointing(model, enabled=bool(raw.get("grad_ckpt", False)) and (not freeze))
        updates = max(1, math.ceil(len(train_loader) / max(1, accum_steps)))
        opt, sched = build_optimizer_and_scheduler(
            model,
            updates_per_epoch=updates,
            epochs=end_ep - start_ep + 1,
            lr_head=lr_head,
            lr_backbone=lr_backbone,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_min_factor=lr_min_factor,
        )
        for epoch in range(start_ep, end_ep + 1):
            t0 = time.time()
            tr = run_epoch(
                model, train_loader, scaler, is_train=True, optimizer=opt, scheduler=sched,
                amp_scaler=scaler_amp, accum_steps=accum_steps, loss_alpha=alpha,
                grad_clip=grad_clip, target_noise=target_noise, y_std=y_std, use_amp=use_amp,
            )
            va = run_epoch(model, val_loader, scaler, is_train=False, loss_alpha=alpha, use_amp=False)
            row = {
                "phase": phase, "epoch": epoch, "alpha": alpha, "seconds": round(time.time() - t0, 3),
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"val_{k}": v for k, v in va.items()},
            }
            append_log_csv(log_path, row)
            print(
                f"[{phase}:{epoch:03d}] train_rmse={tr['rmse']:.4f} val_rmse={va['rmse']:.4f} "
                f"val_mae={va['mae']:.4f} val_r2={va['r2']:.4f} val_ccc={va['ccc']:.4f}"
            )
            cur = score(va)
            if cur < best_score - 1e-12:
                best_score = cur
                best_epoch = epoch
                best_phase = phase
                bad = 0
                torch.save({
                    "model": model.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": {"train": raw, "predictor": asdict(pred_config)},
                    "best_epoch": best_epoch,
                    "best_phase": best_phase,
                    "best_alpha": alpha,
                    "best_score": best_score,
                    "early_metric": early_metric,
                }, ckpt_path)
                print(f"[Saved] best -> {ckpt_path}")
            else:
                bad += 1
                if bad >= patience:
                    print(f"[EarlyStop] best_epoch={best_epoch} phase={best_phase} score={best_score:.6f}")
                    return

    if freeze_epochs > 0:
        run_phase("A", 1, min(freeze_epochs, epochs), alpha_a, True, lr_head_a, None)
    if epochs > freeze_epochs:
        run_phase("B", freeze_epochs + 1, epochs, alpha_b, False, lr_head_b, lr_backbone_b)

    if not ckpt_path.exists():
        raise RuntimeError("Training finished without saving a checkpoint.")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    best_alpha = float(state.get("best_alpha", alpha_b))
    val_metrics = run_epoch(model, val_loader, scaler, is_train=False, loss_alpha=best_alpha, use_amp=False)
    metrics = {"best_epoch": best_epoch, "best_phase": best_phase, "best_score": best_score, "val": val_metrics}
    write_predictions_csv(model, val_loader, scaler, pred_config, device, out_dir / "val_predictions.csv")
    if test_loader is not None:
        test_metrics = run_epoch(model, test_loader, scaler, is_train=False, loss_alpha=best_alpha, use_amp=False)
        metrics["test"] = test_metrics
        write_predictions_csv(model, test_loader, scaler, pred_config, device, out_dir / "test_predictions.csv")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    with open(out_dir / "train_config_resolved.json", "w", encoding="utf-8") as fh:
        json.dump({"train": raw, "predictor": asdict(pred_config)}, fh, ensure_ascii=False, indent=2)
    print(f"[Done] best checkpoint: {ckpt_path}")
    return {"checkpoint": str(ckpt_path), "metrics": metrics}
