from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

try:
    from torch_geometric.nn import global_mean_pool
    from torch_geometric.nn import TransformerConv
    GNN_LAYER_TYPE = "TransformerConv"
except ImportError:
    try:
        from torch_geometric.nn import GCNConv, global_mean_pool
        GNN_LAYER_TYPE = "GCNConv"
    except ImportError as exc:
        raise ImportError(
            "torch-geometric is required for the checkpointed GNN model. "
            "Install a CPU or CUDA build that matches your PyTorch version."
        ) from exc

from transformers import AutoModel, AutoTokenizer

from .config import PredictorConfig


def space_separate(seq: str) -> str:
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = "".join(c if c.upper() in valid else "X" for c in str(seq).upper())
    return " ".join(list(cleaned))


class AttentionPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.mlp(x).squeeze(-1)
        neg_large = torch.tensor(-1e9, dtype=score.dtype, device=score.device)
        score = score.masked_fill(mask == 0, neg_large)
        attn = torch.softmax(score, dim=1)
        pooled = (x * attn.unsqueeze(-1)).sum(1)
        return pooled, attn


class ChannelImportanceBlock(nn.Module):
    def __init__(self, ch_dims: list[int], att_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ch_dims = ch_dims
        self.proj = nn.ModuleList([nn.Linear(d, att_dim) for d in ch_dims])
        self.mha = nn.MultiheadAttention(embed_dim=att_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(att_dim)
        self.ff = nn.Sequential(
            nn.Linear(att_dim, att_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(att_dim * 2, att_dim),
        )
        self.ln2 = nn.LayerNorm(att_dim)
        self.score_head = nn.Sequential(nn.LayerNorm(att_dim), nn.Linear(att_dim, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pieces: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [proj(x) for x, proj in zip(pieces, self.proj)]
        hidden = torch.stack(tokens, dim=1)
        attn_out, _ = self.mha(hidden, hidden, hidden)
        hidden = self.ln1(hidden + attn_out)
        hidden = self.ln2(hidden + self.ff(hidden))
        weights = self.softmax(self.score_head(hidden).squeeze(-1))
        fused = torch.cat([x * weights[:, i:i + 1] for i, x in enumerate(pieces)], dim=1)
        return fused, weights


class ProteinGNN(nn.Module):
    def __init__(self, in_dim: int = 1280, hidden: int = 128, edge_dim: int = 17, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
        )
        self.convs = nn.ModuleList()
        for _ in range(layers):
            if GNN_LAYER_TYPE == "TransformerConv":
                conv = TransformerConv(
                    in_channels=hidden,
                    out_channels=hidden // 2,
                    heads=2,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
            else:
                conv = GCNConv(in_channels=hidden, out_channels=hidden)
            self.convs.append(conv)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        hidden = self.node_proj(x)
        for conv in self.convs:
            residual = hidden
            if GNN_LAYER_TYPE == "TransformerConv":
                hidden = conv(hidden, edge_index, edge_attr)
            else:
                hidden = conv(hidden, edge_index)
            hidden = self.act(hidden)
            hidden = self.dropout(hidden)
            hidden = hidden + residual
        return global_mean_pool(hidden, batch)


class MultiChannelPIPredictor(nn.Module):
    def __init__(
        self,
        *,
        d_seqfeat: int = 45,
        use_esm2: bool = True,
        esm2_model: str = "facebook/esm2_t33_650M_UR50D",
        esm2_hidden: int = 1280,
        esm2_cache_dir: Optional[str] = None,
        esm2_finetune: bool = False,
        use_protbert: bool = True,
        protbert_model: str = "Rostlab/prot_bert",
        protbert_hidden: int = 1024,
        protbert_cache_dir: Optional[str] = None,
        protbert_finetune: bool = False,
        protbert_pool: str = "attn",
        use_gnn: bool = True,
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        d_struct_graph: int = 128,
        fusion_hidden: int = 512,
        dropout: float = 0.2,
        device: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        esm2_max_len: Optional[int] = None,
        protbert_max_len: Optional[int] = None,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_seq_len = max_seq_len
        self.esm2_max_len = esm2_max_len or max_seq_len
        self.protbert_max_len = protbert_max_len or max_seq_len
        self.dropout = dropout

        self.use_seqfeat = d_seqfeat > 0
        if self.use_seqfeat:
            self.seq_proj = nn.Sequential(
                nn.LayerNorm(d_seqfeat),
                nn.Linear(d_seqfeat, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.use_esm2 = use_esm2
        self.esm2_hidden = esm2_hidden
        if self.use_esm2:
            self.esm2_tok = AutoTokenizer.from_pretrained(esm2_model, cache_dir=esm2_cache_dir)
            self.esm2_enc = AutoModel.from_pretrained(esm2_model, cache_dir=esm2_cache_dir)
            if not esm2_finetune:
                for p in self.esm2_enc.parameters():
                    p.requires_grad = False
                self.esm2_enc.eval()
            self.esm2_proj = nn.Sequential(
                nn.LayerNorm(esm2_hidden),
                nn.Linear(esm2_hidden, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.use_protbert = use_protbert
        self.protbert_pool = protbert_pool
        if self.use_protbert:
            self.pb_tok = AutoTokenizer.from_pretrained(
                protbert_model,
                do_lower_case=False,
                cache_dir=protbert_cache_dir,
            )
            self.pb_enc = AutoModel.from_pretrained(protbert_model, cache_dir=protbert_cache_dir)
            if not protbert_finetune:
                for p in self.pb_enc.parameters():
                    p.requires_grad = False
                self.pb_enc.eval()
            self.pb_proj = nn.Sequential(
                nn.LayerNorm(protbert_hidden),
                nn.Linear(protbert_hidden, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.pb_attn = AttentionPool(256, hidden=128)
            self.pb_part_gate = nn.Sequential(
                nn.LayerNorm(256 * 3),
                nn.Linear(256 * 3, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 3),
            )
            self.pb_part_softmax = nn.Softmax(dim=1)

        self.use_gnn = use_gnn
        self.gnn_out_dim = gnn_hidden
        if self.use_gnn:
            self.gnn = ProteinGNN(in_dim=esm2_hidden, hidden=gnn_hidden, edge_dim=17, layers=gnn_layers, dropout=dropout)

        self.ch_dims: list[int] = []
        if self.use_seqfeat:
            self.ch_dims.append(128)
        if self.use_esm2:
            self.ch_dims.append(256)
        if self.use_protbert:
            self.ch_dims.append(256)
        if self.use_gnn:
            self.ch_dims.append(self.gnn_out_dim)

        in_dim = sum(self.ch_dims)
        self.channel_importance = ChannelImportanceBlock(self.ch_dims, att_dim=128) if len(self.ch_dims) > 1 else None
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, 2),
        )
        self.to(self.device)

    def _get_esm2_outputs(self, seqs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tok_kwargs: dict[str, Any] = {"return_tensors": "pt", "padding": True, "truncation": True}
        if self.esm2_max_len:
            tok_kwargs["max_length"] = self.esm2_max_len
        batch = self.esm2_tok(seqs, **tok_kwargs)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.set_grad_enabled(self.esm2_enc.training):
            out = self.esm2_enc(**batch)
        hidden = out.last_hidden_state
        return hidden[:, 0, :], hidden

    def _build_token_mask(self, list_of_mask1b: Optional[list[list[int]]], batch_size: int, token_len: int) -> torch.Tensor:
        mask = torch.zeros(batch_size, token_len, device=self.device, dtype=torch.float32)
        if not list_of_mask1b:
            return mask
        for i, residue_mask in enumerate(list_of_mask1b):
            if residue_mask is None:
                continue
            length = min(len(residue_mask), token_len - 2)
            if length > 0:
                mask[i, 1:1 + length] = torch.tensor(residue_mask[:length], device=self.device, dtype=torch.float32)
        return mask

    def _safe_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.clone()
        zero_rows = mask.sum(dim=1) <= 0
        if zero_rows.any() and mask.shape[1] > 1:
            mask[zero_rows, 1] = 1.0
        return mask

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(1, keepdim=True).clamp_min(1e-6)
        return (x * mask.unsqueeze(-1)).sum(1) / denom

    def forward(
        self,
        seqs: list[str],
        seq_feats: Optional[torch.Tensor] = None,
        masks: Optional[Any] = None,
        batch_graph=None,
    ) -> dict[str, torch.Tensor | dict | None]:
        pieces: list[torch.Tensor] = []
        channel_weights = None
        pb_part_weights = None
        attn_surface = None

        esm_cls, esm_full = None, None
        if self.use_esm2 or self.use_gnn:
            esm_cls, esm_full = self._get_esm2_outputs(seqs)

        if self.use_seqfeat and seq_feats is not None:
            pieces.append(self.seq_proj(seq_feats.to(self.device)))

        if self.use_esm2 and esm_cls is not None:
            pieces.append(self.esm2_proj(esm_cls))

        if self.use_protbert:
            spaced = [space_separate(s) for s in seqs]
            tok_kwargs: dict[str, Any] = {"return_tensors": "pt", "padding": True, "truncation": True}
            if self.protbert_max_len:
                tok_kwargs["max_length"] = self.protbert_max_len
            pb_batch = self.pb_tok(spaced, **tok_kwargs)
            pb_batch = {k: v.to(self.device) for k, v in pb_batch.items()}
            with torch.set_grad_enabled(self.pb_enc.training):
                pb_out = self.pb_enc(**pb_batch)
            projected = self.pb_proj(pb_out.last_hidden_state)
            batch_size, token_len = projected.shape[:2]

            if masks is None:
                masks_dict = {"exposed": None, "semi": None, "internal": None}
            elif isinstance(masks, dict):
                masks_dict = {
                    "exposed": masks.get("exposed"),
                    "semi": masks.get("semi"),
                    "internal": masks.get("internal"),
                }
            else:
                masks_dict = {"exposed": masks, "semi": None, "internal": None}

            m_ex = self._build_token_mask(masks_dict["exposed"], batch_size, token_len)
            m_se = self._build_token_mask(masks_dict["semi"], batch_size, token_len)
            m_in = self._build_token_mask(masks_dict["internal"], batch_size, token_len)

            if self.protbert_pool == "mean":
                v_ex = self._masked_mean(projected, m_ex)
                v_se = self._masked_mean(projected, m_se)
                v_in = self._masked_mean(projected, m_in)
            else:
                v_ex, att_ex = self.pb_attn(projected, self._safe_mask(m_ex))
                v_se, att_se = self.pb_attn(projected, self._safe_mask(m_se))
                v_in, att_in = self.pb_attn(projected, self._safe_mask(m_in))
                attn_surface = {"exposed": att_ex, "semi": att_se, "internal": att_in}

            gate_in = torch.cat([v_ex, v_se, v_in], dim=1)
            pb_part_weights = self.pb_part_softmax(self.pb_part_gate(gate_in))
            pb_pooled = (
                v_ex * pb_part_weights[:, 0:1]
                + v_se * pb_part_weights[:, 1:2]
                + v_in * pb_part_weights[:, 2:3]
            )
            pieces.append(pb_pooled)

        if self.use_gnn:
            has_graph = batch_graph is not None
            is_dummy = False
            if has_graph and hasattr(batch_graph, "is_dummy"):
                is_dummy = bool(torch.as_tensor(batch_graph.is_dummy).all().item())
            if has_graph and not is_dummy:
                batch_graph = batch_graph.to(self.device)
                node_feats_list = []
                batch_size = esm_full.shape[0] if esm_full is not None else len(seqs)
                for i in range(batch_size):
                    if hasattr(batch_graph, "ptr"):
                        n_nodes = int((batch_graph.ptr[i + 1] - batch_graph.ptr[i]).item())
                    else:
                        n_nodes = int((batch_graph.batch == i).sum().item())
                    if esm_full is None:
                        node_feats_list.append(torch.zeros(n_nodes, self.esm2_hidden, device=self.device))
                        continue
                    valid_len = min(n_nodes, esm_full.shape[1] - 2)
                    feats = esm_full[i, 1:1 + valid_len, :]
                    if valid_len < n_nodes:
                        pad = torch.zeros(n_nodes - valid_len, feats.shape[1], device=self.device)
                        feats = torch.cat([feats, pad], dim=0)
                    node_feats_list.append(feats)
                flat_node_feats = torch.cat(node_feats_list, dim=0)
                pieces.append(self.gnn(flat_node_feats, batch_graph.edge_index, batch_graph.edge_attr, batch_graph.batch))
            else:
                pieces.append(torch.zeros(len(seqs), self.gnn_out_dim, device=self.device))

        if self.channel_importance:
            fused, channel_weights = self.channel_importance(pieces)
        else:
            fused = torch.cat(pieces, dim=1)
        out = self.head(fused)
        return {
            "mu_z": out[:, 0],
            "log_var_z": out[:, 1],
            "attn_surface": attn_surface,
            "channel_weights": channel_weights,
            "pb_part_weights": pb_part_weights,
        }


class ZScoreTarget:
    def __init__(self):
        self.mu: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, y: torch.Tensor):
        mu = y.mean()
        var = ((y - mu) ** 2).mean()
        std = torch.sqrt(var).clamp_min(1e-8)
        self.mu = mu.detach().float()
        self.std = std.detach().float()

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.mu is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted or loaded.")
        return (y - self.mu.to(y.device)) / self.std.to(y.device)

    def inverse(self, y_z: torch.Tensor) -> torch.Tensor:
        if self.mu is None or self.std is None:
            raise RuntimeError("Scaler has not been loaded.")
        return y_z * self.std.to(y_z.device) + self.mu.to(y_z.device)

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self.mu is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted or loaded.")
        return {"mu": self.mu.cpu(), "std": self.std.cpu()}

    def load_state_dict(self, state: dict):
        self.mu = state["mu"].float()
        self.std = state["std"].float()


def build_model(config: PredictorConfig, device: str | torch.device) -> MultiChannelPIPredictor:
    return MultiChannelPIPredictor(
        d_seqfeat=config.d_seqfeat,
        use_esm2=config.use_esm2,
        esm2_model=config.esm2_model,
        esm2_hidden=config.esm2_hidden,
        esm2_cache_dir=config.esm2_cache,
        esm2_finetune=config.esm2_finetune,
        use_protbert=config.use_protbert,
        protbert_model=config.protbert_model,
        protbert_hidden=config.protbert_hidden,
        protbert_cache_dir=config.protbert_cache,
        protbert_finetune=config.protbert_finetune,
        protbert_pool=config.protbert_pool,
        use_gnn=config.use_gnn,
        gnn_hidden=config.gnn_hidden,
        gnn_layers=config.gnn_layers,
        d_struct_graph=config.d_struct_graph,
        fusion_hidden=config.fusion_hidden,
        dropout=config.dropout,
        device=str(device),
        max_seq_len=config.max_seq_len,
    )


def normalize_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state):
        return state
    return {k.removeprefix("module."): v for k, v in state.items()}
