from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer

from textvae.lit_module import LitTextVAE


@dataclass
class SweetSpot:
    ckpt_path: str
    model_name: str
    latent_dim: int
    beta: float


def load_sweet_spot(sweet_spot_json: str | Path) -> SweetSpot:
    p = Path(sweet_spot_json)
    d = json.loads(p.read_text(encoding="utf-8"))
    spot = d.get("sweet_spot", d)

    ckpt = spot.get("best_model_path") or d.get("best_model_path")
    if not ckpt:
        raise KeyError("Could not find best_model_path in sweet_spot.json")

    model_name = spot.get("model_name") or d.get("model_name")
    if not model_name:
        raise KeyError("Could not find model_name in sweet_spot.json")

    latent_dim = int(spot.get("latent_dim", 0))
    beta = float(spot.get("beta", 0.0))

    return SweetSpot(
        ckpt_path=str(ckpt),
        model_name=str(model_name),
        latent_dim=latent_dim,
        beta=beta,
    )


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> LitTextVAE:
    model = LitTextVAE.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    # If transformer is frozen, keep it in eval mode for determinism
    if getattr(model.hparams, "freeze_transformer", False):
        model.vae.encoder.model.eval()
    return model


@torch.no_grad()
def embed_texts(
    model: LitTextVAE,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    max_length: int,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Returns normalized sentence embeddings for a list of texts: (N, D)
    Uses the frozen transformer encoder (not VAE decoder).
    """
    embs: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tok = tokenizer(
            list(batch),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)

        emb = model.vae.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )  # (B, D)
        emb = F.normalize(emb, p=2, dim=1)
        embs.append(emb.detach().cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def encode_to_emb_and_mu(
    model: LitTextVAE,
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      emb: (D,) normalized original embedding
      mu : (Z,) latent mean
    """
    tok = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    out = model.vae(input_ids=input_ids, attention_mask=attention_mask)
    emb = F.normalize(out.emb.squeeze(0), p=2, dim=0)
    mu = out.mu.squeeze(0)
    return emb.detach().cpu(), mu.detach().cpu()


@torch.no_grad()
def decode_from_mu(
    model: LitTextVAE, mu: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    mu: (Z,) on CPU or GPU
    returns: recon_emb (D,) normalized on CPU
    """
    mu = mu.to(device).unsqueeze(0)  # (1, Z)
    recon = model.vae.decoder(mu).squeeze(0)  # (D,)
    recon = F.normalize(recon, p=2, dim=0)
    return recon.detach().cpu()


def sample_pairs_different_labels(
    ds: Dataset,
    num_pairs: int,
    seed: int,
    min_len: int = 20,
) -> List[Tuple[Dict, Dict]]:
    """
    Samples sentence pairs (A, B) where labels differ.
    Returns list of dict rows with keys: text, label (AG News format).
    """
    rng = random.Random(seed)
    # Group indices by label
    by_label: Dict[int, List[int]] = {}
    for i in range(len(ds)):
        row = ds[i]
        txt = row.get("text", "")
        if not isinstance(txt, str) or len(txt) < min_len:
            continue
        lab = int(row["label"])
        by_label.setdefault(lab, []).append(i)

    labels = sorted(by_label.keys())
    if len(labels) < 2:
        raise RuntimeError("Need at least 2 labels to sample different-label pairs.")

    pairs: List[Tuple[Dict, Dict]] = []
    for _ in range(num_pairs):
        a_lab = rng.choice(labels)
        b_lab = rng.choice([l for l in labels if l != a_lab])
        a_idx = rng.choice(by_label[a_lab])
        b_idx = rng.choice(by_label[b_lab])
        pairs.append((ds[a_idx], ds[b_idx]))
    return pairs


@torch.no_grad()
def topk_nearest_texts(
    query_emb: torch.Tensor,  # (D,) normalized on CPU
    corpus_embs: torch.Tensor,  # (N, D) normalized on CPU
    corpus_texts: Sequence[str],
    k: int = 3,
) -> List[Tuple[int, float, str]]:
    """
    Returns [(idx, cosine, text), ...] top-k by cosine similarity.
    """
    q = query_emb.unsqueeze(0)  # (1, D)
    sims = (q @ corpus_embs.T).squeeze(0)  # (N,)
    vals, idxs = torch.topk(sims, k=min(k, sims.numel()))
    out: List[Tuple[int, float, str]] = []
    for idx, v in zip(idxs.tolist(), vals.tolist()):
        out.append((idx, float(v), corpus_texts[idx]))
    return out
