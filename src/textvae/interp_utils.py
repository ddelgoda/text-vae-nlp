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
from typing import Tuple

from textvae.lit_module import LitTextVAE
PROJECT_ROOT = Path(__file__).resolve().parents[2]




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
        ) # (B, D)
        emb = F.normalize(emb, p=2, dim=1)
        embs.append(emb.detach().cpu())

    return torch.cat(embs, dim=0) # (N, D)


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
    z = mu.to(device).unsqueeze(0)  # (1, Z)
    recon = model.vae.decoder(z).squeeze(0)  # (D,)
    recon = F.normalize(recon, p=2, dim=0)
    return recon.detach().cpu()

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine for already-normalized vectors (CPU tensors)."""
    a = F.normalize(a, p=2, dim=0)
    b = F.normalize(b, p=2, dim=0)
    return float((a * b).sum().item())
    
def sample_pairs_different_labels(
    ds: Dataset,
    num_pairs: int,
    seed: int,
    min_len: int = 20,
    max_cos_sim: float | None = 0.95,
    # If you pass these, we can filter out "too similar" pairs reliably:
    corpus_embs: torch.Tensor | None = None, # (N, D) normalized
    ) -> List[Tuple[Dict, Dict]]:
    """
    Samples sentence pairs (A, B) where labels differ.

    If corpus_embs is provided (shape (N, D)), also enforces that
    cosine(corpus_embs[a_idx], corpus_embs[b_idx]) <= max_cos_sim
    to avoid trivial pairs that are nearly identical in embedding space.

    Returns list of dict rows with keys: text, label (AG News format).
    """
    rng = random.Random(seed)

    # Group valid indices by label
    by_label: Dict[int, List[int]] = {}
    valid_idxs: List[int] = []
    for i in range(len(ds)):
        row = ds[i]
        txt = row.get("text", "")
        if not isinstance(txt, str) or len(txt) < min_len:
            continue
        lab = int(row["label"])
        by_label.setdefault(lab, []).append(i)
        valid_idxs.append(i)

    labels = sorted(by_label.keys())
    if len(labels) < 2:
        raise RuntimeError("Need at least 2 labels to sample different-label pairs.")

    # If we want cosine filtering, we need embeddings
    use_cos_filter = max_cos_sim is not None and corpus_embs is not None
    if use_cos_filter and corpus_embs.dim() != 2:
        raise ValueError("corpus_embs must be a 2D tensor of shape (N, D).")

    pairs: List[Tuple[Dict, Dict]] = []
    used: set[tuple[int, int]] = set()

    # Try multiple attempts to satisfy constraints
    max_tries = max(200, num_pairs * 200)
    tries = 0

    while len(pairs) < num_pairs and tries < max_tries:
        tries += 1

        a_lab = rng.choice(labels)
        b_lab = rng.choice([l for l in labels if l != a_lab])

        a_idx = rng.choice(by_label[a_lab])
        b_idx = rng.choice(by_label[b_lab])

        key = (a_idx, b_idx)
        if key in used:
            continue

        # Optional: avoid trivial nearly-identical pairs in embedding space
        if use_cos_filter:
            ca = corpus_embs[a_idx]
            cb = corpus_embs[b_idx]
            cos_ab = cosine(ca, cb)
            if cos_ab > float(max_cos_sim):
                continue

        used.add(key)
        pairs.append((ds[a_idx], ds[b_idx]))

    if len(pairs) < num_pairs:
        msg = (
            f"Only sampled {len(pairs)}/{num_pairs} different-label pairs "
            f"after {tries} tries."
        )
        if max_cos_sim is not None and corpus_embs is None:
            msg += " (Tip: pass corpus_embs to enable max_cos_sim filtering.)"
        raise RuntimeError(msg)

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

def find_ckpt_for_run(runs_dir: str | Path, latent_dim: int, beta: float) -> tuple[str, float]:
    runs_dir = Path(runs_dir)

    if not runs_dir.is_absolute():
        runs_dir = (PROJECT_ROOT/runs_dir).resolve()
        print("run_dir", runs_dir)
    

    # if user passed a single run dir, shortcut
    ms=runs_dir/"metrics_summary.json"
    print("ms", ms)
    
    if ms.exists():
        files=[ms]
    else:
        files=sorted(runs_dir.rglob("metrics_summary.json"))

    if not files:
        raise FileNotFoundError(f"No metrics_summary.json found under {runs_dir}")
    # Match by latent_dim + beta (float-safe)
    candidates = []
    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        if int(d.get("latent_dim", -1)) != latent_dim:
            continue
        b = float(d.get("beta", float("nan")))
        if abs(b - beta) > 1e-9:
            continue
        ckpt = d.get("best_model_path") or d.get("last_model_path")
        kl_val = float(d.get("kl_loss", float("nan")))
        if ckpt and Path(ckpt).exists():
            candidates.append((float(d.get("best_model_score", 1e18)), ckpt, kl_val))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for latent_dim={latent_dim}, beta={beta}. "
            f"Did you train that combo?"
        )
    # lowest best_model_score is best
    candidates.sort(key=lambda x: x[0])
    _, ckpt_path, kl_val = candidates[0]
    return ckpt_path, kl_val 