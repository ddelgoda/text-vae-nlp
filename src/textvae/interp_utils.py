from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer

from textvae.lit_module import LitTextVAE

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> LitTextVAE:
    model = LitTextVAE.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    # If transformer is frozen, keep it in eval mode for determinism
    if getattr(model.hparams, "freeze_transformer", False):
        model.vae.encoder.model.eval()
    return model

def path_geometry(E: List[torch.Tensor]) -> Dict[str, float]:
    """
    E: list of (D,) CPU tensors (ideally normalized)
    returns: path_len, curv_mean, curv_sum, delta_mean
    """
    if len(E) < 2:
        return {"path_len": 0.0, "curv_mean": 0.0, "curv_sum": 0.0, "delta_mean": 0.0}

    deltas = []
    for i in range(len(E) - 1):
        deltas.append(float(torch.norm(E[i + 1] - E[i]).item()))
    path_len = float(sum(deltas))
    delta_mean = float(sum(deltas) / len(deltas))

    if len(E) < 3:
        return {"path_len": path_len, "curv_mean": 0.0, "curv_sum": 0.0, "delta_mean": delta_mean}

    curvs = []
    for i in range(1, len(E) - 1):
        curvs.append(float(torch.norm(E[i + 1] - 2 * E[i] + E[i - 1]).item()))
    curv_sum = float(sum(curvs))
    curv_mean = float(curv_sum / len(curvs))

    return {"path_len": path_len, "curv_mean": curv_mean, "curv_sum": curv_sum, "delta_mean": delta_mean}

def curvature_ratios(
    geom_lat: Dict[str, float],
    geom_emb: Dict[str, float],
    eps: float = 1e-6,
    cap: float = 1e4,
    ) -> Dict[str, float]:
    """
    Ratios relative to embedding-linear baseline.
    eps prevents division by ~0.
    cap prevents exploding values from dominating plots/tables.
    """
    path_ratio = geom_lat["path_len"] / max(geom_emb["path_len"], eps)
    curv_ratio = geom_lat["curv_mean"] / max(geom_emb["curv_mean"], eps)

    path_ratio_capped = min(path_ratio, cap)
    curv_ratio_capped = min(curv_ratio, cap)

    return {
        "path_ratio": float(path_ratio),
        "curv_ratio_mean": float(curv_ratio),
        "path_ratio_capped": float(path_ratio_capped),
        "curv_ratio_mean_capped": float(curv_ratio_capped),
    }



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
    logvar = out.logvar.squeeze(0)
    return emb.cpu(), mu.cpu(), logvar.cpu()

@torch.no_grad()
def decode_from_mu_logvar_sampled(
    model: LitTextVAE,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    device: torch.device,
    n_samples: int = 8,
) -> torch.Tensor:
    """
    Returns normalized recon embedding (D,) on CPU using Monte Carlo sampling.
    """
    mu = mu.to(device)
    logvar = logvar.to(device)
    std = torch.exp(0.5 * logvar)

    outs = []
    for _ in range(n_samples):
        eps = torch.randn_like(std)
        z = mu + std * eps
        recon = model.vae.decoder(z.unsqueeze(0)).squeeze(0)
        recon = F.normalize(recon, p=2, dim=0)
        outs.append(recon)

    recon_mean = torch.stack(outs, dim=0).mean(dim=0)
    recon_mean = F.normalize(recon_mean, p=2, dim=0)
    return recon_mean.detach().cpu()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine for already-normalized vectors (CPU tensors)."""
    a = F.normalize(a, p=2, dim=0)
    b = F.normalize(b, p=2, dim=0)
    return float((a * b).sum().item())

def sample_pairs_sts_low_cos(
    ds: Dataset,
    num_pairs: int,
    seed: int,
    sim_min: float = 0.0,
    sim_max: float = 2.0,
    min_len: int = 1,
    pool_size: int = 2000,
    batch_size: int = 32,
    # optional deps (can be defaulted)
    model: Optional["LitTextVAE"] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_length: int = 128,
    device: Optional[torch.device] = None,
    ) -> List[Tuple[str, str, float, float]]:
    """
    Returns (s1, s2, sim_score, cos_emb) for the lowest-cosine pairs
    among candidates within [sim_min, sim_max].

    Requires embeddings; if model/tokenizer/device are not provided,
    uses defaults (model must be provided OR you must load it before calling).
    """

    # ---- defaults ----
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model is None:
        raise ValueError(
            "sample_pairs_sts_low_cos requires `model` (LitTextVAE) to embed texts. "
            "Pass model=... (loaded from your checkpoint)."
        )

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    # ---- candidate collection ----
    cands: List[Tuple[str, str, float]] = []
    for i in idxs:
        row = ds[i]
        s1 = row.get("sentence1", "")
        s2 = row.get("sentence2", "")
        if not isinstance(s1, str) or not isinstance(s2, str):
            continue
        if len(s1) < min_len or len(s2) < min_len:
            continue

        s1_clean = " ".join(s1.split())
        s2_clean = " ".join(s2.split())
        if s1_clean == s2_clean:
            continue

        s = float(row["similarity_score"])
        if sim_min <= s <= sim_max:
            cands.append((s1, s2, s))
            if len(cands) >= pool_size:
                break

    if not cands:
        raise RuntimeError(
            f"No candidate pairs found in sim range [{sim_min}, {sim_max}]"
        )

    # ---- embed endpoints ----
    s1s = [c[0] for c in cands]
    s2s = [c[1] for c in cands]

    emb1 = embed_texts(
    model=model,
    tokenizer=tokenizer,
    texts=s1s,
    max_length=max_length,
    device=device,
    batch_size=batch_size,
    )
    emb2 = embed_texts(
    model=model,
    tokenizer=tokenizer,
    texts=s2s,
    max_length=max_length,
    device=device,
    batch_size=batch_size,
    )

    # ---- cosine scoring ----
    scored: List[Tuple[float, str, str, float]] = []
    for (s1, s2, sim_score), e1, e2 in zip(cands, emb1, emb2):
        cos12 = cosine(e1, e2)
        scored.append((cos12, s1, s2, sim_score))

    scored.sort(key=lambda x: x[0]) # lowest cosine first
    picked = scored[:num_pairs]

    return [(s1, s2, sim_score, cos12) for (cos12, s1, s2, sim_score) in picked]


def topk_nearest_texts(
    query_emb: torch.Tensor,
    corpus_embs: torch.Tensor,
    corpus_texts: list[str],
    k: int = 5,
):
    """
    query_emb: (D,) or (1,D)
    corpus_embs: (N,D)
    Returns: list of (rank, sim, text)
    """

    # ---- shape safety ----
    if query_emb.dim() == 2:
        # (1, D) -> (D,)
        if query_emb.shape[0] != 1:
            raise ValueError(f"query_emb should be (D,) or (1,D), got {tuple(query_emb.shape)}")
        query_emb = query_emb[0]
    elif query_emb.dim() != 1:
        raise ValueError(f"query_emb should be (D,) or (1,D), got {tuple(query_emb.shape)}")

    if corpus_embs.dim() != 2:
        raise ValueError(f"corpus_embs should be (N,D), got {tuple(corpus_embs.shape)}")

    # ensure on same device/dtype
    query_emb = query_emb.to(corpus_embs.device, dtype=corpus_embs.dtype)

    # cosine sims if both are normalized
    sims = corpus_embs @ query_emb # (N,)
    kk = min(int(k), sims.shape[0])

    vals, idxs = torch.topk(sims, k=kk)

    out = []
    for rank, (idx, sim) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
        out.append((rank, float(sim), corpus_texts[idx]))
    return out


def find_ckpt_for_run(runs_dir: str | Path, latent_dim: int, beta: float) -> tuple[str, float]:
    runs_dir = Path(runs_dir)

    if not runs_dir.is_absolute():
        runs_dir = (PROJECT_ROOT/runs_dir).resolve()

    # if user passed a single run dir, shortcut
    ms=runs_dir/"metrics_summary.json"

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
            #candidates.append((float(d.get("last_model_score", 1e18)), ckpt, kl_val))
            ckpt_path = Path(ckpt)
            candidates.append((ckpt_path.stat().st_mtime, ckpt, kl_val))
            candidates.sort(key=lambda x: x[0], reverse=True) # newest first
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for latent_dim={latent_dim}, beta={beta}. "
            f"Did you train that combo?"
        )
    # lowest best_model_score is best
    candidates.sort(key=lambda x: x[0])
    _, ckpt_path, kl_val = candidates[0]
    return ckpt_path, kl_val
