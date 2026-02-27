from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from textvae.interp_utils import find_ckpt_for_run
import torch.nn.functional as F

# --- Ensure src/ is importable when running as a script ---

ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "src"))

from textvae.interp_utils import (  # noqa: E402
    decode_from_mu,
    embed_texts,
    encode_to_emb_and_mu,
    load_model_from_ckpt,
    load_sweet_spot,
    sample_pairs_different_labels,
    topk_nearest_texts,
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine for already-normalized vectors (CPU tensors)."""
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return float((a * b).sum().item())


def plot_cos_curves(
    ts: List[float],
    cos_a: List[float],
    cos_b: List[float],
    outpath: Path,
    title: str,
) -> None:

    plt.figure(figsize=(7, 5))
    plt.plot(ts, cos_a, label="cos(recon(t), emb_A)")
    plt.plot(ts, cos_b, label="cos(recon(t), emb_B)")
    plt.xlabel("t (mu interpolation)")
    plt.ylabel("cosine similarity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def sanitize_one_line(s: str, max_chars: int) -> str:
    return s.replace("\n", " ").replace("\r", " ").strip()[:max_chars]


def write_pair_csv(
    outpath: Path,
    pair_id: int,
    lab_a: int,
    lab_b: int,
    text_a: str,
    text_b: str,
    rows: List[List[object]],
) -> None:
    """

    File layout:

      1) pair metadata header row
      2) pair metadata values row
      3) blank row
      4) table header row
      5) table rows: t, ca, cb, rank, nn_cos, nn_text

    """

    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "label_A", "label_B", "text_A", "text_B"])
        w.writerow(
            [
                pair_id,
                lab_a,
                lab_b,
                sanitize_one_line(text_a, 600),
                sanitize_one_line(text_b, 600),
            ]
        )

        w.writerow([])  
        w.writerow(["t", "cos_to_A", "cos_to_B", "nn_rank", "nn_cos", "nn_text"])

        for r in rows:
            w.writerow(r)

def beta_tag(beta: float) -> str:
   return str(beta).replace(".", "p")

def append_geometry_master(outpath: Path, rows: List[List[object]]) -> None:
   """
   Appends per-pair geometry metrics for one run.
   rows format:
     [latent_dim, beta, pair_id, label_A, label_B, path_length, curvature_mean, cos_start, cos_end, KL_loss]
   """
   outpath.parent.mkdir(parents=True, exist_ok=True)
   header = [
       "latent_dim",
       "beta",
       "pair_id",
       "label_A",
       "label_B",
       "path_length",
       "curvature_mean",
       "cos_start",
       "cos_end",
       "KL_loss"
   ]
   file_exists = outpath.exists()
   with outpath.open("a", newline="", encoding="utf-8") as f:
       w = csv.writer(f)
       if not file_exists:
           w.writerow(header)
       w.writerows(rows)

def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="artifacts")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--corpus_size", type=int, default=2000)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--steps", type=int, default=11)  # points between 0..1 inclusive
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--beta", type=float, required=True)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load sweet spot + model ---

    # sweet = load_sweet_spot(args.sweet_spot_json)
    # tokenizer = AutoTokenizer.from_pretrained(sweet.model_name)
    # model = load_model_from_ckpt(sweet.ckpt_path, device=device)

    ckpt_path, model_kl = find_ckpt_for_run(args.runs_root, args.latent_dim, args.beta)
    model = load_model_from_ckpt(ckpt_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)

    print("beta", model.hparams.beta)
    print("beta warmup", model.hparams.beta_warmup_epochs)
    

    # --- Load dataset (validation/test split) ---

    ds = load_dataset("ag_news")["test"]
    ds_corpus = ds.select(range(min(args.corpus_size, len(ds))))
    corpus_texts: List[str] = list(ds_corpus["text"])

    corpus_embs = embed_texts(
        model=model,
        tokenizer=tokenizer,
        texts=corpus_texts,
        max_length=args.max_length,
        device=device,
        batch_size=args.batch_size,
    )  # (N, D) normalized CPU tensor

    # --- Sample A/B pairs from different labels ---

    pairs = sample_pairs_different_labels(ds, num_pairs=args.pairs, seed=args.seed)

    if args.steps < 2:
        raise ValueError("--steps must be >= 2")

    ts = [i / (args.steps - 1) for i in range(args.steps)]

    run_prefix=f"Id{args.latent_dim}_{beta_tag(args.beta)}"
    for pidx, (a, b) in enumerate(pairs, start=1):
        text_a = a["text"]
        text_b = b["text"]
        lab_a = int(a["label"])
        lab_b = int(b["label"])

        # Encode endpoints

        emb_a, mu_a = encode_to_emb_and_mu(
            model=model,
            tokenizer=tokenizer,
            text=text_a,
            max_length=args.max_length,
            device=device,
        )

        emb_b, mu_b = encode_to_emb_and_mu(
            model=model,
            tokenizer=tokenizer,
            text=text_b,
            max_length=args.max_length,
            device=device,
        )



        cos_a_list: List[float] = []
        cos_b_list: List[float] = []
        csv_rows: List[List[object]] = []

        # Interpolate

        E = []  # list of (D,) torch tensors on CPU

        for t in ts:
            mu_t = (1.0 - t) * mu_a + t * mu_b
            recon_t = decode_from_mu(model=model, mu=mu_t, device=device)
            E.append(recon_t)
            c_a = cosine(recon_t, emb_a)
            c_b = cosine(recon_t, emb_b)
            cos_a_list.append(c_a)
            cos_b_list.append(c_b)


            # print(cosine(E[0],emb_a))
            # print(cosine(E[-1],emb_b))
            # print(cosine(emb_a,emb_b))


            # Retrieval-based "readable decoding"

            nns = topk_nearest_texts(
                query_emb=recon_t,
                corpus_embs=corpus_embs,
                corpus_texts=corpus_texts,
                k=args.topk,
            )

            for rank, (_, nn_cos, nn_text) in enumerate(nns, start=1):
                csv_rows.append(
                    [
                        t,
                        c_a,
                        c_b,
                        rank,
                        nn_cos,
                        sanitize_one_line(nn_text, 320),
                    ]
                )
        cos_start = cosine(E[0], emb_a)
        cos_end = cosine(E[-1], emb_b)
        # Path length: sum ||E[i+1]-E[i]||
        path_len = 0.0
        for i in range(len(E) - 1):
            path_len += float(torch.norm(E[i + 1] - E[i]).item())
        # Curvature: mean ||E[i+1] - 2E[i] + E[i-1]||
        curvs = []
        for i in range(1, len(E) - 1):
            c = torch.norm(E[i + 1] - 2 * E[i] + E[i - 1]).item()
            curvs.append(float(c))
        curv_mean = sum(curvs) / max(1, len(curvs))

        deltas=[torch.norm(E[i+1]-E[i]).item() for i in range(len(E)-1)]
        print(len(E))
        print(min(deltas), sum(deltas)/len(deltas), max(deltas))
        print(sum(deltas))

        print(mu_a.norm().item())
        print(mu_b.norm().item())
        print((mu_a-mu_b).norm().item())
        print(cos_start)
        print(cos_end)

        # Write per-pair CSV

        pair_csv = outdir / f"{run_prefix}_pair{pidx:02d}.csv"

        write_pair_csv(
            outpath=pair_csv,
            pair_id=pidx,
            lab_a=lab_a,
            lab_b=lab_b,
            text_a=text_a,
            text_b=text_b,
            rows=csv_rows,
        )

        # Plot curves

        plot_path = outdir / f"{run_prefix}_pair{pidx:02d}.png"
        title = f"Latent interpolation (pair {pidx}): label {lab_a} → {lab_b}"
        plot_cos_curves(ts, cos_a_list, cos_b_list, plot_path, title)
        geom_csv=outdir / f"phase2_1_geometry_summary.csv"
        geom_rows:list[list[object]]=[]
        geom_rows.append([args.latent_dim, args.beta, pidx, lab_a, lab_b, path_len, curv_mean, cos_start, cos_end, model_kl])
        append_geometry_master(
            geom_csv,geom_rows
        )




if __name__ == "__main__":
    main()
