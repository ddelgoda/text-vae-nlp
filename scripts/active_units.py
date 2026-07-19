# ------------------------------------------------------------
# active_units.py
#
# Active Units (AU) diagnostic for the frozen Phase 1 sweep.
#
# For each latent dimension i, AU is defined (Burda et al., 2015) as:
#     A_i = Var_x( E_q[z_i | x] )   i.e. the variance, across the
#     validation set, of the posterior mean mu_i.
# A unit counts as "active" if A_i > au_threshold. A model whose
# active-unit count stays far below latent_dim (and flat as latent_dim
# grows) is evidence of posterior collapse independent of the raw KL
# number, since KL can be small while individual units still carry
# some signal, or vice versa.
#
# Only runs with freeze_transformer == True are considered here: this
# is the frozen 75-run sweep. Later unfrozen runs use a different
# training recipe (beta warmup, free bits) and are not comparable.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from textvae.lit_module import LitTextVAE  # noqa: E402


def load_eval_texts(dataset: str, limit_val: int) -> tuple[List[str], str]:
    """
    Returns (texts, tag) for the requested diagnostic dataset.

    "ag_news" uses the same split (ag_news/test) and raw text field that
    scripts/train.py validates on for the Phase 1 sweep, so the active-units
    check runs on in-domain data. "stsb" is the original cross-domain
    diagnostic (stsb_multi_mt/test, sentence1, whitespace-normalized).
    """
    if dataset == "ag_news":
        ds = load_dataset("ag_news")["test"]
        ds = ds.select(range(min(limit_val, len(ds))))
        texts = list(ds["text"])
        return texts, "agnews"
    elif dataset == "stsb":
        ds = load_dataset("stsb_multi_mt", name="en")["test"]
        ds = ds.select(range(min(limit_val, len(ds))))
        texts = [" ".join(s.split()) for s in ds["sentence1"]]
        return texts, "stsb"
    raise ValueError(f"Unknown --dataset {dataset!r} (expected 'ag_news' or 'stsb')")


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> LitTextVAE:
    model = LitTextVAE.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    if getattr(model.hparams, "freeze_transformer", False):
        model.vae.encoder.model.eval()
    return model


@torch.no_grad()
def compute_mu_matrix(
    model: LitTextVAE,
    tokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Returns (N, Z) matrix of posterior means over the validation set."""
    mus = []
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
        out = model.vae(input_ids=input_ids, attention_mask=attention_mask)
        mus.append(out.mu.detach().cpu())
    return torch.cat(mus, dim=0)


def active_units_for_run(
    ckpt_path: str,
    texts: List[str],
    tokenizer,
    max_length: int,
    device: torch.device,
    batch_size: int,
    au_threshold: float,
) -> dict:
    model = load_model_from_ckpt(ckpt_path, device)
    mu = compute_mu_matrix(model, tokenizer, texts, max_length, device, batch_size)
    # Var_x(E_q[z_i|x]) per dimension, ddof=0 (population variance)
    dim_var = mu.var(dim=0, unbiased=False)
    n_active = int((dim_var > au_threshold).sum().item())
    latent_dim = mu.shape[1]
    dim_var_list = dim_var.tolist()
    del model
    return {
        "latent_dim": latent_dim,
        "n_active": n_active,
        "active_ratio": n_active / latent_dim,
        "dim_var": dim_var_list,
        "mean_dim_var": float(dim_var.mean().item()),
        "max_dim_var": float(dim_var.max().item()),
        "min_dim_var": float(dim_var.min().item()),
    }


def plot_active_vs_latent(df: pd.DataFrame, outdir: Path, tag: str) -> None:
    plt.figure(figsize=(7, 5))
    for beta, g in df.groupby("beta"):
        g = g.sort_values("latent_dim")
        plt.plot(g["latent_dim"], g["n_active"], marker="o", label=f"beta={beta}")
    max_ld = df["latent_dim"].max()
    plt.plot(
        [df["latent_dim"].min(), max_ld],
        [df["latent_dim"].min(), max_ld],
        linestyle="--",
        color="gray",
        alpha=0.6,
        label="n_active = latent_dim",
    )
    plt.xlabel("latent_dim")
    plt.ylabel("active units (count)")
    plt.title(f"Active units vs latent capacity (frozen sweep, {tag})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"active_units_vs_latent_dim_{tag}.png", dpi=180)
    plt.close()


def plot_heatmap(df: pd.DataFrame, outdir: Path, tag: str) -> None:
    pivot = (
        df.pivot_table(
            index="latent_dim", columns="beta", values="active_ratio", aggfunc="mean"
        )
        .sort_index()
        .sort_index(axis=1)
    )
    plt.figure(figsize=(9, 6))
    im = plt.imshow(pivot.values, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, label="active_ratio (n_active / latent_dim)")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), [str(r) for r in pivot.index])
    plt.xlabel("beta")
    plt.ylabel("latent_dim")
    plt.title(f"Active-unit ratio (mean per grid cell, frozen sweep, {tag})")
    plt.tight_layout()
    plt.savefig(outdir / f"heatmap_active_units_{tag}.png", dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--outdir", type=str, default="artifacts")
    ap.add_argument("--pattern", type=str, default="metrics_summary.json")
    ap.add_argument("--au_threshold", type=float, default=0.01)
    ap.add_argument("--limit_val", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument(
        "--dataset",
        choices=["ag_news", "stsb"],
        default="ag_news",
        help=(
            "Diagnostic corpus. 'ag_news' (default) uses the same "
            "ag_news/test split the Phase 1 sweep validated on, in-domain "
            "with training. 'stsb' is the original cross-domain diagnostic "
            "(stsb_multi_mt/test). Output filenames are tagged by dataset "
            "so both result sets can coexist."
        ),
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = sorted(runs_root.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No {args.pattern} found under {runs_root}")

    rows = []
    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        # Condition filter: only the frozen sweep is comparable here.
        if not d.get("freeze_transformer", False):
            continue
        ckpt = d.get("best_model_path")
        if not ckpt or not Path(ckpt).exists():
            print(f"Skipping {d.get('run_id')}: no usable checkpoint")
            continue
        rows.append(d)

    if not rows:
        raise RuntimeError("No frozen runs with usable checkpoints found.")

    print(f"Found {len(rows)} frozen runs (of {len(files)} total metrics files).")

    # Shared validation set + tokenizer across all runs (same encoder, same split).
    model_name = rows[0]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts, tag = load_eval_texts(args.dataset, args.limit_val)
    print(f"Dataset: {args.dataset} ({len(texts)} texts, tag={tag})")

    results = []
    for i, d in enumerate(rows, start=1):
        run_id = d["run_id"]
        ckpt = d["best_model_path"]
        print(f"[{i}/{len(rows)}] {run_id}")
        au = active_units_for_run(
            ckpt_path=ckpt,
            texts=texts,
            tokenizer=tokenizer,
            max_length=args.max_length,
            device=device,
            batch_size=args.batch_size,
            au_threshold=args.au_threshold,
        )
        results.append(
            {
                "run_id": run_id,
                "latent_dim": d["latent_dim"],
                "beta": d["beta"],
                "kl_loss": d.get("kl_loss"),
                "recon_loss": d.get("recon_loss"),
                **au,
            }
        )

    # Persist raw per-dimension variances (not just min/max/mean) so the
    # active-unit threshold can be swept/audited without rerunning inference.
    dim_var_by_run = {r["run_id"]: r.pop("dim_var") for r in results}
    (outdir / f"active_units_dim_var_{tag}.json").write_text(
        json.dumps(dim_var_by_run, indent=2), encoding="utf-8"
    )

    df = pd.DataFrame(results)
    df.to_csv(outdir / f"active_units_per_run_{tag}.csv", index=False)

    # Threshold sensitivity: recompute n_active at several thresholds from
    # the raw per-dimension variances to check the au_threshold=0.01 choice
    # isn't hiding a borderline result.
    sweep_thresholds = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    sweep_rows = []
    for t in sweep_thresholds:
        total_active = 0
        total_dims = 0
        max_active_in_a_run = 0
        for dv in dim_var_by_run.values():
            n = sum(1 for v in dv if v > t)
            total_active += n
            total_dims += len(dv)
            max_active_in_a_run = max(max_active_in_a_run, n)
        sweep_rows.append(
            {
                "threshold": t,
                "total_active_units": total_active,
                "total_units": total_dims,
                "active_frac": total_active / total_dims,
                "max_active_in_any_run": max_active_in_a_run,
            }
        )
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(outdir / f"active_units_threshold_sweep_{tag}.csv", index=False)

    summary = (
        df.groupby(["latent_dim", "beta"])
        .agg(
            n_active_mean=("n_active", "mean"),
            active_ratio_mean=("active_ratio", "mean"),
            kl_loss_mean=("kl_loss", "mean"),
            recon_loss_mean=("recon_loss", "mean"),
            n_seeds=("run_id", "count"),
        )
        .reset_index()
        .sort_values(["latent_dim", "beta"])
    )
    summary.to_csv(outdir / f"active_units_summary_{tag}.csv", index=False)

    plot_active_vs_latent(df, outdir, tag)
    plot_heatmap(df, outdir, tag)

    print("\nWrote:")
    print(outdir / f"active_units_per_run_{tag}.csv")
    print(outdir / f"active_units_summary_{tag}.csv")
    print(outdir / f"active_units_dim_var_{tag}.json")
    print(outdir / f"active_units_threshold_sweep_{tag}.csv")
    print(outdir / f"active_units_vs_latent_dim_{tag}.png")
    print(outdir / f"heatmap_active_units_{tag}.png")

    print("\nSummary (mean active units / latent_dim):")
    print(summary.to_string(index=False))

    print(f"\nThreshold sensitivity (au_threshold={args.au_threshold} used above; sweep for audit):")
    print(sweep_df.to_string(index=False))


if __name__ == "__main__":
    main()
