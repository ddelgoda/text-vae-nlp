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


def plot_active_vs_latent(df: pd.DataFrame, outdir: Path) -> None:
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
    plt.title("Active units vs latent capacity (frozen sweep)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "active_units_vs_latent_dim.png", dpi=180)
    plt.close()


def plot_heatmap(df: pd.DataFrame, outdir: Path) -> None:
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
    plt.title("Active-unit ratio (mean per grid cell, frozen sweep)")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_active_units.png", dpi=180)
    plt.close()


def run_single_checkpoint(args, device):
    """
    One-off AU check against a single run directory, bypassing the
    frozen-only grid path below. For checkpoints trained under a
    different recipe (e.g. unfrozen encoder, beta warmup, free bits)
    that aren't part of the frozen Phase 1 sweep and can't be pooled
    into its grid summary/plots.
    """
    run_dir = Path(args.run_dir)
    ms_path = run_dir / "metrics_summary.json"
    d = json.loads(ms_path.read_text(encoding="utf-8"))

    ckpt = d.get("best_model_path") or d.get("last_model_path")
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(f"No usable checkpoint recorded in {ms_path}")
    ckpt_kind = "best" if d.get("best_model_path") else "last"
    print(f"Run: {d['run_id']}")
    print(f"Checkpoint: {ckpt} ({ckpt_kind}.ckpt)")
    print(f"freeze_transformer={d.get('freeze_transformer')}, "
          f"kl_free_bits={d.get('kl_free_bits')}, "
          f"beta_warmup_epochs={d.get('beta_warmup_epochs')}")

    tokenizer = AutoTokenizer.from_pretrained(d["model_name"])
    ds = load_dataset("stsb_multi_mt", name="en")[args.split]
    n = min(args.limit_val, len(ds))
    ds = ds.select(range(n))
    texts = [" ".join(s.split()) for s in ds["sentence1"]]
    print(f"Data: stsb_multi_mt (en), split='{args.split}', n={len(texts)} sentences, "
          f"max_length={args.max_length}")

    au = active_units_for_run(
        ckpt_path=ckpt,
        texts=texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        device=device,
        batch_size=args.batch_size,
        au_threshold=args.au_threshold,
    )
    dim_var = au.pop("dim_var")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = {
        "run_id": d["run_id"],
        "ckpt_path": ckpt,
        "ckpt_kind": ckpt_kind,
        "split": args.split,
        "n_texts": len(texts),
        "max_length": args.max_length,
        "au_threshold": args.au_threshold,
        "latent_dim": d["latent_dim"],
        "beta": d["beta"],
        "kl_free_bits": d.get("kl_free_bits"),
        "kl_loss": d.get("kl_loss"),
        "kl_used": d.get("kl_used"),
        "recon_loss": d.get("recon_loss"),
        **au,
        "dim_var": dim_var,
    }
    out_path = outdir / f"active_units_{d['run_id']}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    sweep_thresholds = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    print(f"\nn_active (threshold={args.au_threshold}): {au['n_active']} / {au['latent_dim']}")
    print("\nThreshold sweep:")
    for t in sweep_thresholds:
        n = sum(1 for v in dim_var if v > t)
        print(f"  {t:>8}: {n:2d} / {au['latent_dim']} active")
    sorted_var = sorted(dim_var)
    print(f"\nSorted per-dimension variances ({len(sorted_var)} dims):")
    print(", ".join(f"{v:.5f}" for v in sorted_var))
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--run_dir", type=str, default=None,
                     help="If set, run a one-off AU check on this single run "
                          "directory instead of scanning runs_root for the "
                          "frozen grid.")
    ap.add_argument("--split", type=str, default="test",
                     help="stsb_multi_mt split to use as input texts "
                          "(only used with --run_dir).")
    ap.add_argument("--outdir", type=str, default="artifacts")
    ap.add_argument("--pattern", type=str, default="metrics_summary.json")
    ap.add_argument("--au_threshold", type=float, default=0.01)
    ap.add_argument("--limit_val", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run_dir:
        run_single_checkpoint(args, device)
        return

    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
    ds = load_dataset("stsb_multi_mt", name="en")["test"]
    ds = ds.select(range(min(args.limit_val, len(ds))))
    texts = [" ".join(s.split()) for s in ds["sentence1"]]

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
    (outdir / "active_units_dim_var.json").write_text(
        json.dumps(dim_var_by_run, indent=2), encoding="utf-8"
    )

    df = pd.DataFrame(results)
    df.to_csv(outdir / "active_units_per_run.csv", index=False)

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
    sweep_df.to_csv(outdir / "active_units_threshold_sweep.csv", index=False)

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
    summary.to_csv(outdir / "active_units_summary.csv", index=False)

    plot_active_vs_latent(df, outdir)
    plot_heatmap(df, outdir)

    print("\nWrote:")
    print(outdir / "active_units_per_run.csv")
    print(outdir / "active_units_summary.csv")
    print(outdir / "active_units_dim_var.json")
    print(outdir / "active_units_threshold_sweep.csv")
    print(outdir / "active_units_vs_latent_dim.png")
    print(outdir / "heatmap_active_units.png")

    print("\nSummary (mean active units / latent_dim):")
    print(summary.to_string(index=False))

    print(f"\nThreshold sensitivity (au_threshold={args.au_threshold} used above; sweep for audit):")
    print(sweep_df.to_string(index=False))


if __name__ == "__main__":
    main()
