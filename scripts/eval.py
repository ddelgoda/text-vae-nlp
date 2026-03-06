from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
def pareto_front_min(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    sdf = df.sort_values([x, y], ascending=[True, True]).reset_index(drop=True)
    keep = []
    best_y = float("inf")
    for _, r in sdf.iterrows():
        if float(r[y]) < best_y - 1e-12:
            keep.append(r)
            best_y = float(r[y])
    return pd.DataFrame(keep).reset_index(drop=True)


def normalize_01(s: pd.Series) -> pd.Series:
    mn = float(s.min())
    mx = float(s.max())
    if math.isclose(mn, mx):
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def knee_point(front: pd.DataFrame, x: str, y: str) -> Tuple[int, float]:
    f = front.copy()
    f["_x"] = normalize_01(f[x])
    f["_y"] = normalize_01(f[y])
    f = f.sort_values("_x").reset_index(drop=True)
    x1, y1 = float(f.loc[0, "_x"]), float(f.loc[0, "_y"])
    x2, y2 = float(f.loc[len(f) - 1, "_x"]), float(f.loc[len(f) - 1, "_y"])
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    denom = math.sqrt(a * a + b * b) + 1e-12
    best_i, best_d = 0, -1.0
    for i in range(len(f)):
        xi, yi = float(f.loc[i, "_x"]), float(f.loc[i, "_y"])
        d = abs(a * xi + b * yi + c) / denom
        if d > best_d:
            best_i, best_d = i, d
    return best_i, best_d


def plot_pareto(df, front, sweet, outdir, kl_min: float):

    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))

    # Sweet spot
    plt.scatter(
        [sweet["kl_loss"]],
        [sweet["recon_loss"]],
        s=200,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        label="Selected sweet spot",
    )

    # All runs
    if "latent_dim" in df.columns:
        sc = plt.scatter(
            df["kl_loss"],
            df["recon_loss"],
            c=df["latent_dim"],
            alpha=0.5,
        )
        plt.colorbar(sc, label="latent_dim")
    else:
        plt.scatter(df["kl_loss"], df["recon_loss"], alpha=0.4, label="All runs")

    # Pareto front
    plt.plot(
        front["kl_loss"],
        front["recon_loss"],
        linewidth=2,
        label="Pareto front",
    )

    # Annotate sweet spot with key hyperparams
    label_bits = []
    for k in ["latent_dim", "beta"]:
        if k in sweet.index:
            label_bits.append(f"{k}={sweet[k]}")
    if label_bits:
        plt.annotate(
            ", ".join(label_bits),
            (sweet["kl_loss"], sweet["recon_loss"]),
            textcoords="offset points",
            xytext=(10, -10),
            fontsize=9,
        )

    plt.savefig(outdir / "pareto_plot_annotated.png", dpi=180)

    # Collapse threshold line
    plt.axvline(
        x=kl_min,
        linestyle="--",
        alpha=0.5,
        label="KL collapse threshold",
    )

    plt.xlabel("KL divergence")
    plt.ylabel("Reconstruction loss")
    plt.title("KL vs Reconstruction (Filtered Pareto Front)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pareto_plot.png", dpi=180)
    plt.close()


def plot_heatmaps(df: pd.DataFrame, outdir: Path) -> None:

    required = {"latent_dim", "beta", "recon_loss", "kl_loss"}
    if not required.issubset(df.columns):
        print(f"Skipping heatmaps: missing columns {required - set(df.columns)}")
        return

    # Aggregate in case multiple runs exist per cell (e.g., different timestamps)
    pivot_recon = (
        df.pivot_table(
            index="latent_dim",
            columns="beta",
            values="recon_loss",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    pivot_kl = (
        df.pivot_table(
            index="latent_dim",
            columns="beta",
            values="kl_loss",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    # Helper to plot one heatmap
    def _plot(pivot: pd.DataFrame, title: str, cbar_label: str, outname: str) -> None:
        plt.figure(figsize=(9, 6))
        im = plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
        plt.colorbar(im, label=cbar_label)

        plt.xticks(
            range(len(pivot.columns)),
            [str(x) for x in pivot.columns],
            rotation=45,
            ha="right",
        )
        plt.yticks(range(len(pivot.index)), [str(y) for y in pivot.index])

        plt.xlabel("beta")
        plt.ylabel("latent_dim")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outdir / outname, dpi=180)
        plt.close()

    _plot(
        pivot_recon,
        title="Reconstruction Loss Heatmap (mean per grid cell)",
        cbar_label="recon_loss",
        outname="heatmap_recon.png",
    )

    _plot(
        pivot_kl,
        title="KL Loss Heatmap (mean per grid cell)",
        cbar_label="kl_loss",
        outname="heatmap_kl.png",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--outdir", type=str, default="artifacts")
    ap.add_argument("--pattern", type=str, default="metrics_summary.json")
    ap.add_argument("--kl_min", type=float, default=0.005)
    args = ap.parse_args()
    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(runs_root.rglob(args.pattern))
    kl_min = args.kl_min
    if not files:
        raise FileNotFoundError(f"No {args.pattern} found under {runs_root}")
    rows: List[dict] = []
    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        # Require key metrics
        if "recon_loss" not in d or "kl_loss" not in d:
            continue
        rows.append(d)
    if not rows:
        raise RuntimeError(
            "No usable metrics_summary.json files found (missing recon_loss/kl_loss)."
        )
    raw_df = pd.DataFrame(rows)
    raw_df["recon_loss"] = pd.to_numeric(raw_df["recon_loss"])
    raw_df["kl_loss"] = pd.to_numeric(raw_df["kl_loss"])
    # Save raw grid
    raw_df.to_csv(outdir / "grid_all_raw.csv", index=False)
    # Filter only for Pareto
    df = raw_df[raw_df["kl_loss"] > kl_min].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No runs left after filtering with kl_min={kl_min}")
    df.to_csv(outdir / "pareto_all.csv", index=False)

    front = pareto_front_min(df, x="kl_loss", y="recon_loss")
    front.to_csv(outdir / "pareto_front.csv", index=False)
    kidx, kdist = knee_point(front, x="kl_loss", y="recon_loss")
    sweet = front.iloc[kidx]
    sweet_out = {
        "sweet_spot": sweet.to_dict(),
        "selection_method": "Pareto knee (KL-filtered)",
        "kl_min_threshold": kl_min,
        "rationale": "Models with near-zero KL were excluded to avoid posterior collapse.",
    }
    (outdir / "sweet_spot.json").write_text(
        json.dumps(sweet_out, indent=2), encoding="utf-8"
    )
    pd.DataFrame([sweet.to_dict()]).to_csv(outdir / "sweet_spot.csv", index=False)

    plot_pareto(df, front, sweet, outdir, kl_min)
    plot_heatmaps(raw_df, outdir)
    print("Wrote:")
    print(outdir / "pareto_all.csv")
    print(outdir / "pareto_front.csv")
    print(outdir / "sweet_spot.json")
    print(outdir / "pareto_plot.png")
    print("\nSweet spot:")
    print(json.dumps(sweet_out, indent=2))


if __name__ == "__main__":
    main()
