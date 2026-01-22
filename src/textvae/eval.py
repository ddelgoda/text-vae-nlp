from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


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


def plot_pareto(df, front, sweet, outdir):

    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))

    # All runs
    plt.scatter(
        df["kl_loss"],
        df["recon_loss"],
        alpha=0.4,
        label="All runs",
    )

    # Pareto front
    plt.plot(
        front["kl_loss"],
        front["recon_loss"],
        linewidth=2,
        label="Pareto front",
    )

    # Sweet spot
    plt.scatter(
        [sweet["kl_loss"]],
        [sweet["recon_loss"]],
        marker="X",
        s=120,
        label="Selected sweet spot",
    )

    # Collapse threshold line
    plt.axvline(
        x=0.05,
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
    df = pd.DataFrame(rows)
    df["recon_loss"] = pd.to_numeric(df["recon_loss"])
    df["kl_loss"] = pd.to_numeric(df["kl_loss"])
    df = df[df["kl_loss"] > kl_min].reset_index(drop=True)  # avoids posterior collapse
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
    plot_pareto(df, front, sweet, outdir)
    print("Wrote:")
    print(outdir / "pareto_all.csv")
    print(outdir / "pareto_front.csv")
    print(outdir / "sweet_spot.json")
    print(outdir / "pareto_plot.png")
    print(outdir / "pareto_plot_annotated.png")
    print("\nSweet spot:")
    print(json.dumps(sweet_out, indent=2))


if __name__ == "__main__":
    main()
