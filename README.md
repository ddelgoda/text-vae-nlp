# Text VAE for Sentence Embeddings

This repository explores a **Variational Autoencoder (VAE)** trained on sentence embeddings and studies the geometry of its latent space.

The project is organised into two main phases:

- **Phase 1 — Model Selection & Pareto Analysis**
- **Phase 2 — Latent Interpolation Geometry**

## Project Structure

This repository explores the geometry of a Variational Autoencoder (VAE) trained on sentence embeddings.
The work is organised into two experimental phases that use different datasets for different purposes.

### Phase 1 — Model Selection (AG News)

Phase 1 performs hyperparameter sweeps on the **AG News** dataset to study the trade-off between:

- reconstruction accuracy
- latent compression (KL divergence)

These experiments sweep the reconstruction–compression trade-off under a frozen encoder. The result is negative: an active-units check found 0 of 1860 latent units active across all 75 runs, so the frozen sweep is posterior-collapsed throughout.

**Correction:** a pooling bug in `eval.py` had folded 2 unfrozen runs from later work into the Pareto front. With the condition filter fixed, the corrected knee is `ld=4, β=0.1` at `kl_min=0.001` — the least-collapsed point in the grid, not a working VAE configuration. See `reports/00_phase1_pareto.md` for full detail.

### Phase 2 — Latent Geometry Analysis (STS-B)

Phase 2 probes the **geometry of the learned latent space** using sentence pairs from the **STS-B (Semantic Textual Similarity Benchmark)** dataset.

STS-B provides interpretable sentence pairs that are well suited for **latent interpolation experiments**, allowing the trajectory between two sentences to be analysed in embedding space.


---

## Branch Structure

To keep the main codebase stable, experimental work for Phase 2 is maintained in a separate branch.

- **`main`**
Stable implementation and results for Phase 1 (AG News Pareto analysis).

- **`phase2-latent-interp`**
Experimental work exploring latent interpolation and geometry analysis using STS-B.


Full project documentation is available here:

➡ **[Project Overview](reports/README.md)**
➡ **[Phase 1 – Pareto Analysis](reports/00_phase1_pareto.md)**
➡ **[Phase 2 – Latent Geometry](reports/01_phase2_latent.md)**

---

## Repository Structure

```
 text-vae-nlp/
│
├── src/textvae/
│   ├── model.py          # Transformer + VAE
│   ├── lit_module.py     # Lightning training logic
│   └── __init__.py
│
├── scripts/
│   ├── train.py              # Training entry point
│   ├── eval.py               # Pareto analysis + plotting
|   └── interp.py             # Latent inperpolation
│
├── artifacts/
│   ├── pareto_plot.png
│   ├── pareto_front.csv
│   └── sweet_spot.json
│
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Key Result

Both training recipes collapsed. Neither produced a latent space carrying information about the input: 0 of 1860 active units across the 75-run frozen sweep, and 0 of 32 on the unfrozen checkpoint used for interpolation.

The notable part is how the second one hid it. Free bits raised raw KL above every Phase 1 run while active units fell to zero — the KL floor was satisfied by shrinking posterior variance rather than by making the posterior mean depend on the input. KL divergence without information transfer, from the mitigation intended to prevent exactly that.

The earlier claim of **smooth semantic trajectories** and **curved embedding manifolds** described a near-constant decoder. See `reports/00_phase1_pareto.md` and `reports/01_phase2_latent.md`.

## Disclaimer

A personal learning project — I built it to understand VAEs by working through one end to end, since passive reading doesn't produce real understanding.

Written collaboratively with AI tools throughout: ChatGPT and the Codex fork for initial development, Claude and Claude Code for later analysis and correction. My role was direction, questioning, and cross-checking. The Phase 1 evaluation bug and the posterior-collapse finding both came out of that process.

Corrections are appended rather than edited in, so the original analysis and what was wrong with it stay visible.

Experiments use public datasets and open-source tools, independently of my employer.
