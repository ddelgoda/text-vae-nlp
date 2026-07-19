# Text VAE for Sentence Embeddings

This repository explores a **Variational Autoencoder (VAE)** trained on sentence embeddings and studies the geometry of its latent space.

The project is organised into two main phases:

- **Phase 1 — Model Selection & Pareto Analysis**
- **Phase 2 — Latent Interpolation Geometry**

## Project Structure

This repository explores the geometry of a Variational Autoencoder (VAE) trained on sentence embeddings.
The work is organised into two experimental phases that use different datasets for different purposes.

### Phase 1 — Hyperparameter Sweep and Collapse Diagnosis (AG News)

Phase 1 performs hyperparameter sweeps on the **AG News** dataset to study the trade-off between:

- reconstruction accuracy
- latent compression (KL divergence)

These experiments sweep the reconstruction–compression trade-off under a frozen encoder. The result is negative: an active-units check on held-out AG News found no active dimensions in any of the 75 runs, so the frozen sweep is posterior-collapsed throughout.

**Correction:** a pooling bug in `eval.py` had folded 2 unfrozen runs from later work into the Pareto front. With the condition filter fixed, the corrected knee is `ld=4, β=0.1` at `kl_min=0.001` — but with every run collapsed, both Pareto axes track `latent_dim` rather than a real trade-off, so the knee is not a preferred configuration. See `reports/00_phase1_pareto.md` for full detail.

### Phase 2 — Latent Geometry Analysis (STS-B)

Phase 2 probes the **geometry of the learned latent space** using sentence pairs from the **STS-B (Semantic Textual Similarity Benchmark)** dataset.

STS-B provides interpretable sentence pairs that are well suited for **latent interpolation experiments**, allowing the trajectory between two sentences to be analysed in embedding space.

Using separate datasets keeps:

- **Phase 1 focused on model selection**
- **Phase 2 focused on geometric evaluation**

**Correction:** an active-units check on the interpolation checkpoint found 0 of 32 units active at every threshold tested — no detectable input-dependent signal in the posterior means. See the Phase 2 report on [`feat/phase2-latent-interp`](https://github.com/ddelgoda/text-vae-nlp/blob/feat/phase2-latent-interp/reports/01_phase2_latent.md) for full detail.

---

## Branch Structure

To keep the main codebase stable, experimental work for Phase 2 is maintained in a separate branch.

- **`main`**
Stable implementation and results for Phase 1 (AG News Pareto analysis).

- **`feat/phase2-latent-interp`**
Experimental work exploring latent interpolation and geometry analysis using STS-B. The Phase 2 report and its correction exist only on this branch; links to it below are absolute so they resolve from either branch.

Full project documentation is available here:

➡ **[Project Overview](reports/README.md)**
➡ **[Phase 1 – Pareto Analysis](reports/00_phase1_pareto.md)**
➡ **[Phase 2 – Latent Geometry](https://github.com/ddelgoda/text-vae-nlp/blob/feat/phase2-latent-interp/reports/01_phase2_latent.md)** (on `feat/phase2-latent-interp`)

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
│   ├── active_units.py       # Active-units diagnostic
|   └── interp.py             # Latent interpolation (feat/phase2-latent-interp only)
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

Neither training recipe produced a latent space carrying detectable information about the input. The active-unit diagnostic (Burda et al. 2015) found no active dimensions in any of the 75 frozen Phase 1 runs, and none in the unfrozen Phase 2 checkpoint used for interpolation.

Phase 1 was checked in-domain on held-out AG News, the data those models were trained on. Per-dimension posterior-mean variance ranges 1.0e-4 to 4.5e-4 — roughly 20–100× below the 0.01 active threshold — and no dimension becomes active even at 0.0005, a 20× looser bar.

The earlier claim of **smooth semantic trajectories** and **curved embedding manifolds** described a near-constant decoder, not a learned manifold.

An earlier version of this section attributed the Phase 2 result to free bits inflating KL while the latent went dead. That explanation is withdrawn — it rested on a miscomputed KL value (see Known Issues). The collapse finding is unaffected: active units are computed directly from posterior means and do not touch the KL code path.

Scope: 1,000 training examples, 500 validation, 5 epochs. A finding about this setup, not about frozen-encoder VAEs in general.

See `reports/00_phase1_pareto.md` and the Phase 2 report on [`feat/phase2-latent-interp`](https://github.com/ddelgoda/text-vae-nlp/blob/feat/phase2-latent-interp/reports/01_phase2_latent.md).

## Known Issues

Found during review. Listed rather than left silently in place.

**KL computation in Phase 2 is wrong.** In `src/textvae/lit_module.py` on `feat/phase2-latent-interp`, `kl_diag_gaussian_per_dim` already sums over the latent dimension and returns shape `(B,)`. `_loss_terms` then unsqueezes to `(1, B)` and sums over `dim=1` — the batch axis. So logged `kl_raw` and `kl_used` are inflated by roughly the batch size (16), and the reported values (0.0305 and 0.0497) are not comparable to Phase 1's correctly-computed KL. The free-bits floor also clamps per-sample totals rather than per-dimension, so free bits never enforced the intended constraint. The Phase 2 checkpoint was trained against a malformed objective. Whether correctly-implemented free bits would prevent collapse is untested. Phase 1's KL, computed on `main`, is not affected.

**Smaller items**, each verified against the source:

- `--min_len` is declared at `scripts/interp.py:146` and never referenced again. `sample_pairs_sts_low_cos` defaults it to 1 (`interp_utils.py:182`), so length filtering ran at 1 regardless of the documented `--min_len 10`.
- `find_ckpt_for_run` (`interp_utils.py:346-355`) collects checkpoints keyed on `st_mtime`, sorts ascending, and takes `[0]` — the oldest file — directly below a comment at line 353 stating that the lowest `best_model_score` is best. The `reverse=True` sort inside the loop at 347 is immediately overridden and does nothing. Inert for single-run directories, wrong if pointed at a directory with several.
- The frozen encoder is set to eval mode once, in `on_fit_start` (`lit_module.py:151-155`). Lightning calls `.train()` on the module at each training epoch, which cascades to submodules and re-enables encoder dropout from epoch 2 onward. Gradients stay disabled, so no weights leak, but the "stable embeddings" invariant holds only for epoch 1.
- `pyproject.toml` declares `gradio`, `pyyaml`, `rich` and `torchmetrics`, none of which are imported anywhere. `pandas` is imported (`eval.py:11`, `active_units.py:29`) but not declared — reproducibility currently relies on it arriving transitively via `datasets`.
- `main.py` is an unwired stub that prints a greeting.
- `pareto_plot.png` and `pareto_plot_annotated.png` are now identical; one should be removed.

## Disclaimer

A personal learning project — I built it to understand VAEs by working through one end to end, since passive reading doesn't produce real understanding.

AI tools were used throughout, in two distinct phases. The original implementation was written with ChatGPT: I reviewed each piece of generated code manually, worked through the design decisions, and ran every experiment myself. Codex was used only to tidy the repository and comments. A later review round with Claude and Claude Code found the Pareto pooling bug, added the active-unit diagnostics, and surfaced several implementation errors that the first pass missed — those are documented in Known Issues rather than quietly fixed.

Corrections are appended rather than edited in, so the original analysis and what was wrong with it stay visible.

Experiments use public datasets and open-source tools, independently of my employer.
