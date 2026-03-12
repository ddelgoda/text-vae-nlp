# Phase 2 — Latent Interpolation Analysis

---

## Objective

Evaluate whether the trained TextEmbeddingVAE latent space forms a meaningful geometric structure.

Core question:

> Does linear interpolation in latent space produce smooth, interpretable trajectories in embedding space?

---

## Reasoning & Hypotheses

If the latent space is structured, interpolation between two latent endpoints should produce:

- smooth cosine similarity transitions
- continuous movement through embedding space
- gradual nearest-neighbour changes rather than abrupt jumps

Posterior collapse would make these trajectories degenerate or uninformative, so collapse mitigation is required before interpolation analysis.

---

## Experimental Setup

### Model and Training Stability

The interpolation study uses the VAE setup selected from Phase 1.

To stabilise latent usage during training:

- **β warm-up** is applied so regularisation ramps up gradually
- **free bits** (`--kl_free_bits`) are used to discourage near-zero KL per latent dimension

These mechanisms reduce collapse risk and preserve informative latent structure for geometric analysis.

### Data Context

- Interpolation evaluation uses sentence-pair style analysis (STS-B style semantic comparison).
- The selected training configuration currently matches the Phase 1 sweet spot manually.

---

## Training

Example training command:

```bash
uv run python scripts/train.py \
  --latent_dim 32 \
  --beta 0.1 \
  --beta_warmup_epochs 5 \
  --num_workers 0 \
  --kl_free_bits 0.1 \
  --max_length 128
```

---

## Evaluation

Run interpolation analysis on a chosen run:

```bash
uv run python scripts/interp.py \
  --min_len 10 \
  --run_dir runs/ld32_b0.1_s42_20260311_001705 \
  --sim_min 0.0 \
  --sim_max 1.5 \
  --num_pairs 3 \
  --corpus_size 2000 \
  --topk 5 \
  --beta 0.1
```

Pipeline overview:

```mermaid
graph TD
A[Sentence A] --> B[Transformer Encoder]
C[Sentence B] --> B
B --> D[Latent z_A and z_B]
D --> E[Linear Interpolation z(t)]
E --> F[Decoder to Embedding Space]
F --> G[Cosine / Path / Curvature Metrics]
G --> H[Nearest-neighbour semantic tracking]
```

---

## Results

Typical observations:

- cosine similarity evolves smoothly between interpolation endpoints
- latent-space paths are often longer than direct embedding-linear baselines
- curvature ratios indicate nonlinear but continuous trajectories

These patterns are consistent with a smooth latent coordinate system over the sentence-embedding manifold.

---

## Key Findings

- The learned latent space supports coherent interpolation behaviour.
- Collapse mitigation (β warm-up + free bits) is important for stable geometry.
- Phase 1 model selection transfers effectively into Phase 2 geometric validation.

---

## Limitations and Next Steps

- Phase 1 → Phase 2 configuration handoff is currently manual.
- Future work should auto-load the selected sweet spot from Pareto outputs.
- Additional sweeps over `β` and `latent_dim` can map how geometry changes with regularisation and capacity.
