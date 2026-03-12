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

### Dataset

Experiments use the **STS-B (Semantic Textual Similarity Benchmark)** dataset.

Two components are used:

| Role | Split |
|-----|------|
| interpolation sentence pairs | `dev` |
| retrieval corpus | `train` |

A corpus of approximately **2000 unique sentences** is built from the training split and embedded using the frozen transformer encoder.

### Method
- Interpolation evaluation uses sentence-pair style analysis (STS-B style semantic comparison).
- The selected training configuration currently matches the Phase 1 sweet spot manually.

#### Latent Interpolation

Given two sentences:

```
A → latent vector μ_A
B → latent vector μ_B
```

Interpolation is defined as:

```
μ_t = (1 - t) μ_A + t μ_B
```

for

```
t ∈ [0,1]
```

At each step:

1. Latent vectors are interpolated  
2. The decoder reconstructs an embedding  
3. Cosine similarity to both endpoints is computed

Pipeline overview:

```mermaid
graph TD

A[Sentence A] --> B[Transformer Encoder];
C[Sentence B] --> B;

B --> D[Sentence Embeddings];

D --> E[VAE Encoder];
E --> F[Latent Mean mu_A];
E --> G[Latent Mean mu_B];

F --> H[Latent Interpolation];
G --> H;

H --> I[Decoder];
I --> J[Reconstructed Embedding E_t];

J --> K[Cosine Similarity to A-B];
J --> L[Nearest Neighbor Retrieval];
J --> M[Geometry Metrics];
```


#### Evaluation Metrics

- Cosine Similarity Curves

For each interpolation point:

```
cos(E(t), emb_A)
cos(E(t), emb_B)
```

Expected behaviour:

- similarity to A decreases
- similarity to B increases

Smooth curves indicate coherent latent structure.

- Path Length

Measures total movement of the decoded path:

```
Σ ||E[i+1] − E[i]||
```

Compared with a baseline:

```
linear interpolation between embeddings
```

- Curvature

Approximates how much the trajectory bends:

```
||E[i+1] − 2E[i] + E[i−1]||
```

Higher curvature suggests the embedding manifold is nonlinear.

- Geometry Ratios

Latent interpolation is compared with embedding interpolation:

```
path_ratio = path_len_lat / path_len_emb
curv_ratio = curv_lat / curv_emb
```

Large ratios indicate the decoder maps latent straight lines into **curved embedding trajectories**.

- Semantic Storyline

To interpret interpolation behaviour, reconstructed embeddings are decoded using **nearest-neighbor retrieval** from the corpus.

Snapshots are taken at:

```
t = 0      (start)
t = 0.5    (middle)
t = 1      (end)
```

The nearest sentences provide a **semantic storyline** showing how the embedding region changes along the interpolation path.


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
