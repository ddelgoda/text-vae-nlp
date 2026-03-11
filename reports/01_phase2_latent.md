# Phase 2 — Latent Interpolation Analysis

## Objective

This phase evaluates whether the **TextEmbeddingVAE latent space** forms a meaningful geometric structure.

The central question is:

> Does linear interpolation in latent space produce smooth trajectories in embedding space?

If the latent space is well-structured, interpolation should show:

- smooth cosine similarity transitions  
- continuous movement in embedding space  
- gradual shifts in nearest-neighbour sentences  

---

# Latent Interpolation Pipeline

```mermaid
flowchart LR

A[Sentence A] --> B[Transformer Encoder]
C[Sentence B] --> B

B --> D[Sentence Embeddings]

D --> E[VAE Encoder]
E --> F[Latent Mean μ_A]
E --> G[Latent Mean μ_B]

F --> H[Latent Interpolation]
G --> H

H --> I[Decoder]
I --> J[Reconstructed Embedding E(t)]

J --> K[Cosine Similarity to A/B]
J --> L[Nearest Neighbor Retrieval]
J --> M[Geometry Metrics]
```

---

# Dataset

Experiments use the **STS-B (Semantic Textual Similarity Benchmark)** dataset.

Two components are used:

| Role | Split |
|-----|------|
| interpolation sentence pairs | `dev` |
| retrieval corpus | `train` |

A corpus of approximately **2000 unique sentences** is built from the training split and embedded using the frozen transformer encoder.

This corpus allows decoding latent points into **human-readable nearest neighbours**.

---

# Latent Interpolation

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

---

# Evaluation Metrics

## Cosine Similarity Curves

For each interpolation point:

```
cos(E(t), emb_A)
cos(E(t), emb_B)
```

Expected behaviour:

- similarity to A decreases
- similarity to B increases

Smooth curves indicate coherent latent structure.

---

## Path Length

Measures total movement of the decoded path:

```
Σ ||E[i+1] − E[i]||
```

Compared with a baseline:

```
linear interpolation between embeddings
```

---

## Curvature

Approximates how much the trajectory bends:

```
||E[i+1] − 2E[i] + E[i−1]||
```

Higher curvature suggests the embedding manifold is nonlinear.

---

## Geometry Ratios

Latent interpolation is compared with embedding interpolation:

```
path_ratio = path_len_lat / path_len_emb
curv_ratio = curv_lat / curv_emb
```

Large ratios indicate the decoder maps latent straight lines into **curved embedding trajectories**.

---

# Semantic Storyline

To interpret interpolation behaviour, reconstructed embeddings are decoded using **nearest-neighbor retrieval** from the corpus.

Snapshots are taken at:

```
t = 0      (start)
t = 0.5    (middle)
t = 1      (end)
```

The nearest sentences provide a **semantic storyline** showing how the embedding region changes along the interpolation path.

---

# Results

Typical observations:

- cosine similarity transitions smoothly between endpoints  
- latent paths are significantly longer than embedding-linear paths  
- curvature ratios indicate nonlinear embedding trajectories  

These results suggest the latent space behaves as a **smooth coordinate system over the embedding manifold**.

---

# Future Work

Future experiments will investigate how training parameters influence latent geometry, including:

- KL regularisation strength (β)  
- free-bits regularisation  
- latent dimensionality  

The current training framework already supports these configurations.

---

## Summary

Phase 2 demonstrates that the VAE latent space supports **smooth interpolation and structured geometry**, consistent with a learned latent manifold underlying sentence embeddings.

uv run python scripts/train.py  --latent_dim 32 --beta 0.1 --beta_warmup_epochs 5  --num_workers 0 --kl_free_bits 0.1 --max_length 128
uv run python scripts/train.py \      
--latent_dim 32 \
--beta 0.1 \
--beta_warmup_epochs 5  --num_workers 0 \
--kl_free_bits 0.1 \
--max_length 128

uv run python scripts/interp.py --min_len 10 \                    
--run_dir runs/ld32_b0.1_s42_20260311_001705 \
--sim_min 0.0 \
--sim_max 1.5 \
--num_pairs 3 \
--corpus_size 2000 \
--topk 5 --beta 0.1

uv run python scripts/interp.py --min_len 10 --run_dir runs/ld32_b0.1_s42_20260311_001705 --sim_min 0.0 --sim_max 1.5 --num_pairs 3 --corpus_size 2000 --topk 5 
