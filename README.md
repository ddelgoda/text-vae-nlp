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

These experiments identify **Pareto-optimal configurations** and a candidate model for further analysis.

### Phase 2 — Latent Geometry Analysis (STS-B)

Phase 2 probes the **geometry of the learned latent space** using sentence pairs from the **STS-B (Semantic Textual Similarity Benchmark)** dataset.

STS-B provides interpretable sentence pairs that are well suited for **latent interpolation experiments**, allowing the trajectory between two sentences to be analysed in embedding space.

Using separate datasets keeps:

- **Phase 1 focused on model selection**
- **Phase 2 focused on geometric evaluation**

---

## Branch Structure

To keep the main codebase stable, experimental work for Phase 2 is maintained in a separate branch.

- **`main`**
Stable implementation and results for Phase 1 (AG News Pareto analysis).

- **`phase2-latent-interp`**
Experimental work exploring latent interpolation and geometry analysis using STS-B.

Phase 2 documentation is available in:

- `reports/phase2_latent_interpolation.md`

Full project documentation is available here:

➡ **[Project Overview](reports/README.md)**
➡ **[Phase 1 – Pareto Analysis](reports/phase1.md)**
➡ **[Phase 2 – Latent Geometry](reports/phase2_latent_interpolation.md)**

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

Latent interpolation produces **smooth semantic trajectories** while embedding interpolation remains nearly linear, indicating that the VAE decoder maps latent straight lines onto **curved embedding manifolds**.

## Disclaimer

This repository documents personal research exploring latent representations in Variational Autoencoders and their geometric properties in embedding spaces.

The work was undertaken as part of ongoing technical development in machine learning and AI assurance.
All experiments use publicly available datasets and open-source tools and are conducted independently of my employer.
