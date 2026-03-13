# Reports Overview

---

## Objectives

This project studies Variational Autoencoders (VAEs) in embedding space, from trade-off analysis to latent-space validation.

Primary goals:

- characterise reconstruction–compression trade-offs across latent dimensions and KL strengths
- detect and mitigate posterior collapse
- validate that learned latent structure is semantically meaningful
- compare VAE behaviour with a plain autoencoder baseline (later phase)
- assess training dynamics such as β scheduling

---

## Configuration Summary

Phase 1 focuses on model selection with AG News.
Phase 2 focuses on interpolation geometry validation with STS-B-style sentence similarity analysis.

### Hyperparameter Selection Across Phases

Phase 1 and Phase 2 use different datasets and therefore evaluate different properties of the model.

Phase 1 uses the AG News dataset to study the trade-off between reconstruction quality and KL divergence across latent dimensions and β values.

Phase 2 uses the STS-B dataset to evaluate latent interpolation behaviour and geometric properties of the learned manifold.

Because the datasets differ substantially in sentence length, topic structure, and semantic granularity, hyperparameters are not strictly transferred between phases. Each phase is therefore analysed independently to avoid introducing bias from dataset-specific optimisation.

### Phase 1

| Component | Value |
|----------|-------|
| Dataset | AG News (model selection) |
| Token length | 64 |
| Optimiser | AdamW |
| Framework | PyTorch Lightning |

### Phase 2

| Component | Value |
|----------|-------|
| Dataset | STS-B style interpolation evaluation |
| Token length | 128 |
| Optimiser | AdamW |
| Framework | PyTorch Lightning |

---

## Report Flow

Both phase reports follow the same structure:

1. Objective
2. Reasoning & Hypotheses
3. Experimental Setup
4. Training
5. Evaluation
6. Results
7. Key Findings
8. Limitations and Next Steps

---

## Experiment Reports

- `00_phase1_pareto.md` — latent dimension vs KL trade-off analysis and sweet-spot selection
- `01_phase2_latent.md` — latent interpolation geometry and semantic trajectory evaluation

---

## Repository Structure

```text
text-vae-nlp/
├── src/textvae/
│   ├── model.py          # Transformer + VAE
│   ├── lit_module.py     # Lightning training logic
│   └── __init__.py
├── scripts/
│   ├── train.py          # Training entry point
│   ├── eval.py           # Pareto analysis + plotting
│   └── interp.py         # Latent interpolation
├── artifacts/
│   ├── pareto_plot.png
│   ├── pareto_front.csv
│   └── sweet_spot.json
├── reports/
│   ├── 00_phase1_pareto.md
│   ├── 01_phase2_latent.md
│   └── README.md
├── pyproject.toml
└── README.md
```

Artifacts are written to the `artifacts/` folder.
