## Objectives
This project aims to systematically study the behaviour of Variational Autoencoders (VAEs) in embedding space, progressing from controlled trade-off analysis to latent space validation and training dynamics.
Across multiple phases, the objectives are to:
• Characterise reconstruction–compression trade-offs across latent dimensions and KL regularisation strengths
• Detect and mitigate posterior collapse
• Validate that the learned latent space is semantically meaningful
• Compare VAE behaviour against a plain autoencoder baseline
• Analyse the effect of training dynamics (e.g., β scheduling) on stability and representation quality
• Structure experiments in a reproducible, extensible research-to-production workflow


### Configuration

| Component | Value |
|----------|-------|
| Dataset | AG News |
| Token length | 64 |
| Optimiser | AdamW |
| Framework | PyTorch Lightning |

# Experiment Reports
This folder contains structured experiment write-ups.
- 00_phase1_pareto.md — Latent dimension vs KL trade-off analysis and sweet-spot selection.

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

Artifacts are written to:
```
artifacts/
├── pareto_plot.png
├── pareto_front.csv
├── pareto_plot_annotated.png
├── pareto_all.csv
├── heatmap_kl.png
├── heatmap_recon.png
├── sweet_spot.csv
└── sweet_spot.json
```