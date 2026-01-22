# Latent Space Trade-offs in Text VAEs
> Exploring reconstruction–compression trade-offs in Variational Autoencoders applied to text embeddings.

---

## Overview

This project investigates how **latent dimensionality** and **KL regularisation** affect the behaviour of **Variational Autoencoders (VAEs)** when applied to sentence embeddings.

Rather than optimising for downstream task accuracy, the focus is on:

- understanding latent space utilisation  
- detecting posterior collapse  
- analysing reconstruction vs compression trade-offs  
- selecting optimal configurations using Pareto analysis  

The project is implemented as a clean, reproducible ML experiment using PyTorch Lightning.

---

## Motivation

In many VAE implementations:
- low reconstruction loss is mistaken for good performance  
- KL collapse goes unnoticed  
- latent variables carry little information  

This project demonstrates:
- why reconstruction loss alone is insufficient  
- how KL divergence behaves across latent sizes  
- how to select models using Pareto efficiency  
- how to structure ML experiments reproducibly  

---

## Model Overview

### Architecture

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Frozen transformer (no fine-tuning)
- VAE applied to embedding space
- Gaussian latent distribution

### Loss

\[

\mathcal{L} = \text{Reconstruction Loss} + \beta \cdot \text{KL Divergence}

\]

### Configuration

| Component | Value |

|----------|-------|

| Dataset | AG News |

| Token length | 64 |

| Latent dims | 4, 8, 16, 32, 64 |

| KL weights | 0.0 – 4.0 |

| Optimiser | AdamW |

| Framework | PyTorch Lightning |

---

##  Experimental Method

1. Train multiple VAE configurations  
2. Log reconstruction and KL loss  
3. Build a Pareto front  
4. Remove collapsed solutions (near-zero KL)  
5. Select the Pareto knee as the optimal trade-off  

---

## Results
<p align="center">
<img src="artifacts/pareto_plot.png" width="700"/>
</p>

### Key Observations

- Near-zero KL indicates posterior collapse  
- Larger latent spaces reduce reconstruction loss but increase KL  
- A clear Pareto knee emerges  
- Optimal models balance compression and expressiveness  

---

## Repository Structure
 text-vae-nlp/
│
├── src/textvae/
│   ├── model.py          # Transformer + VAE
│   ├── lit_module.py     # Lightning training logic
│   └── init.py
│
├── train.py              # Training entry point
├── eval.py               # Pareto analysis + plotting
│
├── artifacts/
│   ├── pareto_plot.png
│   ├── pareto_front.csv
│   └── sweet_spot.json
│
├── pyproject.toml
├── README.md
└── .gitignore

## Training the Model
The model is trained using a Variational Autoencoder (VAE) applied to sentence embeddings.  
Training optimises a combination of reconstruction loss and KL divergence to learn a compact latent representation.

### Run Training
To train the model, run:
```bash
uv run python train.py
```

This will:
- load the AG News dataset
- tokenize text using a transformer tokenizer
- encode sentences into embeddings
- train a VAE over the embeddings
- log reconstruction and KL losses
- save training metrics and checkpoints
---
### Training Configuration
Training behaviour can be controlled via command-line arguments:
``` bash
uv run python train.py \
 --latent_dim 16 \
 --beta 1.0 \
 --max_length 64 \
 --limit_train 1000 \
 --limit_val 500
```
### What Happens During Training

- Sentence embeddings are extracted using a frozen transformer
- A VAE learns to reconstruct embeddings
- KL divergence regularises the latent space
- Metrics are logged per epoch:
- reconstruction loss
- KL divergence
- total loss

Training is handled using PyTorch Lightning for:
- reproducibility
- clean training loops
- structured logging

## Evaluating the model

### Run evalution
To evaluate the model, run:
```bash
uv eval python train.py
```

This performs:
- aggregation of training metrics
- filtering of collapsed solutions
- construction of the Pareto front
- sweet-spot selection
- result visualisation
⸻
This generates:
- Pareto front
- Sweet-spot selection
- Visualisations and summary metrics

Artifacts are written to:

artifacts/
├── pareto_plot.png
├── pareto_front.csv
└── sweet_spot.json

Transformer weights are frozen for stability
- Metrics are logged at epoch level
- KL collapse is explicitly handled
- Experiments are fully reproducible

### Evaluation Configuration
Evaluation behaviour can be controlled via command-line arguments:
``` bash
uv run python eval.py \
 --kl_min 0.01 
```