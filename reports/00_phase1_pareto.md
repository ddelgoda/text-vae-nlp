# Phase 1 — Pareto Analysis of Latent Trade-offs
## Objective
To study how latent dimensionality and KL regularisation influence the reconstruction–compression trade-off in a Text VAE.
We aim to:
- detect posterior collapse
- analyse hyperparameter interaction
- select a principled sweet-spot configuration
- avoid relying on reconstruction loss alone

## Experimental Setup
### Model
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Frozen transformer (no fine-tuning)
- VAE applied to embedding space
- Gaussian latent distribution
- The model is trained using the function:
  ```text
  Loss = Rconstruction Loss+ β x KL Divergence
  ```
### Grid
- latent_dim ∈ {4, 8, 16, 32, 64}
- β ∈ {0.0, 0.1, 0.5, 1.0, 2.0, 4.0}
### Dataset
AG News (subset for controlled experimentation)
###  Experimental Method
1. Train multiple VAE configurations  
2. Log reconstruction and KL loss  
3. Build a Pareto front  
4. Remove collapsed solutions (near-zero KL)  
5. Select the Pareto knee as the optimal trade-off 
### Training the Model
The model is trained using a Variational Autoencoder (VAE) applied to sentence embeddings.  
Training optimises a combination of reconstruction loss and KL divergence to learn a compact latent representation.
To train the model, with a sweep, run:

```bash
for z in 4 8 16 32 64; do
  for b in 0.1 0.5 1.0 2.0 4.0; do
    for s in 0 1 2; do
      uv run python scripts/train.py \
        --latent_dim $z \
        --beta $b \
        --seed $s \
        --max_epochs 5 \
        --limit_train 1000 \
        --limit_val 500 \
        --freeze_transformer
    done
  done
done
```


This will:
- load the AG News dataset
- tokenize text using a transformer tokenizer
- encode sentences into embeddings using a frozen transformer for stability
- train a VAE over the embeddings to reconstruct embeddings. Note, β = 0.0 configuration is avoided in this phase as it corresponds to plain AE behavior, which will be assessed in phase 3.
- log reconstruction and KL losses
- save training metrics (reconstruction loss, KL divergence, total loss) and checkpoints

### Evaluating the model

To evaluate the model, run:
```bash
uv run python scripts/eval.py --kl_min 0.01
```
This performs:
- aggregation of training metrics
- filtering of collapsed solutions
- construction of the Pareto front
- sweet-spot selection
- result visualisation

## Reconstruction Loss Heatmap
![Reconstruction Heatmap](../artifacts/heatmap_recon.png)

*Figure: Mean reconstruction loss across the grid of latent dimensions and β values.*

- Reconstruction loss increases as the **latent dimension grows**, with the strongest degradation appearing around **latent_dim = 32**.
- Across the β values explored, reconstruction remains relatively stable, indicating that **reconstruction quality is largely insensitive to β** in this range.
- Overall, the grid suggests that **reconstruction is primarily controlled by latent compression (latent_dim)** rather than the KL regularisation strength.


## KL Divergence Heatmap
![KL Heatmap](../artifacts/heatmap_kl.png)

*Figure: Mean KL divergence across the same hyperparameter grid.*

- KL divergence increases monotonically with **latent dimensionality**, reflecting the larger representational capacity of higher-dimensional latent spaces.
- Lower β values (e.g., **β = 0.1**) produce slightly weaker regularisation, while larger β values impose stronger constraints on the latent distribution.
- Together with the reconstruction heatmap, this indicates that **β mainly influences latent regularisation**, while reconstruction quality remains relatively stable.


## Pareto Frontier
![Pareto Plot](../artifacts/pareto_plot_annotated.png)

*Figure: Pareto frontier showing the trade-off between reconstruction loss and KL divergence.*

The Pareto front isolates **non-dominated configurations** in the reconstruction–regularisation trade-off.

Collapsed solutions (near-zero KL) are filtered out.

The selected **knee point** represents the best balance between:

- **compression** (KL divergence)
- **reconstruction accuracy**

In this sweep, **reconstruction behaviour is largely governed by latent dimensionality**, while **β primarily controls the strength of latent regularisation**.

This configuration is used as the **candidate model for Phase 2 latent interpolation experiments**.

### Selected Sweet Spot
See:
- `artifacts/sweet_spot.json`
- `artifacts/pareto_front.csv`
The selected configuration balances expressive latent capacity with controlled regularisation.

## Key Findings
- Larger latent spaces reduce reconstruction loss but increase KL. Reconstruction loss alone is insufficient.
- Near-zero KL indicates posterior collapse. KL filtering is necessary to avoid collapse.
- Moderate latent sizes dominate extreme configurations.
- Trade-offs form a structured frontier, not random scatter.Optimal models balance compression and expressiveness 

## Optimisation Notes
Hyperparameter tuning is intentionally out of scope for this repo; the focus is controlled analysis of latent_dim and β. Automated tuning will be added when integrating this encoder into downstream tasks.

## Next Steps
Phase 2 and 3 investigate whether the learned latent space is meaningful via:
- latent interpolation
- comparison with a plain autoencoder baseline