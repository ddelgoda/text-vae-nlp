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
  for b in 0.0 0.1 0.5 1.0 2.0 4.0; do
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
- train a VAE over the embeddings to reconstruct embeddings
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

## Landscape Analysis

### Reconstruction Loss Heatmap
![Reconstruction Heatmap](../artifacts/heatmap_recon.png)
• At β = 0.0, reconstruction loss monotonic with latent dimensionality: it worsens from 4→32.
• As β increases, KL is suppressed across all latent sizes, and reconstruction generally degrades relative to β = 0.0, with the strongest degradation appearing around latent_dim = 32.
• Overall, the grid demonstrates the point that the regulation effect by beta could force the reconstruction to ignore the latent space at beta=0 or to degrade the reconstruction when beta>0.
### KL Divergence Heatmap
![KL Heatmap](../artifacts/heatmap_kl.png)
• At β = 0.0, KL loss monotonic with latent dimensionality: it worsens from 4→32.
• As β increases, KL is suppressed across all latent sizes notwithstanding the value of beta
• Overall, the grid demonstrates posterior collpase across these points.


### Pareto Frontier
![Pareto Plot](../artifacts/pareto_plot_annotated.png)
The Pareto front isolates non-dominated configurations.
Collapsed solutions (near-zero KL) are filtered out.
The knee point is selected as the optimal trade-off between:
- compression (KL divergence)
- reconstruction accuracy

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
No learning rate scheduler is used in this project as the goal is to study **latent space behaviour**, not optimisation performance.  

Hyperparameter tuning is intentionally out of scope for this repo; the focus is controlled analysis of latent_dim and β. Automated tuning will be added when integrating this encoder into downstream tasks.

## Next Steps
Phase 2 investigates whether the learned latent space is meaningful via:
- latent interpolation
- comparison with a plain autoencoder baseline
- beta warm-up scheduling