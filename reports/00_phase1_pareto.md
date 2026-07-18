# Phase 1 — Pareto Analysis of Latent Trade-offs

---

## Objective

Study how latent dimensionality and KL regularisation (`β`) influence the reconstruction–compression trade-off in a Text VAE.

---

## Reasoning & Hypotheses

A low reconstruction loss alone can hide posterior collapse. To select a robust configuration for later phases, we evaluate both:

- reconstruction quality
- latent information usage (KL divergence)

Hypotheses:

- increasing latent dimensionality improves reconstruction capacity
- increasing `β` strengthens regularisation and can reduce latent usage if too strong
- the best configurations lie on a Pareto frontier rather than at a single metric extreme

---

## Experimental Setup

### Model

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Transformer frozen (no fine-tuning)
- VAE applied to embedding space
- Gaussian latent distribution
- Training objective:

```text
Loss = Reconstruction Loss + β × KL Divergence
```

### Grid

- `latent_dim ∈ {4, 8, 16, 32, 64}`
- `β ∈ {0.1, 0.5, 1.0, 2.0, 4.0}`

### Dataset

- AG News (subset for controlled experimentation)

### Method

1. Train multiple VAE configurations.
2. Log reconstruction and KL loss.
3. Build a Pareto front.
4. Remove collapsed solutions (near-zero KL).
5. Select the Pareto knee as the trade-off sweet spot.

> **Superseded** — see Correction at the end of this report. The front as constructed pooled frozen and unfrozen runs; the selected knee comes from the unfrozen set.

---

## Training

Run the sweep:

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

This sweep:

- loads AG News
- encodes sentences using a frozen transformer
- trains the embedding-space VAE
- logs reconstruction, KL, and total loss
- writes checkpoints and run metrics

---

## Evaluation

Run Pareto analysis:

```bash
uv run python scripts/eval.py --kl_min 0.01
```

This performs:

- metric aggregation
- collapse filtering
- Pareto front construction
- sweet-spot selection
- visualisation output

---

## Results

### Reconstruction Loss Heatmap

![Reconstruction Heatmap](../artifacts/heatmap_recon.png)
*Figure: Mean reconstruction loss across the grid of latent dimensions and β values.*

- Reconstruction varies more with latent dimensionality than with `β`.
- Across the β values explored, reconstruction remains relatively stable, indicating that reconstruction quality is largely insensitive to β in this range.
- Overall, the grid suggests that reconstruction is primarily controlled by latent compression (latent_dim) rather than the KL regularisation strength.

### KL Divergence Heatmap

![KL Heatmap](../artifacts/heatmap_kl.png)
*Figure: Mean KL divergence across the same hyperparameter grid.*

- KL divergence increases monotonically with **latent dimensionality**, reflecting the larger representational capacity of higher-dimensional latent spaces.
- Lower β values (e.g., **β = 0.1**) produce slightly weaker regularisation, while larger β values impose stronger constraints on the latent distribution.
- Together with the reconstruction heatmap, this indicates that **β mainly influences latent regularisation**, while reconstruction quality remains relatively stable.

### Pareto Frontier

![Pareto Plot](../artifacts/pareto_plot_annotated.png)
*Figure: Pareto frontier showing the trade-off between reconstruction loss and KL divergence.*

- Non-dominated solutions define the reconstruction–regularisation frontier.
- Near-zero KL runs are filtered as collapsed solutions.
- The knee point is selected as the Phase 2 candidate configuration.

### Selected Sweet Spot Artifacts

- `artifacts/sweet_spot.json`
- `artifacts/pareto_front.csv`

---

## Key Findings

> **Note:** the findings below are superseded — see the Correction section.

- Reconstruction-only model selection is insufficient for VAEs.
- KL filtering is required to detect and remove posterior collapse.
- Moderate latent sizes tend to dominate extreme settings on the frontier.

---

## Limitations and Next Steps

- Automated hyperparameter tuning is intentionally out of scope for this phase.
- Phase 2 validates whether the selected latent space supports meaningful interpolation geometry.

---

## Correction (added after initial analysis)

- **Pooling bug**: `eval.py`'s recursive run search (`rglob("metrics_summary.json")`) pulled 2 unfrozen runs from later work into the same pool as the 75 frozen sweep runs. Those runs use a different training recipe (β warmup, free bits) and are not comparable.
- **Condition filter**: `eval.py` now restricts to `freeze_transformer == True` before building the Pareto front (`--frozen_only`, on by default). The frozen sweep is exactly 75 runs (5 latent_dim × 5 β × 3 seeds).
- **Corrected knee**: with the filter applied, at `kl_min=0.001` the knee is **ld=4, β=0.1** (recon≈0.0262, kl≈0.0020) — not the run originally reported above.
- The knee is threshold-sensitive: at `kl_min=0.005` it moves to **ld=32, β=0.1** (recon≈0.0313, kl≈0.0056). It is not a stable landmark in this regime.
- **Hypotheses falsified**: both original hypotheses in "Reasoning & Hypotheses" are contradicted by the frozen sweep. Reconstruction does not improve with latent capacity — it degrades monotonically, from ≈0.0263 at `ld=4` to ≈0.0331 at `ld=64`. β has no effect distinguishable from seed noise, because with 0 active units there was no latent signal for it to regularise.
- **Active-units check** (Burda, Grosse & Salakhutdinov, 2015 — `A_i = Var_x(E_q[z_i|x])`, active if `A_i > 0.01`): 0 of 1860 latent units are active across all 75 frozen runs, at every latent_dim (4–64) and every β (0.1–4.0).
  - 0.01 is used because it's the field-standard AU threshold, and it's dimensionless by construction — a fraction of the prior's own variance (`Var[z] = 1` under the `N(0,I)` prior). This model's KL term is the same closed-form KL to `N(0,I)` used in the original AU literature, so the cutoff transfers without rescaling.
  - Threshold sweep: 0 active units at 0.005–0.05; only at 0.001 (10× below the standard cutoff, into numerical-noise territory) does the count jump to ~94.5%.
  - No bimodality: all 1860 per-dimension variances cluster in ~0.0009–0.003, with no gap separating a subset of "real" units from a noise floor — the signature of uniform, not partial, collapse.
- **Conclusion**: the frozen sweep is posterior-collapsed throughout. `ld=4, β=0.1` is the least-collapsed point in the grid, not a working VAE configuration. The Phase 2 model was trained under a different recipe (unfrozen encoder, β warm-up, free bits) and is not covered by this result; an active-units check on that checkpoint would be needed to establish whether its latent space is informative.
