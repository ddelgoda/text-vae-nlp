# Reports Overview

---

## Objectives

This project studies Variational Autoencoders (VAEs) in embedding space, from trade-off analysis to latent-space validation.

Primary goals:

- characterise reconstruction–compression trade-offs across latent dimensions and KL strengths
- detect and mitigate posterior collapse
- validate that learned latent structure is semantically meaningful
- compare VAE behaviour with a plain autoencoder baseline (later phase — now moot, see below)
- assess training dynamics such as β scheduling

**Outcome:** both phases found posterior collapse. No configuration tested produced a latent space carrying detectable input-dependent signal, so the semantic-validation and autoencoder-baseline objectives were not reached — there was no informative latent space to validate or compare against. See the correction sections in both phase reports.

---

## Configuration Summary

Phase 1 was intended as model selection on AG News; it became a collapse diagnosis, since no configuration produced a usable latent representation.
Phase 2 was intended as interpolation geometry validation on STS-B; the checkpoint it used is also collapsed, so the geometry results describe a near-constant decoder.

### Hyperparameter Selection Across Phases

Phase 1 and Phase 2 use different datasets and therefore evaluate different properties of the model.

Phase 1 uses the AG News dataset to study the trade-off between reconstruction quality and KL divergence across latent dimensions and β values.

Phase 2 uses the STS-B dataset to evaluate latent interpolation behaviour and geometric properties of the learned manifold.

Because the datasets differ substantially in sentence length, topic structure, and semantic granularity, hyperparameters are not strictly transferred between phases. Each phase is therefore analysed independently to avoid introducing bias from dataset-specific optimisation.

**Correction:** this separation held in practice, but not for the reason implied. The Phase 2 configuration was not carried over from Phase 1's selection — it came from separate later runs under a different training recipe. Both phases are now known to be posterior-collapsed, so neither produced a latent space whose hyperparameters would have been worth transferring.

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

- `00_phase1_pareto.md` — latent dimension vs KL sweep, Pareto analysis, and active-unit collapse diagnosis. Present on both branches.
- `01_phase2_latent.md` — latent interpolation geometry, and the active-unit check that reinterprets it. Present on `feat/phase2-latent-interp` only; from `main`, read it [here](https://github.com/ddelgoda/text-vae-nlp/blob/feat/phase2-latent-interp/reports/01_phase2_latent.md).

Both reports preserve their original analysis verbatim, with corrections appended in a
final section and superseded claims flagged inline.

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
│   ├── active_units.py   # Active-unit collapse diagnostic
│   └── interp.py         # Latent interpolation (feat/phase2-latent-interp only)
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
