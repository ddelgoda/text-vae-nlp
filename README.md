# Text VAE for Sentence Embeddings

This repository explores a **Variational Autoencoder (VAE)** trained on sentence embeddings and studies the geometry of its latent space.

The project is organised into two main phases:

- **Phase 1 — Model Selection & Pareto Analysis**
- **Phase 2 — Latent Interpolation Geometry**

Full project documentation is available here:

➡ **[Project Overview](reports/README.md)**
➡ **[Phase 1 – Pareto Analysis](reports/phase1.md)**
➡ **[Phase 2 – Latent Geometry](reports/phase2_latent_interpolation.md)**

---

## Repository Structure

---

## Key Result

Latent interpolation produces **smooth semantic trajectories** while embedding interpolation remains nearly linear, indicating that the VAE decoder maps latent straight lines onto **curved embedding manifolds**.
