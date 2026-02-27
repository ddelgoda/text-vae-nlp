# ------------------------------------------------------------
# lit_module.py for the TextEmbeddingVAE
#
# This file defines:
# - how loss is computed
# - how training runs
# - which optimizer is used
# ------------------------------------------------------------

import json
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F

from textvae.model import TextEmbeddingVAE


class LitTextVAE(L.LightningModule):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        latent_dim: int = 32,
        hidden_dim: int = 256,
        freeze_transformer: bool = True,
        lr: float = 2e-4,
        beta: float = 1.0,
        beta_warmup_epochs: int = 0,
        kl_free_bits: float = 0.0, 
        run_dir: Optional[str] = None,
    ):
        super().__init__()
        # Save all hyperparameters for logging / reproducibility
        self.save_hyperparameters()
        # The actual VAE model (defined in model.py)
        self.vae = TextEmbeddingVAE(
            model_name=model_name,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            freeze_transformer=freeze_transformer,
        )
        self.lr = lr
        self.beta = beta
        self.run_dir = Path(run_dir) if run_dir else None
        self.beta_warmup_epochs = beta_warmup_epochs
        self.kl_free_bits = kl_free_bits


    @staticmethod
    def kl_diag_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between:
            q(z|x) = N(mu, sigma^2)
        and:
            p(z) = N(0, I)
        Closed-form for diagonal Gaussian.
        """
        kl = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)
        return kl.sum(dim=1).mean()

    @staticmethod
    def kl_diag_gaussian_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Returns KL per sample (B,)
        q(z|x)=N(mu, sigma^2), p(z)=N(0,I)
        """
        kl_per_dim = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)  # (B, Z)
        kl_per_sample = kl_per_dim.sum(dim=1)  # (B,)
        return kl_per_sample

    def training_step(self, batch, batch_idx):
        """
        One training step on one batch.
        """
        # Forward pass through the VAE
        out = self.vae(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        # Reconstruction loss (embedding space)
        recon_loss = F.mse_loss(out.recon_emb, out.emb)
        # KL divergence loss per sample (B,)
        kl_per_sample = self.kl_diag_gaussian_per_sample(out.mu, out.logvar)

        # KL mean over batch for reporting
        kl_raw = kl_per_sample.mean()

        # Free-bits: only penalise KL above  a threshold
        # Total VAE loss
        if self.beta_warmup_epochs>0:
            warm = min(1.0, float(self.current_epoch+1)/float(self.beta_warmup_epochs),)
            beta_eff = self.beta*warm
        else:
            beta_eff = self.beta
        loss = recon_loss + beta_eff * kl_loss

        self.log("train/beta_eff", beta_eff, prog_bar = True)
        # Log losses for progress bar / TensorBoard
        self.log_dict(
            {
                "train/total_loss": loss,
                "train/recon_loss": recon_loss,
                "train/kl_loss": kl_loss,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.vae(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        recon_loss = F.mse_loss(out.recon_emb, out.emb)
        kl_loss = self.kl_diag_gaussian(out.mu, out.logvar)
        loss = recon_loss + self.beta * kl_loss
        self.log_dict(
            {
                "val/total_loss": loss,
                "val/recon_loss": recon_loss,
                "val/kl_loss": kl_loss,
            },
            prog_bar=False,
            on_step=False,  # On setp metrices are noisy
            on_epoch=True,
        )
        return loss

    def on_fit_start(self):

        self.train()
        # If transformer is frozen, keep it in eval mode (no dropout, stable embeddings)
        if getattr(self.hparams, "freeze_transformer", False):
            self.vae.encoder.model.eval()

    def on_fit_end(self):
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)

        def _get_float(key: str):
            v = self.trainer.callback_metrics.get(key)
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            return float(v)

        best_path = None
        best_score = None
        for cb in self.trainer.callbacks:
            if cb.__class__.__name__ == "ModelCheckpoint":
                best_path = getattr(cb, "best_model_path", None) or None
                best_score = getattr(cb, "best_model_score", None)
                if isinstance(best_score, torch.Tensor):
                    best_score = float(best_score.detach().cpu().item())
                break
        summary = {
            "run_id": self.run_dir.name,
            "latent_dim": int(self.hparams.latent_dim),
            "beta": float(self.hparams.beta),
            "model_name": str(self.hparams.model_name),
            "freeze_transformer": bool(self.hparams.freeze_transformer),
            "lr": float(self.hparams.lr),
            "recon_loss": _get_float("val/recon_loss_epoch")
            or _get_float("val/recon_loss"),
            "kl_loss": _get_float("val/kl_loss_epoch") or _get_float("val/kl_loss"),
            "total_loss": _get_float("val/total_loss_epoch")
            or _get_float("val/total_loss"),
            "best_model_path": best_path,
            "best_model_score": best_score,
        }
        (self.run_dir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    def configure_optimizers(self):
        """
        Optimizer used for training.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
