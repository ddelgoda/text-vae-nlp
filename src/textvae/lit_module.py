# ------------------------------------------------------------
# lit_module.py for the TextEmbeddingVAE
#
# This file defines:
# - how loss is computed
# - how training/validation run
# - which optimizer is used
# - how metrics_summary.json is written
# ------------------------------------------------------------

from __future__ import annotations
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
        beta_warmup_epochs: int = 5,
        kl_free_bits: float = 0.05,
        run_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.vae = TextEmbeddingVAE(
            model_name=model_name,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            freeze_transformer=freeze_transformer,
        )
        self.lr = lr
        self.beta = beta
        self.beta_warmup_epochs = beta_warmup_epochs
        self.kl_free_bits = kl_free_bits
        self.run_dir = Path(run_dir) if run_dir else None

    # ----------------------------
    # KL helpers
    # ----------------------------

    @staticmethod
    def kl_diag_gaussian_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL(q(z|x) || p(z)) per sample.
        mu, logvar : (B, Z)
        returns    : (B,)
        """
        kl_per_dim = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)  # (B, Z)
        return kl_per_dim.sum(dim=1)  # (B,)

    def _beta_eff(self) -> float:
        """
        Effective beta for the current epoch (warmup schedule).
        Returns a Python float so logging is simple.
        """
        if self.beta_warmup_epochs and self.beta_warmup_epochs > 0:
            warm = min(
                1.0,
                float(self.current_epoch + 1) / float(self.beta_warmup_epochs),
            )
            return float(self.beta) * warm
        return float(self.beta)

    def _loss_terms(
        self, out
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:

        """
        Shared loss computation for train/val.
        Returns:
            recon_loss (scalar tensor),
            kl_raw     (scalar tensor),
            kl_used    (scalar tensor),
            total_loss (scalar tensor),
            beta_eff   (float)
        """
        recon_loss = F.mse_loss(out.recon_emb, out.emb)
        kl_per_dim = self.kl_diag_gaussian_per_dim(out.mu, out.logvar)  # (B,)

        if kl_per_dim.dim() == 1:
            kl_per_dim = kl_per_dim.unsqueeze(0)
        free_per_dim = float(self.kl_free_bits)/float(self.hparams.latent_dim)
        kl_clamped = torch.clamp(kl_per_dim , min=free_per_dim)
        kl_used=kl_clamped.sum(dim=1).mean()
        kl_raw = kl_per_dim.sum(dim=1).mean()
        
        
        beta_eff = self._beta_eff()
        total_loss = recon_loss + (beta_eff * kl_used)
        return recon_loss, kl_raw, kl_used, total_loss, beta_eff

    # ----------------------------
    # Lightning hooks
    # ----------------------------

    def training_step(self, batch, batch_idx):
        out = self.vae(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        recon_loss, kl_raw, kl_used, loss, beta_eff = self._loss_terms(out)
        # Epoch-level logs (clean + stable)
        self.log("train/beta_eff", beta_eff, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/kl_raw", kl_raw, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/kl_used", kl_used, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(
            {
                "train/total_loss": loss,
                "train/recon_loss": recon_loss,
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
        recon_loss, kl_raw, kl_used, loss, beta_eff = self._loss_terms(out)
        self.log_dict(
            {
                "val/total_loss": loss,
                "val/recon_loss": recon_loss,
                "val/kl_raw": kl_raw,
                "val/kl_used": kl_used,
                "val/beta_eff": beta_eff,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def on_fit_start(self):
        self.train()
        # If transformer is frozen, keep it in eval mode (no dropout, stable embeddings)
        if getattr(self.hparams, "freeze_transformer", False):
            self.vae.encoder.model.eval()

    def on_fit_end(self):
        """
        Write a compact run summary for eval.py to read.
        """
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

        # Find checkpoint info (if any)
        best_path = None
        best_score = None
        last_path = None
        last_score = None
        for cb in self.trainer.callbacks:
            if cb.__class__.__name__ == "ModelCheckpoint":
                best_path = getattr(cb, "best_model_path", None) or None
                best_score = getattr(cb, "best_model_score", None)
                if isinstance(best_score, torch.Tensor):
                    best_score = float(best_score.detach().cpu().item())
                last_path = getattr(cb, "last_model_path", None) or None
                last_score = getattr(cb, "last_model_score", None)
                if isinstance(last_score, torch.Tensor):
                    last_score = float(last_score.detach().cpu().item())
                break
        summary = {
            "run_id": self.run_dir.name,
            "latent_dim": int(self.hparams.latent_dim),
            "beta": float(self.hparams.beta),
            "beta_warmup_epochs": int(getattr(self.hparams, "beta_warmup_epochs", 0)),
            "kl_free_bits": float(getattr(self.hparams, "kl_free_bits", 0.0)),
            "model_name": str(self.hparams.model_name),
            "freeze_transformer": bool(self.hparams.freeze_transformer),
            "lr": float(self.hparams.lr),

            # Prefer validation epoch metrics (Lightning typically aggregates to these keys)
            "recon_loss": _get_float("val/recon_loss_epoch") or _get_float("val/recon_loss"),
            "kl_loss": _get_float("val/kl_raw_epoch") or _get_float("val/kl_raw"),
            "kl_used": _get_float("val/kl_used_epoch") or _get_float("val/kl_used"),
            "total_loss": _get_float("val/total_loss_epoch") or _get_float("val/total_loss"),
            "best_model_path": best_path,
            "best_model_score": best_score,
            "last_model_path": last_path,
            "last_model_score": last_score,
        }
        (self.run_dir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
 