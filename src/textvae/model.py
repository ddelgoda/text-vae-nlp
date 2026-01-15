# ------------------------------------------------------------
# model.py
#
# This file defines the *pure neural network model*:
# - Transformer encoder for text
# - Variational Autoencoder (VAE) on top of embeddings
# ------------------------------------------------------------

from __future__ import annotations 
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModel

# ------------------------------------------------------------
# Transformer encoder wrapper
# ------------------------------------------------------------
class TransformerEmbeddingEncoder(nn.Module):
   """
   Converts tokenised text into a single sentence embedding.
   Pipeline:
       input_ids, attention_mask
           ↓
       Transformer (token-level embeddings)
           ↓
       Masked mean pooling
           ↓
       Sentence embedding (B, D)
   """
   def __init__(self, model_name: str, freeze: bool = True) -> None:
       super().__init__()
       # Load a pretrained transformer (encoder only, no LM head)
       self.model = AutoModel.from_pretrained(model_name)
       # Whether to freeze transformer weights
       self.freeze = freeze
       if freeze:
           # Freezing avoids training the transformer
           for p in self.model.parameters():
               p.requires_grad = False
   @staticmethod
   def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
       """
       Mean pooling with masking.
       last_hidden:     (B, T, H) token embeddings
       attention_mask: (B, T)    1 = real token, 0 = padding
       Returns:
           sentence embedding of shape (B, H)
       """
       # Expand mask so it can multiply token embeddings
       # (B, T) → (B, T, 1)
       mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
       # Zero-out padded token embeddings
       masked = last_hidden * mask
       # Sum over tokens (T dimension)
       summed = masked.sum(dim=1)  # (B, H)
       # Count how many real tokens each sentence has
       counts = mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
       # Mean = sum / count
       return summed / counts
       
   def forward(self, input_ids: torch.Tensor,attention_mask: torch.Tensor) -> torch.Tensor:
       """
       Forward pass of the transformer encoder.
       Inputs:
           input_ids      : (B, T)
           attention_mask : (B, T)
       Output:
           emb            : (B, D) sentence embedding
       """
       # If frozen, do not track gradients
       if self.freeze:
           with torch.no_grad():
               out = self.model(input_ids=input_ids, attention_mask=attention_mask)
       else:
           out = self.model(input_ids=input_ids, attention_mask=attention_mask)
       # Token-level embeddings from the final transformer layer
       last_hidden = out.last_hidden_state  # (B, T, H)
       # Collapse tokens → sentence embedding
       emb = self.mean_pool(last_hidden, attention_mask)  # (B, H)
       return emb


# ------------------------------------------------------------
# Output container for the VAE forward pass
# ------------------------------------------------------------
@dataclass
class VAEOutput:
   """
   Container to group all outputs of the VAE
   Shapes:
       emb        : (B, D)  original sentence embedding
       recon_emb  : (B, D)  reconstructed embedding
       mu         : (B, Z)  latent mean
       logvar     : (B, Z)  latent log-variance
       z          : (B, Z)  sampled latent vector
   """
   emb: torch.Tensor
   recon_emb: torch.Tensor
   mu: torch.Tensor
   logvar: torch.Tensor
   z: torch.Tensor