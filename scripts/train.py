# ------------------------------------------------------------
# train.py for training the TextEmbeddingVAE
#
# This file:
# - loads a small public dataset
# - tokenizes text
# - runs Lightning training
# ------------------------------------------------------------


import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightning as L
from datasets import load_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from textvae.lit_module import LitTextVAE

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--limit_train", type=int, default=1000)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--freeze_transformer", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_val", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--beta_warmup_epochs", type=int, default=5)
    parser.add_argument("--kl_free_bits", type=float, default=0.5)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    L.seed_everything(args.seed, workers=True)
    run_id = f"ld{args.latent_dim}_b{args.beta}_s{args.seed}_{ts}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "freeze_transformer": bool(args.freeze_transformer),
        "lr": args.lr,
        "beta": args.beta,
        "limit_val": args.limit_val,
        "seed": args.seed,
        "kl_free_bits": args.kl_free_bits,
    }

    (run_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2), encoding="utf-8"
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load a simple public dataset (news headlines)
    ds = load_dataset("stsb_multi_mt", name = "en")
    train_ds = ds["train"]
    val_ds = ds["test"]

    train_ds = train_ds.select(range(min(args.limit_train, len(train_ds))))
    val_ds = val_ds.select(range(min(args.limit_val, len(val_ds))))

    def tokenize(batch):
        return tokenizer(
            batch["sentence1"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_ds = train_ds.map(
        tokenize, batched=True, remove_columns=train_ds.column_names
    )
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )
    # Lightning module

    model = LitTextVAE(
        model_name=args.model_name,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        freeze_transformer=args.freeze_transformer,
        lr=args.lr,
        beta=args.beta,
        beta_warmup_epochs = args.beta_warmup_epochs,
        kl_free_bits = args.kl_free_bits,
        run_dir=str(run_dir),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="last",
        monitor="val/total_loss",
        mode="min",
        save_top_k=0,
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        default_root_dir=str(run_dir),
        callbacks=[ckpt_cb],
        enable_checkpointing=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":

    main()
