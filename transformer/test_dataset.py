import os
from os.path import exists
import torch
from models.dataset import load_tokenizers, create_dataloaders

# Main script entry point
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔠 Loading tokenizers...")
    tokenizer_src, tokenizer_tgt = load_tokenizers()

    print("📦 Creating dataloaders...")
    train_loader, valid_loader = create_dataloaders(tokenizer_src, tokenizer_tgt, batch_size=64)

    # Example: Show the shape of the first batch
    for batch in train_loader:
        print("📊 Batch shapes:", {k: v.shape for k, v in batch.items()})
        break