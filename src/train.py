"""
Training Engine for Sequence Labeling RNN Models in PyTorch.
"""
import os
import json
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_utils import load_annotations, build_vocab, save_vocab, PAD_IDX, TAG2IDX
from src.dataset import NERDataset, pad_collate_fn
from src.model import RNNSequenceTagger

def train_ner_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.01,
    device: str = "cpu"
) -> List[float]:
    """
    Trains the sequence tagger using Cross-Entropy Loss with padding ignored.
    """
    model.to(device)
    model.train()

    # Note: ignore_index=PAD_IDX ensures padding tokens do not contribute to loss or gradient updates
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses = []

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_tokens = 0

        for tokens, tags, lengths, _, _ in train_loader:
            tokens = tokens.to(device)
            tags = tags.to(device)

            optimizer.zero_grad()

            # Forward pass: (batch_size, seq_len, tagset_size)
            logits = model(tokens)

            # Flatten logits to (batch_size * seq_len, tagset_size) and tags to (batch_size * seq_len)
            loss = criterion(logits.view(-1, logits.shape[-1]), tags.view(-1))

            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item() * tokens.size(0)
            total_tokens += tokens.size(0)

        avg_loss = total_loss / max(total_tokens, 1)
        epoch_losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch:02d}/{num_epochs:02d} | Loss: {avg_loss:.4f}")

    return epoch_losses

def save_model_checkpoint(
    model: RNNSequenceTagger,
    save_path: str,
    vocab_size: int,
    tagset_size: int
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "tagset_size": tagset_size,
            "embedding_dim": model.embedding.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "cell_type": model.cell_type,
            "bidirectional": model.bidirectional
        }
    }
    torch.save(checkpoint, save_path)
    print(f"[Model] Checkpoint saved successfully to: {save_path}")

def load_model_checkpoint(checkpoint_path: str, device: str = "cpu") -> RNNSequenceTagger:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = RNNSequenceTagger(
        vocab_size=config["vocab_size"],
        tagset_size=config["tagset_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        cell_type=config["cell_type"],
        bidirectional=config["bidirectional"],
        dropout=0.0
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model

def run_training_pipeline(
    cell_type: str = "lstm",
    num_epochs: int = 50,
    save_path: str = "models/ner_bilstm.pt"
):
    print(f"\n--- Starting Training for {cell_type.upper()} ---")
    data = load_annotations("data/annotations.json")
    train_data = data["train"]

    word2idx, _ = build_vocab(train_data)
    save_vocab(word2idx, "data/vocab.json")

    dataset = NERDataset(train_data, word2idx, TAG2IDX)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=pad_collate_fn)

    model = RNNSequenceTagger(
        vocab_size=len(word2idx),
        tagset_size=len(TAG2IDX),
        embedding_dim=64,
        hidden_dim=64,
        cell_type=cell_type,
        bidirectional=True,
        dropout=0.1
    )

    losses = train_ner_model(model, loader, num_epochs=num_epochs, learning_rate=0.01)
    save_model_checkpoint(model, save_path, len(word2idx), len(TAG2IDX))
    return model, losses

if __name__ == "__main__":
    run_training_pipeline(cell_type="lstm", num_epochs=50, save_path="models/ner_bilstm.pt")
