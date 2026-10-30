"""
Phase 3 Test: PyTorch Dataset, Model Architectures (SimpleRNN, GRU, LSTM) and Training Convergence.
"""
import os
import sys
import torch
from torch.utils.data import DataLoader

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import load_annotations, build_vocab, TAG2IDX, PAD_IDX
from src.dataset import NERDataset, pad_collate_fn
from src.model import RNNSequenceTagger
from src.train import train_ner_model, save_model_checkpoint, load_model_checkpoint

def test_dataset_and_collate():
    print("[1/4] Testing PyTorch NERDataset and pad_collate_fn...")
    annotations = load_annotations("data/annotations.json")
    train_data = annotations["train"]
    word2idx, _ = build_vocab(train_data)

    dataset = NERDataset(train_data, word2idx, TAG2IDX)
    assert len(dataset) == 10, f"Expected 10 items in dataset, got {len(dataset)}"

    sample_tokens, sample_tags, raw_toks, raw_tgs = dataset[0]
    assert len(sample_tokens) == len(sample_tags)
    assert isinstance(sample_tokens, torch.Tensor)

    # Test batching with DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=pad_collate_fn)
    padded_tokens, padded_tags, lengths, _, _ = next(iter(loader))

    assert padded_tokens.shape[0] == 4, f"Batch size mismatch: {padded_tokens.shape[0]}"
    assert padded_tokens.shape == padded_tags.shape
    print(f"  [OK] Batch shape: {padded_tokens.shape} (batch_size=4, max_seq_len={padded_tokens.shape[1]})")
    print(f"  [OK] Sentence lengths in batch: {lengths.tolist()}")

def test_architectures():
    print("\n[2/4] Testing forward pass for SimpleRNN, GRU, and BiLSTM...")
    vocab_size = 70
    tagset_size = len(TAG2IDX)
    batch_size = 2
    seq_len = 6
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    for cell in ["rnn", "gru", "lstm"]:
        model = RNNSequenceTagger(
            vocab_size=vocab_size,
            tagset_size=tagset_size,
            embedding_dim=32,
            hidden_dim=32,
            cell_type=cell,
            bidirectional=True,
            dropout=0.0
        )
        logits = model(dummy_input)
        assert logits.shape == (batch_size, seq_len, tagset_size), (
            f"Shape error in {cell}: expected {(batch_size, seq_len, tagset_size)}, got {logits.shape}"
        )
        print(f"  [OK] Architecture '{cell.upper()}': output logits shape {tuple(logits.shape)}")

def test_training_convergence():
    print("\n[3/4] Testing PyTorch training convergence on training dataset...")
    annotations = load_annotations("data/annotations.json")
    train_data = annotations["train"]
    word2idx, _ = build_vocab(train_data)

    dataset = NERDataset(train_data, word2idx, TAG2IDX)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=pad_collate_fn)

    model = RNNSequenceTagger(
        vocab_size=len(word2idx),
        tagset_size=len(TAG2IDX),
        embedding_dim=64,
        hidden_dim=64,
        cell_type="lstm",
        bidirectional=True,
        dropout=0.1
    )

    losses = train_ner_model(model, loader, num_epochs=35, learning_rate=0.01)

    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"  Initial Loss (Epoch 1):  {initial_loss:.4f}")
    print(f"  Final Loss   (Epoch 35): {final_loss:.4f}")

    assert final_loss < initial_loss, "Model did not reduce loss during training!"
    assert final_loss < 0.25, f"Expected final loss < 0.25 on 10 training samples, got {final_loss:.4f}"
    print("  [OK] Model demonstrated clean convergence and learned the training annotations.")

def test_checkpointing():
    print("\n[4/4] Testing model checkpoint serialization and reloading...")
    annotations = load_annotations("data/annotations.json")
    word2idx, _ = build_vocab(annotations["train"])

    model = RNNSequenceTagger(
        vocab_size=len(word2idx),
        tagset_size=len(TAG2IDX),
        embedding_dim=32,
        hidden_dim=32,
        cell_type="lstm",
        bidirectional=True
    )
    test_ckpt = "models/test_checkpoint.pt"
    save_model_checkpoint(model, test_ckpt, len(word2idx), len(TAG2IDX))
    assert os.path.exists(test_ckpt)

    loaded_model = load_model_checkpoint(test_ckpt)
    dummy_input = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        orig_out = model(dummy_input)
        loaded_out = loaded_model(dummy_input)

    assert torch.allclose(orig_out, loaded_out), "Loaded model outputs do not match original model!"
    print(f"  [OK] Checkpoint saved and verified with 100% parameter fidelity.")
    if os.path.exists(test_ckpt):
        os.remove(test_ckpt)

if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING PHASE 3 TEST: PYTORCH DATASET, RNNs & CONVERGENCE")
    print("=" * 60)
    test_dataset_and_collate()
    test_architectures()
    test_training_convergence()
    test_checkpointing()
    print("\n" + "=" * 60)
    print(" ALL PHASE 3 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
