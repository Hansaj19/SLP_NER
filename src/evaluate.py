"""
Model Evaluation and Entity Recognition Analysis.
Computes token-level and chunk-level (seqeval) precision, recall, and F1-score.
"""
import os
import sys
import json
from typing import Dict, Any, List, Tuple
import torch
from tabulate import tabulate
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_utils import load_annotations, load_vocab, PAD_IDX, UNK_IDX, IDX2TAG
from src.train import load_model_checkpoint

def predict_sentence_tags(
    model: torch.nn.Module,
    tokens: List[str],
    word2idx: Dict[str, int],
    device: str = "cpu"
) -> List[str]:
    """
    Runs model inference on a tokenized sentence and returns predicted BIO tags.
    """
    model.eval()
    token_ids = [word2idx.get(t, UNK_IDX) for t in tokens]
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor)  # (1, seq_len, tagset_size)
        preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()  # (seq_len)

    pred_tags = [IDX2TAG.get(idx, "O") for idx in preds]
    return pred_tags

def evaluate_test_set(
    model_path: str = "models/ner_bilstm.pt",
    vocab_path: str = "data/vocab.json",
    data_path: str = "data/annotations.json"
) -> Dict[str, Any]:
    """
    Evaluates model on the test sentences and prints detailed comparison.
    """
    word2idx, _ = load_vocab(vocab_path)
    model = load_model_checkpoint(model_path)
    data = load_annotations(data_path)
    test_items = data.get("test", [])

    all_true_tags: List[List[str]] = []
    all_pred_tags: List[List[str]] = []
    comparison_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print(" 📊 DETAILED TOKEN-BY-TOKEN TEST EVALUATION")
    print("=" * 70)

    for item in test_items:
        tokens = item["tokens"]
        true_tags = item["tags"]
        pred_tags = predict_sentence_tags(model, tokens, word2idx)

        all_true_tags.append(true_tags)
        all_pred_tags.append(pred_tags)

        print(f"\nSentence ID: {item['id']}")
        print(f"Full Text:   \"{item['text']}\"")

        table_rows = []
        for i, (tok, t_tag, p_tag) in enumerate(zip(tokens, true_tags, pred_tags)):
            is_oov = tok not in word2idx
            match = "MATCH" if t_tag == p_tag else "MISMATCH"
            status = f"{tok} (OOV)" if is_oov else tok
            table_rows.append([i + 1, status, t_tag, p_tag, match])

        print(tabulate(
            table_rows,
            headers=["#", "Token (Status)", "Actual Tag", "Predicted Tag", "Result"],
            tablefmt="grid"
        ))

        comparison_records.append({
            "id": item["id"],
            "tokens": tokens,
            "true_tags": true_tags,
            "pred_tags": pred_tags
        })

    # Chunk-level Sequence Evaluation Report
    report = classification_report(all_true_tags, all_pred_tags, digits=4)
    overall_f1 = f1_score(all_true_tags, all_pred_tags)
    overall_prec = precision_score(all_true_tags, all_pred_tags)
    overall_rec = recall_score(all_true_tags, all_pred_tags)

    print("\n" + "=" * 70)
    print(" 🏆 SEQEVAL CHUNK-LEVEL METRICS (Standard CoNLL Evaluation)")
    print("=" * 70)
    print(report)
    print(f"Overall Precision : {overall_prec:.4f}")
    print(f"Overall Recall    : {overall_rec:.4f}")
    print(f"Overall F1-Score  : {overall_f1:.4f}")

    return {
        "report": report,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1": overall_f1,
        "records": comparison_records
    }

if __name__ == "__main__":
    evaluate_test_set()

