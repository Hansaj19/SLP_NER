"""
Phase 4 Test: Full End-to-End Pipeline, Model Comparative Evaluation, and Analysis.
"""
import os
import sys

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import run_training_pipeline
from src.evaluate import evaluate_test_set
from src.pipeline import run_audio_ner

def test_train_all_variants():
    print("[1/3] Training and checkpointing SimpleRNN, GRU, and BiLSTM variants...")
    architectures = [
        ("rnn", "models/ner_rnn.pt"),
        ("gru", "models/ner_gru.pt"),
        ("lstm", "models/ner_bilstm.pt")
    ]
    for cell, path in architectures:
        model, losses = run_training_pipeline(cell_type=cell, num_epochs=45, save_path=path)
        assert os.path.exists(path), f"Failed to save model checkpoint to {path}"
        assert losses[-1] < losses[0], f"Loss did not decrease for {cell}"
        print(f"  [OK] Successfully trained and saved {cell.upper()} (Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f})")

def test_evaluation_metrics():
    print("\n[2/3] Running detailed sequence evaluation against test annotations...")
    eval_results = evaluate_test_set(model_path="models/ner_bilstm.pt")
    assert "report" in eval_results
    assert len(eval_results["records"]) == 2
    print("  [OK] Evaluation completed with full token-by-token comparison.")

def test_pipeline_execution():
    print("\n[3/3] Testing complete Audio -> STT -> BiLSTM -> Entity Pipeline...")
    res1 = run_audio_ner(
        audio_path="audio/test/test_01.wav",
        model_path="models/ner_bilstm.pt",
        fallback_text="Sundar Pichai discussed technology with Tesla in California."
    )
    assert len(res1["tokens"]) > 0
    assert len(res1["predicted_tags"]) == len(res1["tokens"])
    print(f"  [OK] Test 1 Pipeline Output:")
    print(f"       Transcribed: \"{res1['transcribed_text']}\"")
    print(f"       Extracted Entities: {res1['extracted_entities']}")

    res2 = run_audio_ner(
        audio_path="audio/test/test_02.wav",
        model_path="models/ner_bilstm.pt",
        fallback_text="Dr. Kalam delivered an inspiring lecture at Oxford University in England."
    )
    assert len(res2["tokens"]) > 0
    assert len(res2["predicted_tags"]) == len(res2["tokens"])
    print(f"  [OK] Test 2 Pipeline Output:")
    print(f"       Transcribed: \"{res2['transcribed_text']}\"")
    print(f"       Extracted Entities: {res2['extracted_entities']}")

if __name__ == "__main__":
    print("=" * 65)
    print(" RUNNING PHASE 4 TEST: END-TO-END PIPELINE & EVALUATION")
    print("=" * 65)
    test_train_all_variants()
    test_evaluation_metrics()
    test_pipeline_execution()
    print("\n" + "=" * 65)
    print(" ALL PHASE 4 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65)

