"""
Phase 2 Test: Speech-to-Text Pipeline & BIO Preprocessing Engine.
"""
import os
import sys
import json

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stt_service import STTService
from src.data_utils import (
    tokenize_text,
    load_annotations,
    build_vocab,
    save_vocab,
    load_vocab,
    TAG2IDX,
    IDX2TAG,
    PAD_IDX,
    UNK_IDX
)

def test_stt_service():
    print("[1/4] Testing Speech-to-Text (STT) service and fallback...")
    stt = STTService()
    test_audio_path = "audio/test/test_01.wav"
    result = stt.transcribe(
        audio_path=test_audio_path,
        fallback_text="Sundar Pichai discussed technology with Tesla in California."
    )

    assert "text" in result and isinstance(result["text"], str)
    assert "words" in result and isinstance(result["words"], list)
    assert len(result["words"]) > 0
    assert "confidence" in result
    print(f"  [OK] Transcription source: {result.get('source')}")
    print(f"  [OK] Transcribed text: '{result['text']}'")
    print(f"  [OK] Extracted {len(result['words'])} word-level timestamps/confidences.")

def test_tokenizer():
    print("\n[2/4] Testing Tokenizer on complex punctuation...")
    sample_text = "Dr. Kalam delivered an inspiring lecture at Oxford University in England."
    tokens = tokenize_text(sample_text)
    expected_start = ["Dr", ".", "Kalam", "delivered"]
    assert tokens[:4] == expected_start, f"Unexpected tokenization: {tokens}"
    assert tokens[-1] == ".", f"Trailing punctuation not isolated: {tokens[-1]}"
    print(f"  [OK] Tokenized sample into {len(tokens)} tokens: {tokens}")

def test_vocab_and_serialization():
    print("\n[3/4] Testing Vocabulary builder & serialization...")
    annotations = load_annotations("data/annotations.json")
    train_data = annotations["train"]

    word2idx, idx2word = build_vocab(train_data)
    assert word2idx["<PAD>"] == PAD_IDX
    assert word2idx["<UNK>"] == UNK_IDX
    assert "Elon" in word2idx
    assert "California" in word2idx

    vocab_file = "data/vocab.json"
    save_vocab(word2idx, vocab_file)
    assert os.path.exists(vocab_file)

    loaded_w2i, loaded_t2i = load_vocab(vocab_file)
    assert loaded_w2i["Elon"] == word2idx["Elon"]
    assert loaded_t2i["B-PER"] == TAG2IDX["B-PER"]
    print(f"  [OK] Vocab size: {len(word2idx)} unique tokens (including PAD & UNK).")
    print(f"  [OK] Tagset size: {len(TAG2IDX)} tags.")

def test_oov_encoding():
    print("\n[4/4] Testing Out-Of-Vocabulary (OOV) mapping...")
    loaded_w2i, _ = load_vocab("data/vocab.json")
    known_word = "Tesla"  # Notice Tesla is only in test set, not train set!
    encoded = loaded_w2i.get(known_word, UNK_IDX)
    assert encoded == UNK_IDX, "Unseen test token should correctly map to <UNK> (1)"
    print(f"  [OK] Unseen test token '{known_word}' properly mapped to <UNK> (Index {UNK_IDX}).")

if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING PHASE 2 TEST: STT & BIO PREPROCESSING PIPELINE")
    print("=" * 60)
    test_stt_service()
    test_tokenizer()
    test_vocab_and_serialization()
    test_oov_encoding()
    print("\n" + "=" * 60)
    print(" ALL PHASE 2 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
