"""
Phase 1 Test: Environment, Directory Structure, and Dataset Integrity Verification.
"""
import os
import sys
import json

VALID_TAGS = {"B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"}
REQUIRED_DIRS = [
    "audio/train",
    "audio/test",
    "data",
    "src",
    "models",
    "tests",
    "docs"
]

def test_directories():
    print("[1/4] Checking required directory structure...")
    for d in REQUIRED_DIRS:
        assert os.path.exists(d), f"Missing directory: {d}"
        print(f"  [OK] Directory exists: {d}")

def test_imports():
    print("\n[2/4] Checking core Python libraries...")
    try:
        import torch
        print(f"  [OK] PyTorch version: {torch.__version__}")
    except ImportError as e:
        raise AssertionError(f"PyTorch is not installed or importable: {e}")

    try:
        import assemblyai as aai
        print(f"  [OK] AssemblyAI SDK version: {getattr(aai, '__version__', 'installed')}")
    except ImportError as e:
        raise AssertionError(f"AssemblyAI SDK is not installed: {e}")

    try:
        import seqeval
        print(f"  [OK] Seqeval: installed and ready")
    except ImportError as e:
        raise AssertionError(f"Seqeval is not installed: {e}")

    try:
        import sklearn
        print(f"  [OK] Scikit-Learn version: {sklearn.__version__}")
    except ImportError as e:
        raise AssertionError(f"Scikit-Learn is not installed: {e}")

    try:
        import numpy as np
        print(f"  [OK] NumPy version: {np.__version__}")
    except ImportError as e:
        raise AssertionError(f"NumPy is not installed: {e}")

def test_annotation_integrity():
    print("\n[3/4] Verifying annotations.json integrity...")
    annot_path = os.path.join("data", "annotations.json")
    assert os.path.exists(annot_path), f"Missing annotations file: {annot_path}"

    with open(annot_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "train" in data, "Missing 'train' partition in annotations.json"
    assert "test" in data, "Missing 'test' partition in annotations.json"
    assert len(data["train"]) == 10, f"Expected 10 training sentences, got {len(data['train'])}"
    assert len(data["test"]) >= 2, f"Expected at least 2 test sentences, got {len(data['test'])}"

    for split_name in ["train", "test"]:
        for item in data[split_name]:
            item_id = item.get("id")
            tokens = item.get("tokens", [])
            tags = item.get("tags", [])
            assert len(tokens) == len(tags), (
                f"Mismatch in {item_id}: {len(tokens)} tokens vs {len(tags)} tags."
            )
            for tag in tags:
                assert tag in VALID_TAGS, f"Invalid tag '{tag}' found in {item_id}"
            
            non_o_tags = [t for t in tags if t != "O"]
            assert len(non_o_tags) > 0, f"Sentence {item_id} has no named entities!"
            print(f"  [OK] {split_name} sample '{item_id}': {len(tokens)} tokens, {len(non_o_tags)} entity tokens.")

def test_sentences_doc():
    print("\n[4/4] Verifying sentences documentation...")
    sentences_path = os.path.join("docs", "sentences.md")
    assert os.path.exists(sentences_path), f"Missing {sentences_path}"
    with open(sentences_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Training Sentences" in content and "Test Sentences" in content
    print("  [OK] docs/sentences.md is properly formatted and accessible.")

if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING PHASE 1 TEST: ENVIRONMENT, STRUCTURE & DATA INTEGRITY")
    print("=" * 60)
    test_directories()
    test_imports()
    test_annotation_integrity()
    test_sentences_doc()
    print("\n" + "=" * 60)
    print(" ALL PHASE 1 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
