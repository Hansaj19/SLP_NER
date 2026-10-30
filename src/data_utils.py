"""
Data utilities for Tokenization, BIO Tagging Alignment, and Vocabulary Management.
"""
import re
import json
import os
from typing import List, Dict, Tuple, Any

TAGS = ["<PAD>", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
TAG2IDX = {tag: i for i, tag in enumerate(TAGS)}
IDX2TAG = {i: tag for tag, i in TAG2IDX.items()}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

def tokenize_text(text: str) -> List[str]:
    """
    Splits text into words and punctuation tokens.
    Handles abbreviations like 'Dr.' gracefully while splitting trailing punctuation.
    """
    pattern = r"[\w']+|[.,!?;]"
    tokens = re.findall(pattern, text)
    return tokens

def load_annotations(file_path: str = "data/annotations.json") -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads annotated train/test dataset from JSON.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_vocab(train_data: List[Dict[str, Any]], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds word-to-index and index-to-word vocabularies from training data.
    Special tokens:
      <PAD>: 0
      <UNK>: 1
    """
    word_counts: Dict[str, int] = {}
    for item in train_data:
        for token in item["tokens"]:
            word_counts[token] = word_counts.get(token, 0) + 1

    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, count in word_counts.items():
        if count >= min_freq and word not in word2idx:
            word2idx[word] = len(word2idx)

    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

def save_vocab(word2idx: Dict[str, int], path: str = "data/vocab.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "word2idx": word2idx,
            "tag2idx": TAG2IDX
        }, f, indent=2)

def load_vocab(path: str = "data/vocab.json") -> Tuple[Dict[str, int], Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["word2idx"], payload["tag2idx"]
