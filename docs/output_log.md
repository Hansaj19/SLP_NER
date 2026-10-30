# Project Execution and Test Output Log

This document records the output of every phase's test suite, accompanied by simple, clear explanations of what the results mean, why each component was designed this way, and how it aligns with the evaluator's requirements.

---


## Phase 1: Environment, Directory Structure & Dataset Integrity

### Test Run Command
```bash
uv run python tests/test_phase1_env_and_structure.py
```

### Raw Test Output
```text
============================================================
 RUNNING PHASE 1 TEST: ENVIRONMENT, STRUCTURE & DATA INTEGRITY
============================================================
[1/4] Checking required directory structure...
  [OK] Directory exists: audio/train
  [OK] Directory exists: audio/test
  [OK] Directory exists: data
  [OK] Directory exists: src
  [OK] Directory exists: models
  [OK] Directory exists: tests
  [OK] Directory exists: docs

[2/4] Checking core Python libraries...
  [OK] PyTorch version: 2.15.0+cpu
  [OK] AssemblyAI SDK version: 1.8.2
  [OK] Seqeval: installed and ready
  [OK] Scikit-Learn version: 1.9.1
  [OK] NumPy version: 2.5.4

[3/4] Verifying annotations.json integrity...
  [OK] train sample 'train_01': 7 tokens, 4 entity tokens.
  [OK] train sample 'train_02': 7 tokens, 4 entity tokens.
  [OK] train sample 'train_03': 8 tokens, 4 entity tokens.
  [OK] train sample 'train_04': 9 tokens, 5 entity tokens.
  [OK] train sample 'train_05': 7 tokens, 4 entity tokens.
  [OK] train sample 'train_06': 11 tokens, 5 entity tokens.
  [OK] train sample 'train_07': 8 tokens, 5 entity tokens.
  [OK] train sample 'train_08': 8 tokens, 5 entity tokens.
  [OK] train sample 'train_09': 8 tokens, 4 entity tokens.
  [OK] train sample 'train_10': 10 tokens, 4 entity tokens.
  [OK] test sample 'test_01': 9 tokens, 4 entity tokens.
  [OK] test sample 'test_02': 12 tokens, 4 entity tokens.

[4/4] Verifying sentences documentation...
  [OK] docs/sentences.md is properly formatted and accessible.

============================================================
 ALL PHASE 1 CHECKS PASSED SUCCESSFULLY!
============================================================
```

### Simple Explanation of Results:
1. **Directory Structure Validated:**
   - Folders are organized cleanly. The `audio/` directory has dedicated subfolders for training (`audio/train`) and testing (`audio/test`), which can directly be uploaded to Google Drive as requested by the evaluator.
2. **Dependencies Successfully Configured:**
   - **PyTorch (`2.15.0+cpu`)**: The deep learning framework we will use to build and train our RNN/LSTM sequence labeling models.
   - **AssemblyAI (`1.8.2`)**: The speech-to-text SDK that transcribes voice recordings into text tokens.
   - **Seqeval & Scikit-Learn**: Standard NLP evaluation libraries for measuring precision, recall, and F1-score across BIO chunks.
3. **Dataset Integrity Verified:**
   - Exactly **10 training sentences** and **2 test sentences** are prepared.
   - Every token maps 1-to-1 to a valid BIO label (`B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, or `O`).
   - Every single sentence contains at least one named entity, fulfilling the evaluator's explicit prompt constraint.

---

## Phase 2: Speech-to-Text (AssemblyAI) & BIO Data Preprocessing

### Test Run Command
```bash
uv run python tests/test_phase2_stt_and_data.py
```

### Raw Test Output
```text
============================================================
 RUNNING PHASE 2 TEST: STT & BIO PREPROCESSING PIPELINE
============================================================
[1/4] Testing Speech-to-Text (STT) service and fallback...
[STT] Running in mock/fallback mode for: audio/test/test_01.wav
  [OK] Transcription source: mock_fallback
  [OK] Transcribed text: 'Sundar Pichai discussed technology with Tesla in California.'
  [OK] Extracted 9 word-level timestamps/confidences.

[2/4] Testing Tokenizer on complex punctuation...
  [OK] Tokenized sample into 13 tokens: ['Dr', '.', 'Kalam', 'delivered', 'an', 'inspiring', 'lecture', 'at', 'Oxford', 'University', 'in', 'England', '.']

[3/4] Testing Vocabulary builder & serialization...
  [OK] Vocab size: 69 unique tokens (including PAD & UNK).
  [OK] Tagset size: 8 tags.

[4/4] Testing Out-Of-Vocabulary (OOV) mapping...
  [OK] Unseen test token 'Tesla' properly mapped to <UNK> (Index 1).

============================================================
 ALL PHASE 2 CHECKS PASSED SUCCESSFULLY!
============================================================
```

### Simple Explanation of Results:
1. **Speech-to-Text (AssemblyAI) Integration:**
   - The STT service (`src/stt_service.py`) connects to AssemblyAI when an API key is provided, returning spoken text together with confidence scores and word-level timestamps.
   - It also features a fallback test harness so pipelines can execute deterministically even before live recordings are uploaded or if network is offline.
2. **Tokenizer & Punctuation Isolation:**
   - Standard speech transcripts often glue punctuation directly to words (e.g., `"California."` or `"Dr."`). The tokenizer cleanly isolates words and punctuation marks as distinct tokens, preventing punctuation from corrupting entity embeddings.
3. **Vocabulary & Special Tokens (`<PAD>` and `<UNK>`):**
   - The vocabulary consists of 69 distinct tokens extracted from the 10 training sentences.
   - Special token `<PAD>` (index `0`): Used to pad sentences to equal length so they can be processed in batches by PyTorch.
   - Special token `<UNK>` (index `1`): Out-Of-Vocabulary token. Any word seen during testing that wasn't in the 10 training sentences (such as `"Tesla"`) is mapped to `<UNK>`.
4. **BIO Tag Mapping:**
   - 8 total tag classes: `<PAD>` (0), `O` (1), `B-PER` (2), `I-PER` (3), `B-ORG` (4), `I-ORG` (5), `B-LOC` (6), `I-LOC` (7).

---
