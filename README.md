# Named Entity Recognition (NER) System for Voice Input

An end-to-end Speech-to-NER pipeline using **AssemblyAI Speech-to-Text** and a **PyTorch Recurrent Neural Network (SimpleRNN / GRU / BiLSTM)** trained and evaluated on a custom voice-recorded dataset.

---

## 📁 Submission & Directory Structure for Google Drive

When uploading to Google Drive for evaluation, maintain this folder structure:

```
SLP-2/
├── audio/
│   ├── train/                 # 10 voice recordings (train_01.wav to train_10.wav)
│   └── test/                  # 2 voice recordings (test_01.wav and test_02.wav)
├── data/
│   ├── annotations.json       # Ground truth BIO dataset (PER, ORG, LOC)
│   ├── transcripts.json       # AssemblyAI speech-to-text transcriptions
│   └── vocab.json             # Vocabulary index mapping
├── models/
│   ├── ner_bilstm.pt          # Trained Bidirectional LSTM checkpoint
│   ├── ner_gru.pt             # Trained GRU checkpoint
│   └── ner_rnn.pt             # Trained SimpleRNN checkpoint
├── src/
│   ├── dataset.py             # PyTorch Dataset & padding collate function
│   ├── model.py               # SimpleRNN / GRU / BiLSTM architectures
│   ├── train.py               # Model training loop
│   ├── evaluate.py            # Token and entity-level evaluation (Seqeval)
│   ├── stt_service.py         # AssemblyAI API client
│   ├── recorder.py            # Interactive CLI voice recording tool
│   ├── transcribe_all.py      # Batch audio transcriber
│   └── pipeline.py            # End-to-end Audio -> STT -> PyTorch -> NER
├── docs/
│   ├── sentences.md           # The 10 training + 2 test sentence scripts
│   ├── output_log.md          # Phased test execution logs with simple explanations
│   └── analysis_report.md     # In-depth analysis of small audio dataset challenges
├── tests/                     # Automated test suites for Phases 1 to 4
├── pyproject.toml             # UV dependency manifest
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

Using `uv`:

```bash
uv sync
```

*(Or with pip: `pip install torch assemblyai seqeval scikit-learn numpy python-dotenv pyaudio websocket-client tabulate`)*

### 2. Configure AssemblyAI API Key

Add your API key into `.env`:

```ini
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

### 3. Record Audio Recordings

You can record all 10 training sentences and 2 test sentences interactively using our built-in voice recorder tool:

```bash
uv run python src/recorder.py
```

*(Or drop your pre-recorded `.wav` files into `audio/train/` and `audio/test/`)*.

### 4. Transcribe Audio Files

Transcribe all audio recordings using AssemblyAI:

```bash
uv run python src/transcribe_all.py
```

### 5. Train the PyTorch RNN Model

Train the Bidirectional LSTM model on the annotated dataset:

```bash
uv run python src/train.py
```

### 6. Evaluate Model on Test Recordings

Run the token-by-token and entity-level (Seqeval) evaluation:

```bash
uv run python src/evaluate.py
```

### 7. Run End-to-End Pipeline

Run complete audio inference on any recording:

```bash
uv run python src/pipeline.py
```

---

## 🧪 Phased Test Suites

- **Phase 1 Test:** `uv run python tests/test_phase1_env_and_structure.py`
- **Phase 2 Test:** `uv run python tests/test_phase2_stt_and_data.py`
- **Phase 3 Test:** `uv run python tests/test_phase3_model_training.py`
- **Phase 4 Test:** `uv run python tests/test_phase4_end_to_end.py`

All test results and clear, simple explanations are logged in [output_log.md](file:///e:/SLP-2/docs/output_log.md).
