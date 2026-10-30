"""
End-to-End Pipeline: Audio -> AssemblyAI Speech-to-Text -> PyTorch RNN -> BIO Named Entities.
"""
import os
import sys
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.stt_service import STTService
from src.data_utils import tokenize_text, load_vocab
from src.train import load_model_checkpoint
from src.evaluate import predict_sentence_tags

def extract_entities_from_bio(tokens: List[str], tags: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts named entities from a token sequence and corresponding BIO tags.
    """
    entities = []
    current_entity = None

    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            ent_type = tag.split("-")[1]
            current_entity = {"entity": tok, "type": ent_type}
        elif tag.startswith("I-") and current_entity:
            ent_type = tag.split("-")[1]
            if current_entity["type"] == ent_type:
                current_entity["entity"] += " " + tok
            else:
                entities.append(current_entity)
                current_entity = {"entity": tok, "type": ent_type}
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities

def find_audio_path(base_path: str) -> str:
    """Finds existing file among possible extensions (.wav, .mp3, etc.)."""
    if os.path.exists(base_path):
        return base_path
    base, _ = os.path.splitext(base_path)
    for ext in [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"]:
        if os.path.exists(base + ext):
            return base + ext
    return base_path

def run_audio_ner(
    audio_path: str,
    model_path: str = "models/ner_bilstm.pt",
    vocab_path: str = "data/vocab.json",
    fallback_text: str = None
) -> Dict[str, Any]:
    """
    Complete end-to-end execution:
      1. Transcribe audio with AssemblyAI
      2. Tokenize transcribed text
      3. Predict BIO tags with PyTorch model
      4. Extract structured named entities
    """
    audio_path = find_audio_path(audio_path)
    stt = STTService()
    stt_result = stt.transcribe(audio_path, fallback_text=fallback_text)
    transcribed_text = stt_result.get("text", "")

    tokens = tokenize_text(transcribed_text)
    word2idx, _ = load_vocab(vocab_path)
    model = load_model_checkpoint(model_path)

    pred_tags = predict_sentence_tags(model, tokens, word2idx)
    entities = extract_entities_from_bio(tokens, pred_tags)

    return {
        "audio_path": audio_path,
        "transcribed_text": transcribed_text,
        "tokens": tokens,
        "predicted_tags": pred_tags,
        "extracted_entities": entities,
        "stt_confidence": stt_result.get("confidence", 1.0)
    }

if __name__ == "__main__":
    test_wav = "audio/test/test_01.wav"
    result = run_audio_ner(test_wav, fallback_text="Sundar Pichai discussed technology with Tesla in California.")
    print("\n--- End-to-End Pipeline Output ---")
    print(f"Text: {result['transcribed_text']}")
    print(f"Entities: {result['extracted_entities']}")
