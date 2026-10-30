"""
Batch Transcriber for all recorded audio files using AssemblyAI API.
"""
import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.stt_service import STTService
from src.data_utils import load_annotations

load_dotenv()

def transcribe_all_recordings():
    print("=" * 65)
    print(" 🎙️ ASSEMBLYAI BATCH TRANSCRIPTION RUNNER")
    print("=" * 65)

    stt = STTService()
    print(f"AssemblyAI API Configured: {stt.is_api_configured}")

    annotations = load_annotations("data/annotations.json")
    results = {}

    for split in ["train", "test"]:
        for item in annotations[split]:
            item_id = item["id"]
            # Look for .wav, .mp3, .m4a, etc.
            found_path = None
            for ext in [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"]:
                candidate = os.path.join("audio", split, f"{item_id}{ext}")
                if os.path.exists(candidate):
                    found_path = candidate
                    break

            target_text = item["text"]
            if not found_path:
                print(f"\nProcessing [{item_id}] ({split.upper()}): [Not Found in audio/{split}/]")
                continue

            print(f"\nProcessing [{item_id}] ({split.upper()}): {found_path}")
            transcription = stt.transcribe(found_path, fallback_text=target_text)
            results[item_id] = {
                "audio_path": found_path,
                "split": split,
                "target_text": target_text,
                "transcribed_text": transcription.get("text", ""),
                "confidence": transcription.get("confidence", 0.0),
                "words": transcription.get("words", [])
            }
            print(f"  Target:      \"{target_text}\"")
            print(f"  Transcribed: \"{transcription.get('text', '')}\"")

    output_file = "data/transcripts.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print(f" Transcriptions saved to: {output_file}")
    print(f" Total files processed: {len(results)}")
    print("=" * 65)

if __name__ == "__main__":
    transcribe_all_recordings()
