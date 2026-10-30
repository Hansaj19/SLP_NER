"""
Interactive Audio Recorder and AssemblyAI Transcriber.
Guides you through recording each sentence, streaming/saving audio to WAV,
and transcribing via AssemblyAI.
"""
import os
import sys
import wave
import json
import time
from typing import Dict, Any, List
try:
    import pyaudio
except ImportError:
    pyaudio = None
import assemblyai as aai
from dotenv import load_dotenv

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_utils import load_annotations

load_dotenv()
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
aai.settings.api_key = API_KEY

SAMPLE_RATE = 16000
CHANNELS = 1
FRAMES_PER_BUFFER = 800  # 50ms chunk
AUDIO_FORMAT = pyaudio.paInt16 if pyaudio else None

def record_audio_clip(output_wav_path: str, duration_seconds: int = 5) -> str:
    """
    Records audio from the microphone for a fixed duration (or until interrupted)
    and saves it to a 16-bit 16kHz mono WAV file.
    """
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    p = pyaudio.PyAudio()

    print(f"\n[Microphone] Preparing to record for ~{duration_seconds} seconds...")
    print(">>> 3...")
    time.sleep(0.7)
    print(">>> 2...")
    time.sleep(0.7)
    print(">>> 1... SPEAK NOW! 🎙️")

    stream = p.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    frames = []
    total_frames = int(SAMPLE_RATE / FRAMES_PER_BUFFER * duration_seconds)

    try:
        for i in range(total_frames):
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            frames.append(data)
            # Simple progress bar
            if i % 10 == 0:
                print(".", end="", flush=True)
        print(" [Recording Complete!]")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    # Save to WAV
    with wave.open(output_wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    print(f"[Saved] Audio written to: {output_wav_path}")
    return output_wav_path

def transcribe_audio_file(audio_path: str) -> Dict[str, Any]:
    """
    Sends audio file to AssemblyAI for high-accuracy transcription.
    """
    print(f"[AssemblyAI] Transcribing {audio_path}...")
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(punctuate=True, format_text=True, language_code="en")
    transcript = transcriber.transcribe(audio_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"[Error] Transcription failed: {transcript.error}")
        return {"text": "", "error": transcript.error, "confidence": 0.0}

    print(f"[Result] Transcribed: \"{transcript.text}\" (Confidence: {transcript.confidence:.2%})")
    return {
        "text": transcript.text,
        "confidence": transcript.confidence,
        "words": [
            {
                "text": w.text,
                "start": w.start,
                "end": w.end,
                "confidence": w.confidence
            }
            for w in (transcript.words or [])
        ]
    }

def update_transcripts_log(item_id: str, audio_path: str, target_script: str, result: Dict[str, Any]):
    log_file = "data/transcripts.json"
    data = {}
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    data[item_id] = {
        "audio_path": audio_path,
        "target_script": target_script,
        "transcribed_text": result.get("text", ""),
        "confidence": result.get("confidence", 0.0),
        "words": result.get("words", [])
    }

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Logged] Transcript updated in {log_file}")

def main():
    print("=" * 65)
    print(" 🎙️ VOICE RECORDER & ASSEMBLYAI TRANSCRIBER FOR NER PROJECT")
    print("=" * 65)

    annotations = load_annotations("data/annotations.json")
    all_items = [(item, "train") for item in annotations["train"]] + [(item, "test") for item in annotations["test"]]

    print("\nAvailable items to record:")
    for idx, (item, split) in enumerate(all_items, 1):
        target_path = f"audio/{split}/{item['id']}.wav"
        status = "[Already Exists]" if os.path.exists(target_path) else "[Not Recorded]"
        print(f"  {idx:2d}. [{item['id']}] ({split.upper()}) {status}")
        print(f"      Text: \"{item['text']}\"")

    print("\nOptions:")
    print("  Enter item number (1-12) to record a specific sentence.")
    print("  Enter 'all' to record all unrecorded sentences in sequence.")
    print("  Enter 'q' to quit.")

    choice = input("\nYour choice: ").strip()
    if choice.lower() == "q":
        return

    items_to_record = []
    if choice.lower() == "all":
        items_to_record = all_items
    elif choice.isdigit() and 1 <= int(choice) <= len(all_items):
        items_to_record = [all_items[int(choice) - 1]]
    else:
        print("[!] Invalid option. Exiting.")
        return

    for item, split in items_to_record:
        item_id = item["id"]
        target_path = f"audio/{split}/{item_id}.wav"
        target_script = item["text"]

        print("\n" + "-" * 60)
        print(f"RECORDING FOR: [{item_id}] ({split.upper()})")
        print(f"Please read aloud clearly:")
        print(f"👉 \"{target_script}\"")
        print("-" * 60)

        # Estimate duration: ~0.5s per word + 1.5s margin
        est_duration = max(4, int(len(item["tokens"]) * 0.6) + 2)
        confirm = input(f"Press Enter when ready to record (~{est_duration}s), or 's' to skip: ").strip()
        if confirm.lower() == 's':
            continue

        record_audio_clip(target_path, duration_seconds=est_duration)
        trans_res = transcribe_audio_file(target_path)
        update_transcripts_log(item_id, target_path, target_script, trans_res)

    print("\n" + "=" * 65)
    print(" Recording session finished! Check data/transcripts.json")
    print("=" * 65)

if __name__ == "__main__":
    main()

