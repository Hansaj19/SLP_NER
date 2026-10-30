"""
Speech-to-Text Service using AssemblyAI SDK.
Supports cloud API transcription and local mock fallback for deterministic testing.
"""
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class STTService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            try:
                import streamlit as st
                if "ASSEMBLYAI_API_KEY" in st.secrets:
                    self.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
            except Exception:
                pass
        self._aai = None
        if self.api_key and self.api_key != "your_assemblyai_api_key_here":
            try:
                import assemblyai as aai
                aai.settings.api_key = self.api_key
                self._aai = aai
            except Exception as e:
                print(f"[Warning] Could not initialize AssemblyAI SDK: {e}")

    @property
    def is_api_configured(self) -> bool:
        return self._aai is not None

    def transcribe(self, audio_path: str, fallback_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribes an audio file. If AssemblyAI API key is present and audio exists,
        calls the AssemblyAI cloud API. Otherwise, uses fallback text or mock transcription.
        """
        if self.is_api_configured and os.path.exists(audio_path):
            print(f"[STT] Transcribing via AssemblyAI API: {audio_path}")
            transcriber = self._aai.Transcriber()
            # Explicitly force English language code ('en') to avoid auto-detecting Hindi/Devanagari
            config = self._aai.TranscriptionConfig(
                punctuate=True,
                format_text=True,
                language_code="en"
            )
            transcript = transcriber.transcribe(audio_path, config=config)

            if transcript.status == self._aai.TranscriptStatus.error:
                raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

            words = [
                {
                    "text": w.text,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence
                }
                for w in (transcript.words or [])
            ]
            return {
                "text": transcript.text,
                "words": words,
                "confidence": transcript.confidence if hasattr(transcript, "confidence") else 1.0,
                "source": "assemblyai_api"
            }

        # Fallback / Testing Mode
        print(f"[STT] Running in mock/fallback mode for: {audio_path}")
        text = fallback_text or "Sundar Pichai discussed technology with Tesla in California."
        tokens = text.replace(".", " .").replace(",", " ,").split()
        mock_words = [
            {"text": t, "start": i * 500, "end": (i + 1) * 500, "confidence": 0.98}
            for i, t in enumerate(tokens)
        ]
        return {
            "text": text,
            "words": mock_words,
            "confidence": 0.98,
            "source": "mock_fallback"
        }
