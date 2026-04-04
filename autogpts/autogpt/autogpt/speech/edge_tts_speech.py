"""Edge TTS voice provider — uses Microsoft Edge's free neural TTS (no API key)."""
from __future__ import annotations

import asyncio
import subprocess
import tempfile
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.config import Config

from autogpt.speech.base import VoiceBase

# Default to Andrew (en-US, natural male voice)
DEFAULT_VOICE = "en-US-AndrewNeural"


class EdgeTTSSpeech(VoiceBase):
    """Text-to-speech via Microsoft Edge TTS (edge-tts package)."""

    def _setup(self, config: Config) -> None:
        # Voice can be overridden via EDGE_TTS_VOICE in the env/config
        self._voice = getattr(config, "edge_tts_voice", None) or DEFAULT_VOICE

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        try:
            asyncio.run(self._synthesize_and_play(text))
            return True
        except Exception as e:
            print(f"EdgeTTS error: {e}")
            return False

    async def _synthesize_and_play(self, text: str) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text, self._voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        try:
            await communicate.save(tmp_path)
            subprocess.run(
                ["mpv", "--really-quiet", tmp_path],
                check=True,
                timeout=60,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
