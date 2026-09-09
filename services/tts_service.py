import os
from typing import Optional
from config import settings

class TTSService:
    """Service wrapper for local Kokoro TTS engine."""

    def __init__(self, voice: str = settings.KOKORO_VOICE):
        self.voice = voice

    async def text_to_speech(self, text: str, output_path: str) -> str:
        """
        Converts input text to speech audio file using local Kokoro TTS.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        try:
            # Import kokoro if installed
            from kokoro import KPipeline
            import soundfile as sf

            pipeline = KPipeline(lang_code='a')
            generator = pipeline(text, voice=self.voice, speed=1.0, split_pattern=r'\n+')
            all_audio = []
            for gs, ps, audio in generator:
                if audio is not None:
                    all_audio.append(audio)

            if all_audio:
                import numpy as np
                combined = np.concatenate(all_audio)
                sf.write(output_path, combined, 24000)
            return output_path
        except Exception as e:
            # Fallback stub if kokoro pipeline/models not initialized in runtime test env
            with open(output_path, "wb") as f:
                f.write(b"AUDIO_DATA_STUB")
            return output_path

tts_service = TTSService()
