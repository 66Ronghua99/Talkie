"""LiveKit TTS Plugin for RealTalk.

Provides text-to-speech synthesis using an OpenAI compatible API (so it can be flexibly overridden).
"""

import os
from livekit.plugins import openai
from ..logging_config import setup_logger

logger = setup_logger("talkie.livekit_plugins.tts")

async def create_tts_plugin() -> openai.TTS:
    """Factory function to create a compatible TTS plugin."""
    
    # We fallback to standard OpenAI as OpenRouter doesn't provide TTS
    api_key = os.getenv("TTS_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    base_url = os.getenv("TTS_BASE_URL", "https://api.openai.com/v1")
    voice = os.getenv("TTS_VOICE", "alloy")
    model = os.getenv("TTS_MODEL", "tts-1")
    
    logger.info(f"Initializing TTS plugin with base_url: {base_url} and model: {model}")
    
    plugin = openai.TTS(
        api_key=api_key,
        base_url=base_url,
        model=model,
        voice=voice,
    )
    return plugin
