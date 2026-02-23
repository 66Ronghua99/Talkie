"""LiveKit LLM Plugin for RealTalk using OpenRouter.

Provides chat completion with OpenRouter.
"""
import os
from livekit.plugins import openai
from ..logging_config import setup_logger

logger = setup_logger("talkie.livekit_plugins.llm")

async def create_llm_plugin() -> openai.LLM:
    """Factory function to create OpenRouter LLM plugin."""
    
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    
    logger.info(f"Initializing OpenRouter LLM plugin with model: {model}")
    
    plugin = openai.LLM(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    return plugin

