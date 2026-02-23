"""Base classes for LiveKit Agent Plugins.

Defines the common interface that all RealTalk plugins must implement
to be compatible with the LiveKit Agent framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..logging_config import setup_logger

logger = setup_logger("realtalk.livekit_plugins")


class BaseRealTalkPlugin(ABC):
    """Base class for all RealTalk LiveKit plugins.

    Provides common lifecycle management and configuration interface
    for ASR, LLM, and TTS plugins.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize the plugin.

        Args:
            config: Optional configuration dictionary for plugin-specific settings.
        """
        self._config = config or {}
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin and load any required resources.

        This method should be called before using the plugin.
        It handles lazy loading of models, connections, etc.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the plugin and release resources.

        Should be called when the plugin is no longer needed.
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if the plugin has been initialized."""
        return self._initialized

    @property
    def config(self) -> Dict[str, Any]:
        """Get plugin configuration."""
        return self._config

    def _ensure_initialized(self) -> None:
        """Ensure the plugin is initialized before use.

        Raises:
            RuntimeError: If the plugin has not been initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                f"{self.__class__.__name__} is not initialized. "
                "Call initialize() first."
            )
