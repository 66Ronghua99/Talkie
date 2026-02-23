"""LiveKit LLM Plugin for RealTalk.

Wraps RealTalk's OpenRouterLLM as a LiveKit-compatible LLM plugin.
"""

from typing import Any

from livekit.agents.llm import (
    LLM,
    LLMStream,
    ChatContext,
    ChatMessage,
    ChatChunk,
    ChoiceDelta,
    CompletionUsage,
)
from livekit.agents.types import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)

from ..cognition.llm import BaseLLM, OpenRouterLLM, Message, create_llm
from ..logging_config import setup_logger

logger = setup_logger("realtalk.livekit_plugins.llm")


class RealTalkLLMPlugin(LLM):
    """LiveKit LLM plugin wrapping RealTalk's OpenRouterLLM.

    Provides chat completion with streaming support.
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
    ):
        """Initialize the LLM plugin.

        Args:
            llm: Optional pre-configured LLM instance. If None, creates default OpenRouterLLM.
        """
        super().__init__()
        self._llm = llm
        self._initialized = False

    @property
    def model(self) -> str:
        """Return the model name."""
        if self._llm is None:
            return "unknown"
        # Duck typing for model name
        return getattr(self._llm, 'model_name', 'unknown')

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return "realtalk"

    async def initialize(self) -> None:
        """Initialize the LLM if not already done."""
        if self._initialized:
            return

        if self._llm is None:
            logger.info("Creating default OpenRouterLLM instance...")
            self._llm = await create_llm()

        self._initialized = True
        logger.info("RealTalkLLMPlugin initialized")

    async def aclose(self) -> None:
        """Close the LLM and release resources."""
        if self._llm:
            await self._llm.close()
            self._initialized = False
            logger.info("RealTalkLLMPlugin closed")

    def _convert_chat_context(self, chat_ctx: ChatContext) -> tuple[str | None, list[Message]]:
        """Convert LiveKit ChatContext to RealTalk format.

        Args:
            chat_ctx: LiveKit chat context

        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = None
        messages = []

        for item in chat_ctx.items:
            if item.type != "message":
                continue

            msg = item

            # Extract text content
            if isinstance(msg.content, str):
                text = msg.content
            elif isinstance(msg.content, list):
                # Handle mixed content - extract text parts
                text_parts = []
                for content in msg.content:
                    if isinstance(content, str):
                        text_parts.append(content)
                    # Skip image/audio content for now
                text = " ".join(text_parts) if text_parts else ""
            else:
                text = ""

            if not text:
                continue

            # Map role
            role = msg.role
            if role == "developer":
                role = "system"

            if role == "system":
                # Use first system message as system prompt
                if system_prompt is None:
                    system_prompt = text
                else:
                    messages.append(Message(role="system", content=text))
            else:
                messages.append(Message(role=role, content=text))

        return system_prompt, messages

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Any] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "RealTalkLLMStream":
        """Start a chat completion stream.

        Args:
            chat_ctx: Chat context with message history
            tools: Optional list of tools (not supported yet)
            conn_options: API connection options
            parallel_tool_calls: Whether to allow parallel tool calls
            tool_choice: Tool choice configuration
            extra_kwargs: Extra provider-specific parameters

        Returns:
            RealTalkLLMStream for streaming completion
        """
        return RealTalkLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class RealTalkLLMStream(LLMStream):
    """Streaming chat completion for RealTalk LLM.

    Streams responses from the underlying LLM implementation.
    """

    def __init__(
        self,
        *,
        llm: RealTalkLLMPlugin,
        chat_ctx: ChatContext,
        tools: list[Any],
        conn_options: APIConnectOptions,
    ):
        """Initialize the LLM stream.

        Args:
            llm: Parent LLM plugin
            chat_ctx: Chat context
            tools: List of available tools
            conn_options: API connection options
        """
        super().__init__(
            llm=llm,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._realtalk_llm = llm

    async def _run(self) -> None:
        """Main processing loop for streaming completion.

        Connects to the LLM and streams responses as ChatChunks.
        """
        await self._realtalk_llm.initialize()

        try:
            # Convert chat context
            system_prompt, messages = self._realtalk_llm._convert_chat_context(
                self.chat_ctx
            )

            if not messages:
                logger.warning("No messages to send to LLM")
                return

            # Track usage
            prompt_tokens = 0
            completion_tokens = 0

            # Stream responses
            async for response in self._realtalk_llm._llm.stream_chat(
                messages=messages,
                system_prompt=system_prompt,
            ):
                # Send content chunk
                if response.content:
                    completion_tokens += len(response.content) // 4  # Rough estimate

                    chunk = ChatChunk(
                        id=response.model or "chunk",
                        delta=ChoiceDelta(
                            role="assistant",
                            content=response.content,
                        ),
                    )
                    await self._event_ch.send(chunk)

                # Check for finish
                if response.finish_reason:
                    # Send final usage chunk
                    usage_chunk = ChatChunk(
                        id=response.model or "final",
                        delta=None,
                        usage=CompletionUsage(
                            completion_tokens=completion_tokens,
                            prompt_tokens=prompt_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        ),
                    )
                    await self._event_ch.send(usage_chunk)
                    break

        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise


async def create_llm_plugin(
    llm: BaseLLM | None = None,
) -> RealTalkLLMPlugin:
    """Factory function to create LLM plugin.

    Args:
        llm: Optional pre-configured LLM instance

    Returns:
        Initialized RealTalkLLMPlugin
    """
    plugin = RealTalkLLMPlugin(llm=llm)
    await plugin.initialize()
    return plugin
