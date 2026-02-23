"""LiveKit TTS Plugin for RealTalk.

Wraps RealTalk's MinimaxTTS as a LiveKit-compatible TTS plugin.
"""

import asyncio
import io
import uuid

import numpy as np
import soundfile as sf
from livekit import rtc
from livekit.agents import tts
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

from ..cognition.tts import BaseTTS, MinimaxTTS, create_tts
from ..logging_config import setup_logger

logger = setup_logger("realtalk.livekit_plugins.tts")


class RealTalkTTSPlugin(tts.TTS):
    """LiveKit TTS plugin wrapping RealTalk's MinimaxTTS.

    Provides text-to-speech synthesis with streaming support.
    """

    def __init__(
        self,
        tts_engine: BaseTTS | None = None,
        *,
        sample_rate: int = 32000,
    ):
        """Initialize the TTS plugin.

        Args:
            tts_engine: Optional pre-configured TTS instance. If None, creates default MinimaxTTS.
            sample_rate: Audio sample rate in Hz. Default is 32000 (Minimax default).
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,  # MinimaxTTS uses HTTP chunked, not true streaming
                aligned_transcript=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._tts = tts_engine
        self._sample_rate = sample_rate
        self._initialized = False

    @property
    def model(self) -> str:
        """Return the model name."""
        if self._tts is None:
            return "unknown"
        # Duck typing for voice_id which is specific to MinimaxTTS
        if hasattr(self._tts, 'voice_id'):
            return "minimax-speech-2.8-hd"
        return "unknown"

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return "realtalk"

    async def initialize(self) -> None:
        """Initialize the TTS if not already done."""
        if self._initialized:
            return

        if self._tts is None:
            logger.info("Creating default MinimaxTTS instance...")
            self._tts = await create_tts()

        self._initialized = True
        logger.info("RealTalkTTSPlugin initialized")

    async def aclose(self) -> None:
        """Close the TTS and release resources."""
        if self._tts:
            await self._tts.close()
            self._initialized = False
            logger.info("RealTalkTTSPlugin closed")

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "RealTalkChunkedStream":
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            conn_options: API connection options

        Returns:
            RealTalkChunkedStream for streaming audio output
        """
        return RealTalkChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def _mp3_to_audio_frame(
        self,
        mp3_data: bytes,
        target_sample_rate: int = 16000,
    ) -> rtc.AudioFrame:
        """Convert MP3 data to LiveKit AudioFrame.

        Args:
            mp3_data: MP3 encoded audio data
            target_sample_rate: Target sample rate for output

        Returns:
            LiveKit AudioFrame with PCM data
        """
        try:
            # Decode MP3 to numpy array
            audio_array, source_sr = sf.read(
                io.BytesIO(mp3_data),
                dtype=np.int16,
            )

            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1).astype(np.int16)

            # Resample if needed (simple decimation/interpolation)
            if source_sr != target_sample_rate:
                # Calculate new length
                new_length = int(len(audio_array) * target_sample_rate / source_sr)
                # Simple resampling using numpy interp
                indices = np.linspace(0, len(audio_array) - 1, new_length)
                audio_array = np.interp(
                    indices,
                    np.arange(len(audio_array)),
                    audio_array,
                ).astype(np.int16)

            # Create AudioFrame
            frame = rtc.AudioFrame(
                data=audio_array.tobytes(),
                sample_rate=target_sample_rate,
                num_channels=1,
                samples_per_channel=len(audio_array),
            )

            return frame

        except Exception as e:
            logger.error(f"MP3 decoding error: {e}")
            # Return empty frame on error
            return rtc.AudioFrame(
                data=b"",
                sample_rate=target_sample_rate,
                num_channels=1,
                samples_per_channel=0,
            )


class RealTalkChunkedStream(tts.ChunkedStream):
    """Chunked synthesis stream for RealTalk TTS.

    Handles non-streaming TTS API and yields audio chunks.
    """

    def __init__(
        self,
        *,
        tts: RealTalkTTSPlugin,
        input_text: str,
        conn_options: APIConnectOptions,
    ):
        """Initialize the chunked stream.

        Args:
            tts: Parent TTS plugin
            input_text: Text to synthesize
            conn_options: API connection options
        """
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )
        self._realtalk_tts = tts
        self._request_id = str(uuid.uuid4())

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the synthesis and emit audio chunks.

        Args:
            output_emitter: Audio emitter for outputting synthesized audio
        """
        await self._realtalk_tts.initialize()

        try:
            # Initialize the emitter
            output_emitter.initialize(
                request_id=self._request_id,
                sample_rate=16000,  # Output at 16kHz for LiveKit compatibility
                num_channels=1,
                mime_type="audio/pcm",
                stream=False,
            )

            # Get the text to synthesize
            text = self._input_text
            if not text or not text.strip():
                logger.warning("Empty text provided for TTS synthesis")
                return

            logger.info(f"Synthesizing text ({len(text)} chars): {text[:50]}...")

            # Use stream_synthesize for better compatibility
            chunk_count = 0
            async for result in self._realtalk_tts._tts.stream_synthesize(text):
                if result.audio:
                    # Convert MP3 to AudioFrame
                    frame = self._realtalk_tts._mp3_to_audio_frame(
                        result.audio,
                        target_sample_rate=16000,
                    )

                    if frame.samples_per_channel > 0:
                        # Emit the audio
                        output_emitter.push(frame.data)
                        chunk_count += 1
                        logger.debug(f"Emitted audio chunk {chunk_count}: "
                                   f"{frame.samples_per_channel} samples")

                # Check for stop signal
                if result.is_final:
                    break

            logger.info(f"TTS synthesis complete: {chunk_count} chunks emitted")

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise


async def create_tts_plugin(
    tts_engine: BaseTTS | None = None,
    sample_rate: int = 32000,
) -> RealTalkTTSPlugin:
    """Factory function to create TTS plugin.

    Args:
        tts_engine: Optional pre-configured TTS instance
        sample_rate: Audio sample rate

    Returns:
        Initialized RealTalkTTSPlugin
    """
    plugin = RealTalkTTSPlugin(tts_engine=tts_engine, sample_rate=sample_rate)
    await plugin.initialize()
    return plugin
