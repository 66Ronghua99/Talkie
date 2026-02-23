"""LiveKit STT Plugin for RealTalk.

Wraps RealTalk's SherpaOnnxASR as a LiveKit-compatible STT plugin.
"""

import asyncio
import uuid
from typing import List

import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

from ..perception.asr import BaseASR, SherpaOnnxASR, create_asr
from ..perception.vad import BaseVAD, create_vad
from ..logging_config import setup_logger

logger = setup_logger("realtalk.livekit_plugins.stt")


class RealTalkSTTPlugin(stt.STT):
    """LiveKit STT plugin wrapping RealTalk's SherpaOnnxASR.

    Provides both batch recognition and streaming recognition capabilities.
    """

    def __init__(
        self,
        asr: BaseASR | None = None,
        vad: BaseVAD | None = None,
        *,
        sample_rate: int = 16000,
    ):
        """Initialize the STT plugin.

        Args:
            asr: Optional pre-configured ASR instance. If None, creates default SherpaOnnxASR.
            vad: Optional pre-configured VAD instance.
            sample_rate: Audio sample rate in Hz. Default is 16000.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=False,  # SherpaOnnxASR doesn't provide interim results
                diarization=False,
                aligned_transcript=False,
                offline_recognize=True,
            )
        )
        self._asr = asr
        self._vad = vad
        self._sample_rate = sample_rate
        self._initialized = False

    @property
    def model(self) -> str:
        """Return the model name."""
        if self._asr is None:
            return "unknown"
        # Check for SherpaOnnxASR by duck typing
        if hasattr(self._asr, '_recognizer'):
            return "sherpa-onnx-sense-voice"
        return "unknown"

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return "realtalk"

    async def initialize(self) -> None:
        """Initialize the ASR if not already done."""
        if self._initialized:
            return

        if self._asr is None:
            logger.info("Creating default SherpaOnnxASR instance...")
            self._asr = await create_asr()
        elif isinstance(self._asr, SherpaOnnxASR) and self._asr._recognizer is None:
            logger.info("Loading SherpaOnnxASR model...")
            await self._asr.load()

        if self._vad is None:
            logger.info("Creating default VAD instance...")
            self._vad = await create_vad()
        elif hasattr(self._vad, 'load') and getattr(self._vad, '_model', None) is None and getattr(self._vad, '_vad', None) is None:
            logger.info("Loading VAD model...")
            await self._vad.load()

        self._initialized = True
        logger.info("RealTalkSTTPlugin initialized")

    async def close(self) -> None:
        """Close the ASR and release resources."""
        if self._asr:
            await self._asr.close()
        if self._vad:
            await self._vad.close()
        if self._asr or self._vad:
            self._initialized = False
            logger.info("RealTalkSTTPlugin closed")

    def _audio_buffer_to_bytes(self, buffer: utils.AudioBuffer) -> bytes:
        """Convert AudioBuffer to 16-bit PCM bytes.

        Args:
            buffer: AudioBuffer (single frame or list of frames)

        Returns:
            16-bit PCM audio bytes
        """
        frames: List[rtc.AudioFrame]
        if isinstance(buffer, rtc.AudioFrame):
            frames = [buffer]
        else:
            frames = buffer

        # Convert each frame to numpy array and concatenate
        audio_arrays = []
        for frame in frames:
            # Convert frame data to numpy array (int16)
            audio_array = np.frombuffer(frame.data, dtype=np.int16)
            audio_arrays.append(audio_array)

        if not audio_arrays:
            return b""

        # Concatenate all frames
        combined = np.concatenate(audio_arrays)

        # Return as bytes
        return combined.tobytes()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Implement batch speech recognition.

        Args:
            buffer: Audio data to recognize
            language: Optional language hint
            conn_options: API connection options

        Returns:
            SpeechEvent with recognition results
        """
        await self.initialize()

        request_id = str(uuid.uuid4())

        try:
            # Convert audio buffer to bytes
            audio_bytes = self._audio_buffer_to_bytes(buffer)

            if not audio_bytes:
                logger.warning("Empty audio buffer provided for recognition")
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[],
                )

            # Perform recognition
            result = await self._asr.recognize(audio_bytes)

            # Create SpeechData from result
            if result.text:
                speech_data = stt.SpeechData(
                    language=result.language or "zh",
                    text=result.text,
                    confidence=result.confidence,
                )
                alternatives = [speech_data]
            else:
                alternatives = []

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=alternatives,
            )

        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[],
            )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "RealTalkRecognizeStream":
        """Create a streaming recognition session.

        Args:
            language: Optional language hint
            conn_options: API connection options

        Returns:
            RealTalkRecognizeStream for streaming recognition
        """
        return RealTalkRecognizeStream(
            stt=self,
            language=language,
            conn_options=conn_options,
        )


class RealTalkRecognizeStream(stt.RecognizeStream):
    """Streaming recognition session for RealTalk STT.

    Accumulates audio frames and performs recognition when enough
    audio data is collected.
    """

    def __init__(
        self,
        *,
        stt: RealTalkSTTPlugin,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ):
        """Initialize the recognize stream.

        Args:
            stt: Parent STT plugin
            language: Language hint
            conn_options: API connection options
        """
        super().__init__(stt=stt, conn_options=conn_options)
        self._realtalk_stt = stt
        self._language = language
        self._request_id = str(uuid.uuid4())

        # Audio accumulation buffer (1 second at 16kHz 16-bit mono = 32000 bytes)
        self._audio_buffer = bytearray()
        self._buffer_size_threshold = 16000 * 2  # 1 second of 16kHz 16-bit audio

        # Track if we've sent START_OF_SPEECH
        self._speech_started = False
        self._vad_silence_frames = 0
        self._max_silence_frames = 15  # ~1.5 seconds of silence before forcibly emitting END_OF_SPEECH (assuming ~100ms buffers)

    async def _run(self) -> None:
        """Main processing loop for streaming recognition.

        Reads audio frames from input channel, accumulates them,
        and performs recognition when enough data is collected.
        """
        await self._realtalk_stt.initialize()

        try:
            # _input_ch yields rtc.AudioFrame directly, not RecognitionEvent
            async for frame in self._input_ch:
                if isinstance(frame, rtc.AudioFrame):
                    await self._process_frame(frame)
                elif frame is None:
                    # End of stream signal
                    break

        except Exception as e:
            logger.error(f"Stream processing error: {e}")
        finally:
            # Finalize any remaining audio
            await self._finalize()
            # Ensure we send end of speech event
            if self._speech_started:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH,
                        request_id=self._request_id,
                        alternatives=[],
                    )
                )

    async def _process_frame(self, frame: rtc.AudioFrame) -> None:
        """Process a single audio frame.

        Args:
            frame: Audio frame from input channel
        """
        # Convert frame to bytes and add to buffer
        audio_array = np.frombuffer(frame.data, dtype=np.int16)
        self._audio_buffer.extend(audio_array.tobytes())

        # Check if we have enough audio to process
        if len(self._audio_buffer) >= self._buffer_size_threshold:
            await self._process_buffer()

    async def _process_buffer(self) -> None:
        """Process the current audio buffer."""
        if not self._audio_buffer:
            return

        # Keep overlap for next buffer
        overlap_bytes = 1600 * 2  # 100ms overlap
        audio_to_process = bytes(self._audio_buffer)
        self._audio_buffer = bytearray(self._audio_buffer[-overlap_bytes:])
        
        # Audio check: Convert buffer to numpy array for VAD
        audio_array = np.frombuffer(audio_to_process, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            # 1. Run local VAD first
            is_speech_frame = False
            if self._realtalk_stt._vad:
                vad_result = await self._realtalk_stt._vad.detect(audio_array)
                is_speech_frame = vad_result.is_speech
            else:
                is_speech_frame = True # Pass everything if no VAD is present

            if not is_speech_frame:
                if self._speech_started:
                    self._vad_silence_frames += 1
                    # If silence exceeds threshold, manually trigger END_OF_SPEECH
                    if self._vad_silence_frames >= self._max_silence_frames:
                        self._speech_started = False
                        self._vad_silence_frames = 0
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.END_OF_SPEECH,
                                request_id=self._request_id,
                                alternatives=[],
                            )
                        )
                # Skip ASR to prevent hallucination on silence/noise
                return

            # Speech detected: reset silence counter
            self._vad_silence_frames = 0

            # 2. Run ASR on verified speech
            result = await self._realtalk_stt._asr.recognize(audio_to_process)

            if result.text:
                # Send start of speech event if first result
                if not self._speech_started:
                    self._speech_started = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.START_OF_SPEECH,
                            request_id=self._request_id,
                            alternatives=[],
                        )
                    )

                # Send final transcript (SherpaOnnx doesn't do interim)
                speech_data = stt.SpeechData(
                    language=result.language or "zh",
                    text=result.text,
                    confidence=result.confidence,
                )

                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[speech_data],
                    )
                )

        except Exception as e:
            logger.error(f"Buffer processing error: {e}")

    async def _flush_buffer(self) -> None:
        """Flush the current buffer and process remaining audio."""
        if self._audio_buffer:
            await self._process_buffer()

    async def _finalize(self) -> None:
        """Finalize streaming recognition."""
        # Process any remaining audio
        if self._audio_buffer:
            audio_to_process = bytes(self._audio_buffer)
            self._audio_buffer = bytearray()

            try:
                result = await self._realtalk_stt._asr.recognize(audio_to_process)

                if result.text and not self._speech_started:
                    self._speech_started = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.START_OF_SPEECH,
                            request_id=self._request_id,
                            alternatives=[],
                        )
                    )

                if result.text:
                    speech_data = stt.SpeechData(
                        language=result.language or "zh",
                        text=result.text,
                        confidence=result.confidence,
                    )
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=self._request_id,
                            alternatives=[speech_data],
                        )
                    )

            except Exception as e:
                logger.error(f"Finalization error: {e}")

        # Send end of speech event
        if self._speech_started:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH,
                    request_id=self._request_id,
                    alternatives=[],
                )
            )


async def create_stt_plugin(
    asr: BaseASR | None = None,
    vad: BaseVAD | None = None,
    sample_rate: int = 16000,
) -> RealTalkSTTPlugin:
    """Factory function to create STT plugin.

    Args:
        asr: Optional pre-configured ASR instance
        vad: Optional pre-configured VAD instance
        sample_rate: Audio sample rate

    Returns:
        Initialized RealTalkSTTPlugin
    """
    plugin = RealTalkSTTPlugin(asr=asr, vad=vad, sample_rate=sample_rate)
    await plugin.initialize()
    return plugin
