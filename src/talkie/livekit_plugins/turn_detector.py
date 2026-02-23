import asyncio
import time
import numpy as np
from typing import Optional, List
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from livekit import rtc
from livekit.agents.vad import VAD, VADCapabilities, VADStream, VADEvent, VADEventType
from livekit.plugins import silero
from talkie.logging_config import setup_logger

logger = setup_logger("talkie.livekit_plugins.turn_detector")

class SmartTurnVADStream(VADStream):
    def __init__(self, vad: "SmartTurnVAD", silero_stream: VADStream) -> None:
        super().__init__(vad)
        self._smart_turn_vad = vad
        self._silero_stream = silero_stream
        
        # Audio accumulation buffer (8 seconds at 16kHz 16-bit mono = 256000 bytes)
        self._audio_buffer = bytearray()
        self._max_buffer_bytes = 8 * 16000 * 2  # 8 secs
        
        # Feature extractor for Whisper ONNX model
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        
    def push_frame(self, frame: rtc.AudioFrame) -> None:
        super().push_frame(frame)
        self._silero_stream.push_frame(frame)
        
        # Accumulate audio for the ONNX model (assuming 16kHz 16-bit PCM)
        # We only keep the last 8 seconds
        audio_array = np.frombuffer(frame.data, dtype=np.int16)
        self._audio_buffer.extend(audio_array.tobytes())
        if len(self._audio_buffer) > self._max_buffer_bytes:
            self._audio_buffer = self._audio_buffer[-self._max_buffer_bytes:]

    def flush(self) -> None:
        super().flush()
        self._silero_stream.flush()

    def end_input(self) -> None:
        super().end_input()
        self._silero_stream.end_input()
        
    async def aclose(self) -> None:
        await self._silero_stream.aclose()
        await super().aclose()

    async def _main_task(self) -> None:
        in_speech = False
        try:
            async for ev in self._silero_stream:
                if ev.type == VADEventType.START_OF_SPEECH:
                    if not in_speech:
                        in_speech = True
                        self._event_ch.send_nowait(ev)
                    else:
                        # We suppressed a previous END, so to the downstream it's still just one continuous START
                        pass
                elif ev.type == VADEventType.END_OF_SPEECH:
                    # Silero thinks speech ended. Let's verify with Pipecat Smart Turn.
                    is_complete = await self._verify_turn_complete()
                    if is_complete:
                        logger.info("Smart Turn: TURN COMPLETE. Emitting END_OF_SPEECH.")
                        in_speech = False
                        self._event_ch.send_nowait(ev)
                    else:
                        logger.info("Smart Turn: TURN INCOMPLETE. Suppressing END_OF_SPEECH.")
                        # Retain in_speech = True
                else:
                    self._event_ch.send_nowait(ev)
        except Exception as e:
            logger.error(f"Error in SmartTurnVADStream: {e}")
        finally:
            self._event_ch.close()

    async def _verify_turn_complete(self) -> bool:
        """Run the ONNX inference to check if the turn is complete."""
        try:
            if not self._smart_turn_vad._session:
                return True # Fallback if model not loaded
                
            # Convert buffer to float32 numpy array
            if not self._audio_buffer:
                return True
                
            audio_array = np.frombuffer(bytes(self._audio_buffer), dtype=np.int16).astype(np.float32)
            # Normalize int16 PCM to roughly [-1.0, 1.0] if needed by Whisper? 
            # Transformers WhisperFeatureExtractor handles normalization but expects raw waveforms generally in roughly [-1.0, 1.0] range
            audio_array = audio_array / 32768.0
            
            # Pad with zeros at the beginning if less than 8 seconds
            max_samples = 8 * 16000
            if len(audio_array) < max_samples:
                padding = max_samples - len(audio_array)
                audio_array = np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
            else:
                audio_array = audio_array[-max_samples:]

            # This can be CPU intensive, ideally run in an executor
            loop = asyncio.get_running_loop()
            prediction, probability = await loop.run_in_executor(
                None, self._run_onnx_inference, audio_array
            )
            
            logger.info(f"Smart Turn probability: {probability:.4f}")
            return probability > 0.5
            
        except Exception as e:
            logger.error(f"Error executing ONNX inference: {e}")
            return True # If inference fails, fallback to Silero's decision
            
    def _run_onnx_inference(self, audio_array: np.ndarray):
        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)  
        
        outputs = self._smart_turn_vad._session.run(None, {"input_features": input_features})
        probability = outputs[0][0].item()
        prediction = 1 if probability > 0.5 else 0
        return prediction, probability


class SmartTurnVAD(VAD):
    def __init__(self, silero_vad: VAD, onnx_model_path: str) -> None:
        super().__init__(capabilities=silero_vad.capabilities)
        self._silero_vad = silero_vad
        
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self._session = ort.InferenceSession(onnx_model_path, sess_options=so)
            logger.info(f"Loaded Smart Turn v3 model from {onnx_model_path}")
        except Exception as e:
            logger.error(f"Failed to load Smart Turn v3 model: {e}")
            self._session = None

    @property
    def model(self) -> str:
        return "pipecat-smart-turn-v3"

    @property
    def provider(self) -> str:
        return "pipecat"

    def stream(self) -> VADStream:
        return SmartTurnVADStream(self, self._silero_vad.stream())
