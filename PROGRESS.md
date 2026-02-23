# Progress

## Business Context & Architecture
Currently modifying `realtime_agent.py` to replace LiveKit official plugins with custom plugins from `src/talkie/livekit_plugins`.
Using openrouter for LLM, minimax/openrouter for TTS, and local ASR.

## Current Iteration Goal
- Modify `src/talkie/livekit_plugins` to replace the official LiveKit plugin.
- Get a complete CLI voice agent running.

## TODOs
- [ ] Connect agent to Frontend (next phase)

## DONE
- [x] Analyze `docs/LIVEKIT_ARCHITECTURE.md`.
- [x] Review existing `src/talkie/livekit_plugins` implementations.
- [x] Refactor `llm.py`, `tts.py` to use OpenRouter API and OpenAI Compatible TTS Endpoint. 
- [x] Stub `logging_config`, `config`, `exceptions` and `vad` modules for `asr.py`.
- [x] Modify `realtime_agent.py` to leverage the compiled pipeline.
- [x] Test the pipeline by starting the LiveKit Server.
- [x] Fix `sherpa_onnx` dynamic library loading issue on macOS.
