# Memory

### [2026-02-23] Custom Plugin Refactor Architecture Lessons
- **问题描述**: When refactoring to custom Livekit plugins based on RealTalk models, several missing internal dependent modules (`logging_config`, `exceptions`, `cognition`) prevented the CLI commands from spinning up successfully in Python.
- **根本原因**: The wrapper implementations contained pseudo-code or partially copied system paths that referred to internal project modules that were not provided. Additionally, OpenRouter does not support TTS out of the box.
- **解决方案**: To get the pipeline running, the TTS plugin was rewritten to use an `openai.TTS` instance pointed to any compatible TTS provider endpoint via `.env`, and the LLM plugin was routed to use `openai.LLM` pointed at OpenRouter. Other pseudo-code modules were stubbed.
- **预防措施**: In future module refactors, decouple deeply-nested dependencies by instantiating API clients directly in the plugin layer when replacing a large monolithic provider like an overarching OpenAI Realtime agent model. Ensure environment variables are thoroughly documented.

### [2026-02-23] sherpa-onnx dynamic library missing on macOS
- **问题描述**: When running `realtime_agent.py console`, `sherpa_onnx` failed to import with `Library not loaded: @rpath/libonnxruntime.1.23.2.dylib`.
- **根本原因**: The `sherpa-onnx` wheel on macOS expects `libonnxruntime.1.23.2.dylib` to exist in `@rpath` (relative to the `.so` file), but it was only installed in `onnxruntime`'s `capi` folder within the `.venv`.
- **解决方案**: Created a symbolic link from `.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.23.2.dylib` to `.venv/lib/python3.12/site-packages/sherpa_onnx/lib/libonnxruntime.1.23.2.dylib`.
- **预防措施**: When installing pip packages with native bindings like `sherpa-onnx` that depend on `onnxruntime`, be prepared to manually link dynamic libraries if the packaging is incomplete for the OS.
