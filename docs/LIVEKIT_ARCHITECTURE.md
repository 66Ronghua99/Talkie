# LiveKit Agents 架构深度解析

> 文档版本: 1.0.0
> 基于: LiveKit Agents 1.4.x
> 目标: 深度自定义开发与插件替换指南

---

## 目录

1. [架构概览](#1-架构概览)
2. [核心类详解](#2-核心类详解)
3. [数据流完整链路](#3-数据流完整链路)
4. [插件系统](#4-插件系统)
5. [可替换组件指南](#5-可替换组件指南)
6. [可复用组件指南](#6-可复用组件指南)
7. [自定义开发模式](#7-自定义开发模式)
8. [调试与监控](#8-调试与监控)

---

## 1. 架构概览

### 1.1 三层核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent (配置层)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Instructions│  │   Tools     │  │  ChatContext│          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │    STT      │  │    LLM      │  │    TTS      │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│  ┌─────────────┐  ┌─────────────┐                            │
│  │    VAD      │  │allow_interruptions                      │
│  └─────────────┘  └─────────────┘                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                 AgentSession (协调层)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Input / Output 管理                       │  │
│  │   ┌──────────────┐          ┌──────────────┐         │  │
│  │   │  Audio Input │          │ Audio Output │         │  │
│  │   └──────────────┘          └──────────────┘         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Global ChatContext (历史)                │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                AgentActivity (执行层)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   VAD Task  │  │   STT Task  │  │ Turn Detect │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Scheduling │  │  Generation │  │Interruption │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 关键设计原则

| 原则 | 说明 | 体现 |
|------|------|------|
| **分层解耦** | 配置/协调/执行分离 | Agent 无状态，Session 协调，Activity 执行 |
| **流式优先** | 所有组件支持流式处理 | STT.stream(), LLM.chat(), TTS.synthesize() |
| **事件驱动** | 松散耦合的异步事件 | EventEmitter 模式 |
| **可插拔** | 标准接口，易于替换 | 基类抽象方法定义清晰 |

---

## 2. 核心类详解

### 2.1 Agent 类

**文件**: `livekit/agents/voice/agent.py`

**职责**: 定义 Agent 的行为配置，作为"蓝图"存在

```python
class Agent:
    def __init__(
        self,
        *,
        instructions: str,                    # 系统提示词
        stt: STT,                             # STT 插件
        llm: LLM,                             # LLM 插件
        tts: TTS,                             # TTS 插件
        vad: VAD,                             # VAD 插件
        tools: list[Tool] = [],               # 可用工具
        allow_interruptions: bool = True,     # 允许打断
        # ... 更多配置
    ):
```

**关键回调方法**:

```python
async def on_user_turn_completed(
    self,
    chat_ctx: ChatContext,        # 当前对话上下文
    new_message: ChatMessage,     # 用户新消息
) -> ChatContext | None:
    """
    用户说完话后、LLM 响应前调用

    返回值:
    - ChatContext: 继续处理，可能修改了上下文
    - None: 跳过此次响应（用于 Gatekeeper 等场景）
    """
```

**可覆盖的 Pipeline 节点**:

```python
def stt_node(self, stt: STT, chat_ctx: ChatContext) -> STTNode:
    """自定义 STT 处理逻辑"""
    return default_stt_node(stt)

def llm_node(self, llm: LLM, chat_ctx: ChatContext) -> LLMNode:
    """自定义 LLM 处理逻辑"""
    return default_llm_node(llm)

def tts_node(self, tts: TTS, chat_ctx: ChatContext) -> TTSNode:
    """自定义 TTS 处理逻辑"""
    return default_tts_node(tts)
```

### 2.2 AgentSession 类

**文件**: `livekit/agents/voice/agent_session.py`

**职责**: 管理 Agent 生命周期，协调输入输出，维护全局状态

```python
class AgentSession(rtc.EventEmitter):
    def __init__(
        self,
        *,
        stt: STT | None = None,               # 可覆盖 Agent 的配置
        llm: LLM | None = None,
        tts: TTS | None = None,
        vad: VAD | None = None,
        allow_interruptions: bool = True,
        min_endpointing_delay: float = 0.5,   # 最小断句延迟
        max_endpointing_delay: float = 3.0,   # 最大断句延迟
        # ...
    ):
```

**核心属性**:

```python
self._chat_ctx: ChatContext           # 全局对话历史
self._input: AgentInput               # 输入管理（音频/视频）
self._output: AgentOutput             # 输出管理（音频/文本）
self._current_speech: SpeechHandle    # 当前正在播放的语音
self._user_state: UserState           # 用户状态（IDLE/SPEAKING）
self._agent_state: AgentState         # Agent 状态（IDLE/SPEAKING）
```

**关键方法**:

```python
async def start(
    self,
    agent: Agent,
    room: rtc.Room | None = None,        # LiveKit 房间
    room_options: RoomOptions | None = None,
    # ...
) -> SpeechHandle:
    """启动 Agent，开始监听和处理"""

async def interrupt(self, *, force: bool = False) -> None:
    """打断当前播放"""
```

### 2.3 AgentActivity 类

**文件**: `livekit/agents/voice/agent_activity.py` (内部类)

**职责**: 实时处理的核心调度器，用户通常不直接交互

**核心任务**:

```python
# 1. 语音识别任务 (VAD + STT)
async def _stt_task(self) -> None:
    """持续监听 STT 输出，更新 transcript"""

# 2. 语音活动检测任务
async def _vad_task(self) -> None:
    """监听 VAD，检测打断和语音开始/结束"""

# 3. 调度任务
async def _scheduling_task(self) -> None:
    """管理语音队列，决定何时播放哪段语音"""

# 4. 生成任务
async def _generation_task(self) -> None:
    """执行 LLM + TTS 生成"""
```

---

## 3. 数据流完整链路

### 3.1 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            用户说话流程                                  │
└─────────────────────────────────────────────────────────────────────────┘

  User Audio
      ↓
┌─────────────────┐
│  RoomIO Input   │  ← 从 WebRTC 轨道接收音频帧
│  (Participant)  │
└────────┬────────┘
         ↓
┌─────────────────┐
│ AudioRecognition│  ← 协调 VAD + STT
│                 │
│  ┌───────────┐  │
│  │  VAD      │  │  → START_OF_SPEECH (语音开始)
│  │  Stream   │  │  → END_OF_SPEECH (语音结束)
│  └───────────┘  │
│       ↓         │
│  ┌───────────┐  │
│  │  STT      │  │  → INTERIM_TRANSCRIPT (临时文本)
│  │  Stream   │  │  → FINAL_TRANSCRIPT (最终结果)
│  └───────────┘  │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Turn Detection │  ← 综合判断：是否该回应了？
│                 │
│  触发条件:       │
│  - VAD 检测到    │
│    足够长的静音  │
│  - STT 返回      │
│    FINAL         │
│  - Realtime LLM  │
│    server_vad    │
└────────┬────────┘
         ↓
┌─────────────────┐
│  AgentActivity  │
│  _on_speech_committed
│                 │
│  1. 调用         │
│  agent.on_user_ │
│  turn_completed │
│                 │
│  2. 如果返回     │
│  ChatContext,   │
│  启动生成        │
└────────┬────────┘
         ↓
┌─────────────────┐     ┌─────────────────┐
│  LLM Generation │  →  │  TTS Generation │
│                 │     │                 │
│  stream_chat()  │     │  synthesize()   │
│  流式输出文本    │     │  或 stream()    │
└────────┬────────┘     └────────┬────────┘
         ↓                        ↓
         └──────────┬─────────────┘
                    ↓
┌─────────────────────────────────┐
│      TranscriptSynchronizer      │
│  (同步音频播放和文本显示)          │
└─────────────────────────────────┘
                    ↓
┌─────────────────┐
│  RoomIO Output  │  → 发送到 WebRTC 轨道
│  (Audio Track)  │
└─────────────────┘
```

### 3.2 打断流程

```
用户说话（Agent 正在播放时）
         ↓
┌─────────────────┐
│  VAD 检测到      │
│  新语音          │
└────────┬────────┘
         ↓
┌─────────────────┐
│ 检查打断条件     │
│                 │
│ - 持续时间 >     │
│   min_interruption_duration
│ - 字数 >         │
│   min_interruption_words
└────────┬────────┘
         ↓
┌─────────────────┐
│ SpeechHandle.   │
│ interrupt()     │
└────────┬────────┘
         ↓
┌─────────────────┐
│ 1. 停止 TTS     │
│ 2. 取消生成任务  │
│ 3. 清空播放队列  │
│ 4. 触发打断事件  │
└────────┬────────┘
         ↓
┌─────────────────┐
│ 开始新的        │
│ 用户轮次        │
└─────────────────┘
```

---

## 4. 插件系统

### 4.1 基类架构

```
┌────────────────────────────────────────────────────────────┐
│                      基类层次结构                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  STT (stt/stt.py)                                          │
│  ├── capabilities: STTCapabilities                         │
│  ├── _recognize_impl() - 批量识别                           │
│  └── stream() → RecognizeStream - 流式识别                  │
│                                                            │
│  RecognizeStream (stt/stt.py)                              │
│  ├── _input_ch: AudioFrame 输入队列                        │
│  ├── _event_ch: SpeechEvent 输出队列                        │
│  └── _run() - 核心处理循环                                  │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  LLM (llm/llm.py)                                          │
│  ├── chat() → LLMStream - 开始对话                         │
│  └── 也可以是 RealtimeModel (服务端 VAD)                    │
│                                                            │
│  LLMStream (llm/llm.py)                                    │
│  ├── _run() - 核心处理循环                                  │
│  └── 产出 ChatChunk                                        │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  TTS (tts/tts.py)                                          │
│  ├── capabilities: TTSCapabilities                         │
│  ├── synthesize() → ChunkedStream - 非流式合成              │
│  └── stream() → SynthesizeStream - 流式合成                 │
│                                                            │
│  ChunkedStream (tts/tts.py)                                │
│  ├── _run() - 核心处理循环                                  │
│  └── 产出 SynthesizedAudio                                 │
│                                                            │
│  SynthesizeStream (tts/tts.py)                             │
│  ├── push_text() - 输入文本                                │
│  ├── end_input() - 结束输入                                │
│  └── 产出 SynthesizedAudio                                 │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  VAD (vad.py)                                              │
│  ├── stream() → VADStream                                  │
│  └── 产出 VADEvent (START/END/INFERENCE)                   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 4.2 实现自定义插件

#### 4.2.1 自定义 STT 插件

```python
from livekit.agents import stt
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

class MyCustomSTT(stt.STT):
    def __init__(self):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,           # 支持流式
                interim_results=True,     # 支持中间结果
                diarization=False,
                aligned_transcript=False,
                offline_recognize=True,   # 支持批量识别
            )
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """批量识别 - 用于非实时场景"""
        # 1. 转换音频格式
        # 2. 调用你的 ASR
        # 3. 返回 SpeechEvent
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[
                stt.SpeechData(
                    language="zh",
                    text="识别结果",
                    confidence=0.95,
                )
            ],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "MyCustomStream":
        """创建流式识别会话"""
        return MyCustomStream(stt=self, language=language, conn_options=conn_options)


class MyCustomStream(stt.RecognizeStream):
    def __init__(self, *, stt: MyCustomSTT, language, conn_options):
        super().__init__(stt=stt, conn_options=conn_options)
        self._language = language

    async def _run(self) -> None:
        """核心处理循环 - 从 _input_ch 读取，向 _event_ch 发送"""
        # 注意: _input_ch 直接产生 AudioFrame，不是 RecognitionEvent
        async for frame in self._input_ch:
            if isinstance(frame, rtc.AudioFrame):
                # 1. 累积音频帧
                # 2. 检测语音开始/结束 (可选，如果信源已有 VAD)
                # 3. 调用 ASR
                # 4. 发送事件

                # 发送开始事件
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.START_OF_SPEECH,
                        request_id=self._request_id,
                        alternatives=[],
                    )
                )

                # 发送识别结果
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[
                            stt.SpeechData(
                                language="zh",
                                text="识别的文本",
                                confidence=0.9,
                            )
                        ],
                    )
                )

                # 发送结束事件
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH,
                        request_id=self._request_id,
                        alternatives=[],
                    )
                )

            elif frame is None:
                # 流结束信号
                break
```

#### 4.2.2 自定义 LLM 插件

```python
from livekit.agents.llm import LLM, LLMStream, ChatContext, ChatChunk, ChoiceDelta
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

class MyCustomLLM(LLM):
    def __init__(self):
        super().__init__()

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Any] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        # ... 其他参数
    ) -> "MyCustomStream":
        """开始对话流"""
        return MyCustomStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class MyCustomStream(LLMStream):
    def __init__(self, *, llm: MyCustomLLM, chat_ctx, tools, conn_options):
        super().__init__(
            llm=llm,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )

    async def _run(self) -> None:
        """核心处理循环 - 向 _event_ch 发送 ChatChunk"""
        try:
            # 转换 ChatContext 为你的格式
            messages = self._convert_chat_context(self.chat_ctx)

            # 调用你的 LLM 流式接口
            async for chunk in your_llm_client.stream_chat(messages):
                # 构造 ChatChunk
                chat_chunk = ChatChunk(
                    id=chunk.id,
                    delta=ChoiceDelta(
                        role="assistant",
                        content=chunk.text,
                    ),
                )
                await self._event_ch.send(chat_chunk)

            # 发送结束标记（可选，包含 usage）
            final_chunk = ChatChunk(
                id="final",
                delta=None,
                usage=CompletionUsage(
                    completion_tokens=tokens,
                    prompt_tokens=prompt_tokens,
                    total_tokens=total,
                ),
            )
            await self._event_ch.send(final_chunk)

        except asyncio.CancelledError:
            # 被打断时清理
            raise
```

#### 4.2.3 自定义 TTS 插件

```python
from livekit.agents import tts
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

class MyCustomTTS(tts.TTS):
    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,      # 不支持流式输入文本
                aligned_transcript=False,
            ),
            sample_rate=24000,        # 输出采样率
            num_channels=1,
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "MyCustomChunkedStream":
        """非流式合成 - 返回音频块"""
        return MyCustomChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "MyCustomSynthesizeStream":
        """流式合成 - 支持 push_text"""
        return MyCustomSynthesizeStream(
            tts=self,
            conn_options=conn_options,
        )


class MyCustomChunkedStream(tts.ChunkedStream):
    """非流式 TTS 实现"""

    def __init__(self, *, tts: MyCustomTTS, input_text: str, conn_options):
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """核心处理循环"""
        # 初始化 emitter
        output_emitter.initialize(
            request_id=self._request_id,
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            stream=False,
        )

        # 调用 TTS API
        audio_data = await your_tts_client.synthesize(self._input_text)

        # 转换为 AudioFrame
        frame = rtc.AudioFrame(
            data=audio_data,
            sample_rate=24000,
            num_channels=1,
            samples_per_channel=len(audio_data) // 2,  # int16 = 2 bytes
        )

        # 发送音频
        output_emitter.push(frame.data)


class MyCustomSynthesizeStream(tts.SynthesizeStream):
    """流式 TTS 实现 - 支持实时输入文本"""

    def __init__(self, *, tts: MyCustomTTS, conn_options):
        super().__init__(tts=tts, conn_options=conn_options)
        self._text_buffer = ""

    def push_text(self, text: str) -> None:
        """接收输入文本"""
        self._text_buffer += text

    def end_input(self) -> None:
        """标记输入结束"""
        self._input_done = True

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """核心处理循环"""
        output_emitter.initialize(
            request_id=self._request_id,
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        # 等待输入或处理缓冲的文本
        while not self._input_done or self._text_buffer:
            if self._text_buffer:
                text = self._text_buffer
                self._text_buffer = ""

                # 流式合成
                async for audio_chunk in your_tts_client.stream_synthesize(text):
                    output_emitter.push(audio_chunk)

            await asyncio.sleep(0.01)
```

---

## 5. 可替换组件指南

### 5.1 可替换点清单

| 组件 | 替换难度 | 关键接口 | 注意事项 |
|------|---------|---------|---------|
| **STT** | 低 | `stt.STT`, `RecognizeStream` | 处理好 START/END_OF_SPEECH 事件 |
| **LLM** | 低 | `llm.LLM`, `LLMStream` | 正确转换 ChatContext 格式 |
| **TTS** | 中 | `tts.TTS`, `ChunkedStream` | 注意采样率和音频格式转换 |
| **VAD** | 低 | `vad.VAD`, `VADStream` | 保持事件语义一致 |
| **Turn Detection** | 中 | 继承 Agent 覆盖方法 | 需要理解 EOU 检测流程 |
| **打断逻辑** | 高 | 修改 AgentSession 参数 | 影响整体交互体验 |

### 5.2 RealTalk 中的替换实践

**STT 替换示例** (`src/realtalk/livekit_plugins/stt.py`):

```python
# 将 RealTalk 的 SherpaOnnxASR 包装为 LiveKit STT 插件
class RealTalkSTTPlugin(stt.STT):
    def __init__(self, asr: BaseASR | None = None):
        super().__init__(capabilities=...)
        self._asr = asr or SherpaOnnxASR()

    async def _recognize_impl(self, buffer, ...):
        # 转换音频格式: AudioBuffer → PCM bytes
        audio_bytes = self._audio_buffer_to_bytes(buffer)
        # 调用 RealTalk ASR
        result = await self._asr.recognize(audio_bytes)
        # 包装为 SpeechEvent
        return stt.SpeechEvent(...)

    def stream(self, ...) -> RealTalkRecognizeStream:
        return RealTalkRecognizeStream(...)


class RealTalkRecognizeStream(stt.RecognizeStream):
    async def _run(self) -> None:
        async for frame in self._input_ch:  # 直接是 AudioFrame
            # 能量检测 + ASR
            rms = self._calculate_rms(frame)
            if rms > threshold:
                # 累积并识别
                result = await self._asr.recognize(audio_bytes)
                self._event_ch.send_nowait(SpeechEvent(...))
```

**TTS 替换示例** (`src/realtalk/livekit_plugins/tts.py`):

```python
# 关键: MP3 → AudioFrame 转换
class RealTalkTTSPlugin(tts.TTS):
    def _mp3_to_audio_frame(
        self,
        mp3_data: bytes,
        target_sample_rate: int = 16000,
    ) -> rtc.AudioFrame:
        # 使用 soundfile 解码 MP3
        audio_array, source_sr = sf.read(io.BytesIO(mp3_data), dtype=np.int16)

        # 重采样（如果需要）
        if source_sr != target_sample_rate:
            audio_array = resample(audio_array, source_sr, target_sample_rate)

        # 创建 AudioFrame
        return rtc.AudioFrame(
            data=audio_array.tobytes(),
            sample_rate=target_sample_rate,
            num_channels=1,
            samples_per_channel=len(audio_array),
        )
```

---

## 6. 可复用组件指南

### 6.1 推荐复用的 LiveKit 组件

| 组件 | 复用价值 | 说明 |
|------|---------|------|
| `AudioRecognition` | 高 | 复杂的 VAD+STT 协调逻辑 |
| `TurnDetector` (如果可用) | 高 | ML -based 断句检测 |
| `TranscriptSynchronizer` | 高 | 音频和字幕同步 |
| `RoomIO` | 高 | WebRTC 房间管理 |
| `SpeechHandle` | 中 | 语音播放队列管理 |
| `AgentSession` | 高 | 整体协调框架 |

### 6.2 打断处理逻辑（推荐复用）

LiveKit 的打断处理非常精细，建议复用:

```python
# 来自 agent_activity.py 的打断处理

class AgentActivity:
    def _interrupt_by_audio_activity(self) -> None:
        """基于音频活动的打断"""
        # 1. 检查最小打断持续时间
        if speaking_time < self._min_interruption_duration:
            return  # 忽略过短的输入

        # 2. 检查最小打断字数
        if word_count < self._min_interruption_words:
            return

        # 3. 假打断保护
        # 如果用户说了话但没有产生有效 transcript，
        # 在一段时间后恢复之前被打断的播放

    def _on_false_interruption_timeout(self) -> None:
        """假打断超时 - 恢复播放"""
        if not user_produced_transcript:
            self._playout_resume_queue.append(speech)
```

### 6.3 端到端延迟优化（复用策略）

```python
# LiveKit 内置的延迟优化策略

# 1. 预emptive generation
AgentSession(
    preemptive_generation=True,  # 在 STT 中间结果上启动 LLM
)

# 2. 首句快速响应
# 在 TTS 中实现: 第一句接收后立即开始播放，不等待完整文本

# 3. 智能断句
# 基于 VAD + STT + 可选的 TurnDetector 综合判断
```

---

## 7. 自定义开发模式

### 7.1 模式 1: 继承 Agent 类（推荐）

适用于: 修改对话逻辑、添加 Gatekeeper、自定义上下文管理

```python
class RealTalkAgent(Agent):
    def __init__(self, *, gatekeeper, accumulator, ...):
        super().__init__(instructions=...)
        self._gatekeeper = gatekeeper
        self._accumulator = accumulator

    async def on_user_turn_completed(
        self, chat_ctx: ChatContext, new_message: ChatMessage
    ) -> ChatContext | None:
        """RealTalk 的 Gatekeeper 集成点"""

        # 1. 提取用户文本
        user_text = self._extract_message_text(new_message)

        # 2. Gatekeeper 决策
        decision = self._gatekeeper.decide_sync(
            GatekeeperInput(text=user_text, ...)
        )

        # 3. 根据决策处理
        if decision.action == Action.ACCUMULATE:
            self._accumulator.add(user_text)
            return None  # 不触发 LLM

        elif decision.action == Action.REPLY:
            full_text = self._accumulator.flush() + user_text
            self._update_last_message(chat_ctx, full_text)
            return chat_ctx

        elif decision.action == Action.WAIT:
            return None
```

### 7.2 模式 2: 继承 AgentSession 类

适用于: 深度修改会话管理逻辑

```python
class CustomAgentSession(AgentSession):
    def __init__(self, *args, custom_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_config = custom_config

    async def start(self, agent: Agent, **kwargs):
        # 自定义启动逻辑
        await self._custom_setup()
        return await super().start(agent, **kwargs)
```

### 7.3 模式 3: 完全自定义 Pipeline

适用于: 不遵循标准 STT→LLM→TTS 流程的场景

```python
class CustomPipeline:
    def __init__(self):
        self._vad = silero.VAD.load()
        self._stt = MyCustomSTT()
        self._llm = MyCustomLLM()
        self._tts = MyCustomTTS()

    async def run(self, room: rtc.Room):
        # 完全自定义数据流
        audio_stream = await self._get_audio_stream(room)

        async for frame in audio_stream:
            # 自定义处理逻辑
            if await self._custom_vad(frame):
                text = await self._custom_stt(frame)
                response = await self._custom_llm(text)
                audio = await self._custom_tts(response)
                await self._play_audio(room, audio)
```

### 7.4 RealTalk 的混合模式

RealTalk 采用"插件包装 + Agent 继承"的混合模式:

```python
# 1. 插件层: 将 RealTalk 组件包装为 LiveKit 插件
RealTalkSTTPlugin(stt.STT)      # 包装 SherpaOnnxASR
RealTalkLLMPlugin(llm.LLM)      # 包装 OpenRouterLLM
RealTalkTTSPlugin(tts.TTS)      # 包装 MinimaxTTS

# 2. Agent 层: 继承并添加 RealTalk 逻辑
RealTalkAgent(Agent)            # 添加 Gatekeeper 集成

# 3. Session 层: 标准 AgentSession
AgentSession(...)               # 使用 LiveKit 原生协调
```

---

## 8. 调试与监控

### 8.1 关键日志点

```python
# 在自定义插件中添加结构化日志

class MyCustomSTT(stt.STT):
    async def _run(self) -> None:
        logger.info(f"[STT] Stream started")

        async for frame in self._input_ch:
            logger.debug(f"[STT] Frame: samples={len(frame.data)}, "
                        f"rms={rms:.2f}")

            if speech_detected:
                logger.info(f"[STT] Speech detected, rms={rms:.2f}")

            if transcript:
                logger.info(f"[STT] Transcript: '{transcript}'")
```

### 8.2 性能监控

```python
import time

class MonitoredLLM(LLM):
    async def _run(self) -> None:
        start_time = time.monotonic()
        first_chunk_time = None
        chunk_count = 0

        async for chunk in self._llm.stream_chat(...):
            if first_chunk_time is None:
                first_chunk_time = time.monotonic()
                ttft = first_chunk_time - start_time
                logger.info(f"[LLM] TTFT: {ttft*1000:.1f}ms")

            chunk_count += 1
            await self._event_ch.send(chunk)

        total_time = time.monotonic() - start_time
        logger.info(f"[LLM] Total: {total_time*1000:.1f}ms, "
                   f"Chunks: {chunk_count}")
```

### 8.3 常见问题排查

| 问题 | 检查点 | 解决方向 |
|------|--------|---------|
| STT 无输出 | `_input_ch` 是否收到数据 | 检查音频格式、采样率 |
| LLM 不触发 | `on_user_turn_completed` 是否返回 ChatContext | 检查返回值 |
| TTS 无声音 | `output_emitter.push()` 是否被调用 | 检查音频格式转换 |
| 打断不工作 | `allow_interruptions` 和 VAD 设置 | 检查 VAD 阈值 |
| 高延迟 | TTFT 和 TTS 首包时间 | 使用 preemptive generation |

---

## 附录 A: 类关系图

```
Agent
  ├── AgentSession (contains)
  │     ├── AgentActivity (creates internally)
  │     │     ├── AudioRecognition (coordinates)
  │     │     │     ├── VAD.stream()
  │     │     │     └── STT.stream()
  │     │     ├── GenerationBuilder (orchestrates)
  │     │     │     ├── LLM.chat()
  │     │     │     └── TTS.synthesize()
  │     │     └── SpeechQueue (manages)
  │     ├── RoomIO (manages)
  │     │     ├── ParticipantAudioInput
  │     │     └── ParticipantAudioOutput
  │     └── ChatContext (maintains)
  ├── STT (uses)
  ├── LLM (uses)
  ├── TTS (uses)
  └── VAD (uses)
```

## 附录 B: 事件类型速查

### STT Events
- `START_OF_SPEECH` - 检测到语音开始
- `INTERIM_TRANSCRIPT` - 临时识别结果
- `FINAL_TRANSCRIPT` - 最终识别结果
- `END_OF_SPEECH` - 检测到语音结束

### VAD Events
- `START_OF_SPEECH` - 语音活动开始
- `INFERENCE_DONE` - 推理完成（概率更新）
- `END_OF_SPEECH` - 语音活动结束

### LLM Events
- `ChatChunk` - 流式文本块
- `FunctionCall` - 工具调用

### TTS Events
- `SynthesizedAudio` - 合成音频帧
