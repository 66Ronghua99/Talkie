from dotenv import load_dotenv

import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import (
    silero,
    noise_cancellation,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from talkie.livekit_plugins.stt import create_stt_plugin
from talkie.livekit_plugins.llm import create_llm_plugin
from talkie.livekit_plugins.tts import create_tts_plugin
from talkie.livekit_plugins.turn_detector import SmartTurnVAD

load_dotenv(".env")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

server = AgentServer()

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext):
    stt_plugin = await create_stt_plugin()
    llm_plugin = await create_llm_plugin()
    tts_plugin = await create_tts_plugin()

    session = AgentSession(
        stt=stt_plugin,
        # stt="deepgram/nova-3:multi",
        llm=llm_plugin,
        tts=tts_plugin,
        # vad=silero.VAD.load(),
        # turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance. You should start by speaking in English."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)