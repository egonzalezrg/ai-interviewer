from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    langchain,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from graph import create_workflow


load_dotenv(".env.local")


class InterviewAgent(Agent):
    """Realtime voice agent used to conduct the interview."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a professional interviewer conducting a job interview. "
                "The LangGraph workflow will drive the conversation flow. "
                "Simply speak the questions and responses as they come from the graph. "
                "Be conversational, professional, and helpful throughout the interview process."
            )
        )


async def entrypoint(ctx: agents.JobContext):
    #connects the LangGraph interview workflow to LiveKit
    interview_workflow = create_workflow()
    lg_llm = langchain.LLMAdapter(
        graph=interview_workflow,
        stream_mode="custom",
    )
    #configures the realtime speech pipeline.
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
        ),
        llm=lg_llm,
        tts=cartesia.TTS(
            model="sonic-2",
            voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=InterviewAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    print("Starting interview workflow...")

    #requests the opening interview question when the session starts.
    await session.generate_reply(
        instructions=(
            "Begin the interview now. Give the opening greeting "
            "and ask the first interview question."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )