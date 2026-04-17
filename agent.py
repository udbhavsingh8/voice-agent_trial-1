import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from loguru import logger
from livekit.api import AccessToken, VideoGrants

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import LLMRunFrame

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamHttpTTSService
from pipecat.services.sarvam.llm import SarvamLLMService

from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams

load_dotenv(override=True)

def generate_token():
    token = AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET")
    ).with_identity("arjun-bot") \
     .with_name("Arjun") \
     .with_grants(VideoGrants(room_join=True, room="asha-demo")) \
     .to_jwt()
    return token

async def bot():
    token = generate_token()
    transport = LiveKitTransport(
        url=os.getenv("LIVEKIT_URL"),
        token=token,
        room_name="asha-demo",
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )
    stt = SarvamSTTService(
        api_key=os.getenv("SARVAM_API_KEY"),
        settings=SarvamSTTService.Settings(
            model="saarika:v2.5",
            language="unknown",
        ),
    )
    llm = SarvamLLMService(
        api_key=os.getenv("SARVAM_API_KEY"),
        model="sarvam-30b",
    )
    async with aiohttp.ClientSession() as session:
        tts = SarvamHttpTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            aiohttp_session=session,
            settings=SarvamHttpTTSService.Settings(
                model="bulbul:v2",
                voice="karun",
            ),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    ""You are a friendly and helpful AI assistant named Arjun. "
"You can speak and understand multiple Indian languages including Hindi, English, Punjabi, Kannada, Tamil, Marathi, Gujarati, and Bhojpuri. "
"IMPORTANT: Always detect the language the user is speaking and reply in the EXACT SAME language. "
"If they speak Hindi, reply in Hindi. If they speak Tamil, reply in Tamil. If they mix languages, mix the same way. "
"Keep your responses brief, warm, and conversational. "
"Never give long paragraphs — speak like a human would in a phone call."You are a friendly and helpful AI assistant named Arjun. "
                    "You speak naturally in Indian English and can understand Hindi too. "
                    "Keep your responses brief, warm, and conversational. "
                    "Never give long paragraphs — speak like a human would in a phone call."
                ),
            }
        ]
        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)
        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])
        task = PipelineTask(pipeline)

        @transport.event_handler("on_participant_connected")
        async def on_participant_connected(transport, participant):
            logger.info(f"Participant connected: {participant}")
            messages.append({
                "role": "system",
                "content": "Greet the user by saying exactly: Hello UD, I am Arjun, your assistant. How can I help you today?"
            })
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_participant_disconnected")
        async def on_participant_disconnected(transport, participant):
            logger.info(f"Participant disconnected: {participant}")
            await task.cancel()

        runner = PipelineRunner()
        await runner.run(task)

if __name__ == "__main__":
    asyncio.run(bot())
