import os
import asyncio
import aiohttp
import requests
from dotenv import load_dotenv
from loguru import logger
from livekit.api import AccessToken, VideoGrants

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamHttpTTSService
from pipecat.services.sarvam.llm import SarvamLLMService

from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams

load_dotenv(override=True)

PERSONAS = {
    "default": (
        "You are Arjun, a friendly and helpful AI assistant. "
        "You can speak Hindi, English, Punjabi, Kannada, Tamil, Marathi, Gujarati, and Bhojpuri. "
        "Always reply in the same language the user speaks. "
        "Keep responses brief and conversational like a phone call."
    ),
    "doctor": (
        "You are Dr. Arjun, a friendly and knowledgeable medical assistant. "
        "You can speak Hindi, English, Punjabi, Kannada, Tamil, Marathi, Gujarati, and Bhojpuri. "
        "Always reply in the same language the user speaks. "
        "Help users understand symptoms, medicines, and when to see a doctor. "
        "Always remind users you are an AI and they should consult a real doctor for serious issues. "
        "Keep responses brief, caring, and easy to understand."
    ),
    "support": (
        "You are Arjun, a friendly customer support assistant. "
        "You can speak Hindi, English, Punjabi, Kannada, Tamil, Marathi, Gujarati, and Bhojpuri. "
        "Always reply in the same language the user speaks. "
        "Help users resolve issues, answer questions, and escalate when needed. "
        "Be patient, polite, and solution focused. Keep responses brief and clear."
    ),
    "tutor": (
        "You are Arjun, a friendly and encouraging tutor for students. "
        "You can speak Hindi, English, Punjabi, Kannada, Tamil, Marathi, Gujarati, and Bhojpuri. "
        "Always reply in the same language the user speaks. "
        "Help students understand concepts in maths, science, history, and more. "
        "Use simple examples and analogies. Be encouraging and patient. "
        "Keep explanations short and check if the student understood."
    ),
}

GREETINGS = {
    "default": "Hello! I am Arjun, your assistant. How can I help you today?",
    "doctor": "Hello! I am Dr. Arjun, your medical assistant. How are you feeling today?",
    "support": "Hello! I am Arjun from customer support. How can I assist you today?",
    "tutor": "Hello! I am Arjun, your personal tutor. What would you like to study today?",
}

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
    persona = os.getenv("PERSONA", "default")
    system_prompt = PERSONAS.get(persona, PERSONAS["default"])
    greeting = GREETINGS.get(persona, GREETINGS["default"])

    logger.info(f"Starting bot with persona: {persona}")

    token = generate_token()
    transport = LiveKitTransport(
        url=os.getenv("LIVEKIT_URL"),
        token=token,
        room_name="asha-demo",
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                stop_secs=0.5
            )),
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
        messages = [{"role": "system", "content": system_prompt}]
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
        task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=True)
        )

        @transport.event_handler("on_participant_connected")
        async def on_participant_connected(transport, participant):
            logger.info(f"Participant connected: {participant}")
            messages.append({
                "role": "system",
                "content": f"Greet the user by saying exactly: {greeting}"
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
