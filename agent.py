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

def get_weather(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url, timeout=5)
        return response.text.strip()
    except:
        return f"Sorry, I could not get weather for {city} right now."

def search_web(query: str) -> str:
    try:
        url = f"https://ddg-api.herokuapp.com/search?query={query}&limit=2"
        response = requests.get(url, timeout=5)
        results = response.json()
        if results:
            snippets = [r.get("snippet", "") for r in results]
            return " ".join(snippets)[:500]
        return "No results found."
    except:
        return f"Sorry, I could not search for {query} right now."

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name e.g. Mumbai, Delhi"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for any question or topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly and helpful AI assistant named Arjun. "
                    "You can speak and understand multiple Indian languages including Hindi, English, Punjabi, Kannada, Tamil, Marathi, Gujarati, and Bhojpuri. "
                    "IMPORTANT: Always detect the language the user is speaking and reply in the EXACT SAME language. "
                    "If they speak Hindi, reply in Hindi. If they speak Tamil, reply in Tamil. If they mix languages, mix the same way. "
                    "You can check the weather and search the web when asked. "
                    "Keep your responses brief, warm, and conversational. "
                    "Never give long paragraphs — speak like a human would in a phone call."
                ),
            }
        ]
        context = LLMContext(messages, tools=tools)
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
