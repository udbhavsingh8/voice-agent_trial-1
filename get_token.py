import os
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants
load_dotenv()
token = AccessToken(
    os.getenv("LIVEKIT_API_KEY"),
    os.getenv("LIVEKIT_API_SECRET")
).with_identity("guest") \
 .with_name("Guest") \
 .with_grants(VideoGrants(room_join=True, room="asha-demo")) \
 .to_jwt()
print("\n--- COPY THIS TOKEN ---")
print(token)
print("-----------------------\n")
