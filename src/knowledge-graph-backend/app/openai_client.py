# app/openai_client.py
import openai
from .config import settings

openai.api_key = settings.OPENAI_API_KEY


async def get_completion(prompt: str):
    response = await openai.Completion.acreate(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()
