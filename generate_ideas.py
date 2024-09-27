import os
import dotenv
import requests
from typing import List, Optional

dotenv.load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_ENDPOINT = os.getenv("OPENROUTER_API_ENDPOINT")

def post(self, chat_history: Optional[List] = None):
    response = requests.post(
        API_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        json={
            "model": "openai/o1-preview-2024-09-12",
            "messages": chat_history or [],
            "temperature": self.temperature,
            "top_p": self.top_p
        },
    )
    response_json = response.json()

    print(self.messages)
    print(chat_history)
    print(response_json)
    return response_json["choices"][0]["message"]["content"]