import os
import requests
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.getenv("OPENROUTER_API_KEY")
API_ENDPOINT = os.getenv("OPENROUTER_API_ENDPOINT")

def post(model:str, messages:list[dict[str, str]], temperature:float=0.9, top_p:float=0.99):
    """ Post to openrouter (i.e use the model names from the openrouter dashboard)"""
    response = requests.post(
        API_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p
        }
    )

    response_json = response.json()
    
    # Check for errors
    if 'error' in response_json or 'choices' not in response_json:
        raise ValueError(f"API Error: {response_json['error']['message']}")

    return response_json['choices'][0]['message']['content']