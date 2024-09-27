    
    # openai/o1-preview-2024-09-12
    
    # async def post(self, session:ClientSession, chat_history:Optional[List] = None):
    #     async with session.post(
    #         "https://openrouter.ai/api/v1/chat/completions"
    #         headers={
    #             "Content-Type": "application/json",
    #             "Authorization": f"Bearer {self.model_tier.api_key}",
    #         },
    #         json={
    #             "model": "openai/o1-preview-2024-09-12",
    #             "messages": self.messages + (chat_history or []),
    #             "temperature": self.temperature,
    #             "top_p": self.top_p
    #         },
    #     ) as response:
    #         response_json = await response.json()

    #         print(self.messages)
    #         print(chat_history)
    #         print(response_json)
    #         return response_json["choices"][0]["message"]["content"]    