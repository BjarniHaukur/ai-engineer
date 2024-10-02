import os
from typing import Union

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

Message = Union[AIMessage, HumanMessage, SystemMessage]


class ChatOpenRouter(ChatOpenAI):
    def __init__(self, model_name:str, **kwargs):
        super().__init__(openai_api_base="https://openrouter.ai/api/v1", openai_api_key=os.getenv("OPENROUTER_API_KEY"), model_name=model_name, **kwargs)

class AI:
    def __init__(self, model_name:str, temperature:float=0.7):
        self.model_name, self.temperature = model_name, temperature
        self.llm = self._create_chat_model()

    #@backoff.on_exception(backoff.expo, [openai.RateLimitError, anthropic.RateLimitError], max_tries=7, max_time=45)  # retry API call with increasing delay
    def next(self, messages:list[Message])->list[Message]:
        return messages + [self.llm.invoke(messages)]

    def _create_chat_model(self)->BaseChatModel:
        if "o1" in self.model_name:
            return ChatOpenRouter(
                model_name=self.model_name,
                temperature=self.temperature,
                streaming=True,
                # callbacks=[StreamingStdOutCallbackHandler()],  # basically a hack to get o1 early, 
            )
        elif "gpt" in self.model_name:
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
            )
        elif "claude" in self.model_name:
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                max_tokens_to_sample=4096,
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")





    # def _collapse_messages(self, messages:list[Message])->list[Message]:
    #     """ Combine subsequent messages of the same type """
    #     collapsed = []
    #     for message in messages:
    #         if collapsed and isinstance(message, type(collapsed[-1])):
    #             collapsed[-1].content += "\n" + message.content
    #         else:
    #             collapsed.append(message)
    #     return collapsed