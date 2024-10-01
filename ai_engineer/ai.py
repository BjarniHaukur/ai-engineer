import logging
from typing import Union

import anthropic
import backoff
import openai
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

Message = Union[AIMessage, HumanMessage, SystemMessage]


class AI:
    def __init__(self, model_name:str, temperature:float=0.7):
        self.model_name, self.temperature = model_name, temperature
        self.llm = self._create_chat_model()

    @backoff.on_exception(backoff.expo, [openai.RateLimitError, anthropic.RateLimitError], max_tries=7, max_time=45)  # retry API call with increasing delay
    def next(self, messages:list[Message])->list[Message]:
        messages = self._collapse_messages(messages)
        return messages + [self.llm.invoke(messages)]

    def _collapse_messages(self, messages:list[Message])->list[Message]:
        """ Combine subsequent messages of the same type """
        collapsed = []
        for message in messages:
            if collapsed and isinstance(message, type(collapsed[-1])):
                collapsed[-1].content += "\n" + message.content
            else:
                collapsed.append(message)
        return collapsed

    def _create_chat_model(self)->BaseChatModel:
        if "gpt" in self.model_name or "o1" in self.model_name:
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
