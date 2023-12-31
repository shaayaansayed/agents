import logging
import os
import time

from agents.memory import Memory
from openai import OpenAI


class OpenAILLM:

    MODEL = "gpt-4-1106-preview"
    TEMPERATURE = 0.8
    SEED = 42
    WAIT_TIME = 20

    def __init__(self, api_key=None, max_chat_history=10):
        self.max_chat_history = int(
            os.getenv("MAX_CHAT_HISTORY", max_chat_history))
        self.client = OpenAI(api_key=os.getenv("API_KEY", api_key))
        self.logger = logging.getLogger('chat_logger')

    def get_response(self,
                     chat_history,
                     system_prompt,
                     last_prompt=None,
                     log_system_fingerprint=False,
                     **kwargs):
        messages = [{
            "role": "system",
            "content": system_prompt
        }] if system_prompt else []

        if chat_history:
            chat_history = chat_history[-self.max_chat_history:]
            messages.extend(
                memory.get_gpt_message("user") if isinstance(memory, Memory
                                                            ) else memory
                for memory in chat_history)

        if last_prompt:
            messages[-1]["content"] += last_prompt

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=messages,
                    temperature=self.TEMPERATURE,
                    seed=self.SEED)
                break
            except Exception as e:
                if "maximum context length is" in str(e):
                    messages.pop(1 if len(messages) > 1 else 0)
                else:
                    print(
                        f"Please wait {self.WAIT_TIME} seconds and resend later ..."
                    )
                    time.sleep(self.WAIT_TIME)

        if log_system_fingerprint:
            self.logger.info(
                f"system fingerprint: {response.system_fingerprint}")

        return response.choices[0].message.content


def init_llm():
    return OpenAILLM()
