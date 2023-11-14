import logging
import os
import time
from typing import Optional

from openai import OpenAI

logger = logging.getLogger("chat_logger")


class OpenAILLM:

    MODEL = "gpt-4-1106-preview"
    TEMPERATURE = 0.8
    SEED = 42
    WAIT_TIME = 20

    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=os.getenv("API_KEY", api_key))

    def get_response(self,
                     system_prompt: str,
                     suffix_prompt: Optional[str] = None,
                     log_messages: bool = False,
                     json_mode: bool = False) -> None:

        messages = [{"role": "system", "content": system_prompt}]

        if suffix_prompt:
            last_message = messages[-1]['content']
            messages[-1]["content"] = f"{last_message}\n\n{suffix_prompt}"

        if log_messages:
            log_message = '\n'.join(
                f"{item['role']}: {item['content']}" for item in messages)
            logger.debug(f"Response generated with inputs:\n{log_message}")

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=messages,
                    temperature=self.TEMPERATURE,
                    seed=self.SEED,
                    response_format={"type": "json_object"}
                    if json_mode else None)
                break
            except Exception as e:
                if "maximum context length is" in str(e):
                    messages.pop(1 if len(messages) > 1 else 0)
                else:
                    print(
                        f"Please wait {self.WAIT_TIME} seconds and resend later ..."
                    )
                    time.sleep(self.WAIT_TIME)

        return response.choices[0].message.content
