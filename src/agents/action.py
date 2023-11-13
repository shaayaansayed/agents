import logging
from dataclasses import dataclass, field

from agents.memory import Memory


@dataclass
class Action:
    response: str
    name: str
    role: str
    state_begin: bool
    agent_begin: bool
    is_user: bool
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger('chat_logger'), init=False)

    def process(self):
        formatted_response = self.response
        parse = f"{self.name}:"
        while parse in formatted_response:
            index = formatted_response.index(parse) + len(parse)
            formatted_response = formatted_response[index:]

        if not self.is_user:
            self.logger.info(
                f"{self.name} ({self.role}): {formatted_response}\n")

        return Memory(self.role, self.name, formatted_response)
