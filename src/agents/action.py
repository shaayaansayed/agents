from dataclasses import dataclass

from agents.memory import Memory


@dataclass(frozen=True)  # frozen=True makes the dataclass immutable
class Action:
    response: str
    name: str
    role: str
    state_begin: bool
    agent_begin: bool
    is_user: bool

    def process(self):
        formatted_response = ''.join(self.response)
        parse = f"{self.name}:"
        while parse in formatted_response:
            index = formatted_response.index(parse) + len(parse)
            formatted_response = formatted_response[index:]

        if not self.is_user:
            print(f"{self.name} ({self.role}): {formatted_response}")

        return Memory(self.role, self.name, formatted_response)
