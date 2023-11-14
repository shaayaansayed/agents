import logging
from typing import Dict, List

from action import Action
from config_manager import ConfigManager
from llm import OpenAILLM

logger = logging.getLogger("chat_logger")

SYSTEM_PROMPT_TEMPLATE = "{system_prompt}\n\nCurrent conversation:\n{chat_summary}\n{chat_messages}"
SUFFIX_PROMPT_TEMPLATE = "Based on the given information, continue the conversation on behalf of {name}. Have your response be as natural and coherent as possible."


class Agent:

    def __init__(self, name: str, state_roles: Dict[str, str], llm: OpenAILLM,
                 is_user: bool, begins: List) -> None:
        self.name = name
        self.state_roles = state_roles
        self.llm = llm
        self.is_user = is_user
        self.begins = begins
        self.environment = None

    @classmethod
    def from_config(cls, config: ConfigManager):
        agents = {}
        for agent_name, agent_info in config.get('agents').items():
            agent_begins = {}
            agent_state_roles = config.get_agent_state_roles(agent_name)
            for state_name, agent_role in agent_state_roles.items():
                begin_role, begin_query = config.get_state_begin_role_and_query(
                    state_name)
                if agent_role == begin_role:
                    agent_begins[state_name] = {
                        "is_begin": True,
                        "begin_query": begin_query
                    }

            agents[agent_name] = cls(agent_name,
                                     state_roles=agent_state_roles,
                                     llm=OpenAILLM(),
                                     is_user=config.is_agent_user(agent_name),
                                     begins=agent_begins)

        return agents

    def step(self, state, input=""):
        agent_begin = state.name in self.begins and self.begins[state.name].get(
            "is_begin")

        if self.is_user:
            response = f"{self.name}:{input}"
        else:
            if agent_begin:
                response = ''.join(self.begins[state.name]["begin_query"])
            else:
                response = self.act(state)

        if state.is_begin:
            state.is_begin = False
        if agent_begin:
            self.begins[state.name]['is_begin'] = False

        return Action(response=response,
                      role=self.state_roles[state.name],
                      name=self.name,
                      state_begin=state.is_begin,
                      agent_begin=agent_begin,
                      is_user=self.is_user)

    def act(self, state):
        system_prompt, suffix_prompt = self.compile(state)

        chat_summary, chat_messages = self.environment.get_memory()
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            system_prompt=system_prompt,
            chat_messages=chat_messages,
            chat_summary=chat_summary)

        return self.llm.get_response(system_prompt,
                                     suffix_prompt=suffix_prompt,
                                     log_messages=True)

    def compile(self, state):
        components = state.components[self.state_roles[state.name]]

        system_prompt_parts = [state.state_desc]
        for component_name, component in components.items():
            if isinstance(component, str):
                if component_name == "role_desc":
                    system_prompt_parts.append(component)

        system_prompt = "\n".join(system_prompt_parts)
        suffix_prompt = SUFFIX_PROMPT_TEMPLATE.format(name=self.name)

        return system_prompt, suffix_prompt
