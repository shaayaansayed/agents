import logging
from typing import Dict, List

from agents.action import Action
from agents.llm import init_llm
from config_manager import ConfigManager

logger = logging.getLogger("chat_logger")

ACT_SYSTEM_PROMPT = "{system_prompt}\n\nCurrent conversation:\n{env_summary}\n{chat_messages}"
LAST_PROMPT = "\n\nBased on the given information, continue the conversation on behalf of {name}. Have your response be as natural and coherent as possible."


class Agent:

    def __init__(self, name: str, state_roles: Dict[str, str], llms: List,
                 is_user: bool, begins: List) -> None:
        self.name = name
        self.state_roles = state_roles
        self.llms = llms
        self.is_user = is_user
        self.begins = begins
        self.environment = None

    @classmethod
    def from_config(cls, config: ConfigManager):
        agents = {}
        for agent_name, agent_info in config.get('agents').items():
            agent_llms, agent_begins = {}, {}

            agent_state_roles = config.get_agent_state_roles(agent_name)
            for state_name, agent_role in agent_state_roles.items():
                begin_role, begin_query = config.get_state_begin_role_and_query(
                    state_name)
                if agent_role == begin_role:
                    agent_begins[state_name] = {
                        "is_begin": True,
                        "begin_query": begin_query
                    }
                # TODO: Initialize LLMs based on functions and tools...
                agent_llms[state_name] = init_llm()

            agents[agent_name] = cls(agent_name,
                                     state_roles=agent_state_roles,
                                     llms=agent_llms,
                                     is_user=config.is_agent_user(agent_name),
                                     begins=agent_begins)

        return agents

    def step(self, state, input=""):
        agent_begin = self.begins[state.name].get("is_begin", False)

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
            self.begins[state.name] = False
        state.chat_nums += 1

        return Action(response=response,
                      role=self.state_roles[state.name],
                      name=self.name,
                      state_begin=state.is_begin,
                      agent_begin=agent_begin,
                      is_user=self.is_user)

    def act(self, state):
        llm = self.llms[state.name]
        last_prompt, system_prompt = self.compile(state)

        env_summary, chat_messages = self.environment.get_memory()
        system_prompt = ACT_SYSTEM_PROMPT.format(system_prompt=system_prompt,
                                                 chat_messages=chat_messages,
                                                 env_summary=env_summary)
        logger.debug(f"Agent {self.name} is acting in state {state.name}.\n")
        return llm.get_response(None,
                                system_prompt,
                                last_prompt,
                                log_messages=True)

    def compile(self, state):
        components = state.components[self.state_roles[state.name]]

        system_prompt_parts = [state.env_desc]
        last_prompt_parts = []

        for component_name, component in components.items():
            if isinstance(component, str):
                if component_name == "role_desc":
                    system_prompt_parts.append(component)

        system_prompt = "\n".join(system_prompt_parts)
        last_prompt = "\n".join(last_prompt_parts)

        last_prompt_formatted = LAST_PROMPT.format(last_prompt=last_prompt,
                                                   name=self.name)

        return last_prompt_formatted, system_prompt
