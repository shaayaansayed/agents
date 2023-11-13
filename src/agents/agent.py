import json
import os

from agents.action import Action
from agents.Component import LastComponent, OutputComponent, PromptComponent
from agents.llm import init_llm

LAST_PROMPT = "{last_prompt}. Please continue the conversation on behalf of {name} based on the given information. Make your response appear as natural and coherent as possible."
SUMMARY_PROMPT = "{summary_prompt};\nHere is the past summary:{short_term_memory};\nHere is the new chat_history:\n{new_conversations};\nPlease summarize the information above."


class Agent:

    def __init__(self, name: str, agent_state_roles: dict, **kwargs) -> None:
        self.state_roles = agent_state_roles
        self.name = name
        self.style = kwargs.get("style")
        self.llms = kwargs.get("llms", {})
        self.is_user = kwargs.get("is_user", False)
        self.begins = kwargs.get("begins", False)
        self.long_term_memory = []
        self.short_term_memory = ""
        self.environment = None

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path, 'r') as file:
            config = json.load(file)

        roles_to_names, agents = {}, {}

        for agent_name, agent_info in config["agents"].items():
            agent_llms, agent_begins = {}, {}

            for state_name, agent_role in agent_info["roles"].items():
                agent_begins[state_name] = {
                    "is_begin":
                        config["states"].get(state_name, {}).get("begin_role")
                        == agent_role,
                    "begin_query":
                        config["states"].get(state_name,
                                             {}).get("begin_query", " ")
                }

                roles_to_names.setdefault(state_name,
                                          {})[agent_role] = agent_name
                agent_llms[state_name] = init_llm()

            agents[agent_name] = cls(agent_name,
                                     agent_info.get("roles"),
                                     llms=agent_llms,
                                     is_user=agent_info.get("is_user", False),
                                     style=agent_info.get("style", ""),
                                     begins=agent_begins)
        names_to_roles = {
            state_name: {
                name: role for role, name in roles_to_names[state_name].items()
            } for state_name in roles_to_names
        }

        return agents, roles_to_names, names_to_roles

    def step(self, state, input=""):
        agent_begin = self.begins[state.name].get("is_begin", False)

        if self.is_user:
            response = f"{self.name}:{input}"
        else:
            # Update agent's long term memory with an observation from the environment.
            if len(self.environment.shared_memory["long_term_memory"]) > 0:
                self.long_term_memory.append(self.environment.observe(self))

            if agent_begin:
                response = ''.join(self.begins[state.name]["begin_query"])
            else:
                response = self.act(state)

        if agent_begin:
            self.begins[state.name]["is_begin"] = False
        if state.is_begin:
            state.is_begin = False
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

        return llm.get_response(self.long_term_memory,
                                system_prompt,
                                last_prompt,
                                log_system_fingerprint=True)

    def update_memory(self, memory, state):
        # The agents long term memory stores the agent's output as a list
        # The agent's specific output is stored as "assistant"
        self.long_term_memory.append({
            "role": "assistant",
            "content": memory.content
        })

        max_chat_history = int(os.environ.get("MAX_CHAT_HISTORY"))

        # If the environment's shared long term memory exceeds the max, summarize the memory and store it as short_term_memory.
        long_term_memory = self.environment.shared_memory["long_term_memory"][
            self.environment.curr_chat_history_idx:]
        last_conversation_idx = self.environment._get_agent_last_conversation_idx(
            self, long_term_memory)
        if len(long_term_memory) - last_conversation_idx >= max_chat_history:
            current_role = self.state_roles[state.name]
            current_component = state.components[current_role]

            new_conversations = self.environment._get_agent_new_memory(
                self, long_term_memory)

            summary_prompt = f"""Your name is {self.name}, your role is{current_component["style"].role}, your task is {current_component["task"].task}.\n"""

            summary_prompt_formatted = SUMMARY_PROMPT.format(
                summary_prompt=summary_prompt,
                new_conversations=new_conversations,
                short_term_memory=self.short_term_memory)
            memory_summary = self.llms[state.name].get_response(
                summary_prompt_formatted)
            self.short_term_memory = memory_summary

    def compile(self, state):
        components = state.components[self.state_roles[state.name]]

        system_prompt_parts = [state.environment_prompt]
        last_prompt_parts = []

        for component in components.values():
            prompt = component.get_prompt(self)
            if isinstance(component, (OutputComponent, LastComponent)):
                last_prompt_parts.append(prompt)
            elif isinstance(component, PromptComponent):
                system_prompt_parts.append(prompt)

        system_prompt = "\n".join(system_prompt_parts)
        last_prompt = "\n".join(last_prompt_parts)

        last_prompt_formatted = LAST_PROMPT.format(last_prompt=last_prompt,
                                                   name=self.name)

        return last_prompt_formatted, system_prompt
