import json
import os
import random

from agents.llm import init_llm
from agents.memory import Memory
from agents.prompts import *
from agents.state import State
from agents.utils import extract, get_relevant_history

ENVIRONMENT_TEMPLATE = "Here are the description of current scenario:{env_desc};\n"
TRANSIT_SYSTEM_PROMPT = "{env_desc};\n{judge_system_prompt}\n"
TRANSIT_MESSAGE_TEMPLATE = "{environment_summary};\n Here is the The chat history:\n {chat_history_messages};\nHere is the last query you especially need to pay attention:\n{last_message};\n Here is the relevant conversation:\n{relevant_history}\n\n"


class SOP:

    def __init__(self, config):
        self.config = config
        self.llm = init_llm()

        self.init_states(config.get("states"))
        self.init_relation(config.get("relations"))

        self.controller_info = {
            state_name: states_dict["controller"]
            for state_name, states_dict in config.get("states", {}).items()
            if state_name != "end_state" and "controller" in states_dict
        }

        self.curr_state = self.states[config.get_root_state()]
        self.finish_state_name = config.get("finish_state_name", "end_state")
        self.finished = False

    def init_states(self, states_dict):
        self.states = {}
        for state_name, state_dict in states_dict.items():
            state_dict["name"] = state_name
            self.states[state_name] = State(**state_dict)

    def init_relation(self, relations):
        for state_name, state_relation in relations.items():
            for idx, next_state_name in state_relation.items():
                self.states[state_name].next_states[idx] = self.states.get(
                    next_state_name)

    def transit(self, chat_history, **kwargs):

        if len(self.curr_state.next_states) == 1:
            next_state_key = "0"
        else:
            controller_info = self.controller_info[self.curr_state.name]
            max_chat_nums = controller_info.get("max_chat_nums", 1000)

            if self.curr_state.chat_nums >= max_chat_nums:
                return self.curr_state.next_states.get("1", self.curr_state)

            env_desc = self.curr_state.env_desc or ""
            transit_system_prompt = TRANSIT_SYSTEM_PROMPT.format(
                environment_prompt=ENVIRONMENT_TEMPLATE.format(
                    environment_prompt=env_desc),
                judge_system_prompt=controller_info.get("judge_system_prompt",
                                                        ""))

            transit_last_prompt = controller_info.get("judge_last_prompt", "")
            relevant_history, environment = kwargs.get(
                "relevant_history"), kwargs.get("environment")
            environment_summary = environment.shared_memory[
                "short_term_memory"] if environment else ""
            chat_history_messages = Memory.get_chat_history(chat_history)
            last_message = chat_history[-1].get_query() if chat_history else ""

            transit_message = TRANSIT_MESSAGE_TEMPLATE.format(
                relevant_history=relevant_history,
                environment_summary=environment_summary,
                chat_history_messages=chat_history_messages,
                last_message=last_message)

            chat_messages = [{"role": "user", "content": transit_message}]
            extract_words = kwargs.get("judge_extract_words", "end")
            response = self.llm.get_response(chat_messages,
                                             transit_system_prompt,
                                             transit_last_prompt)

            next_state_key = response if response.isdigit() else extract(
                response, extract_words)
            if not next_state_key.isdigit():
                next_state_key = "0"

        return self.curr_state.next_states.get(next_state_key, self.curr_state)

    def route(self, chat_history, **kwargs):
        """
        Determine the role that needs action based on the current situation
        Return : 
        current_agent(Agent) : the next act agent
        """

        agents = kwargs["agents"]

        if len(self.curr_state.roles) == 1:
            next_role = self.curr_state.roles[0]
        else:
            relevant_history = kwargs.get("relevant_history")
            controller_type = (
                self.controller_info[self.curr_state.name]["controller_type"]
                if "controller_type"
                in self.controller_info[self.curr_state.name] else "order")

            if controller_type == "rule":
                controller_info = self.controller_info[self.curr_state.name]

                call_last_prompt = controller_info[
                    "call_last_prompt"] if "call_last_prompt" in controller_info else ""

                allocate_prompt = ""
                roles = list(set(self.curr_state.roles))
                for role in roles:
                    allocate_prompt += eval(Allocate_component)

                call_system_prompt = controller_info[
                    "call_system_prompt"] if "call_system_prompt" in controller_info else ""
                environment_prompt = eval(
                    Get_environment_prompt
                ) if self.curr_state.environment_prompt else ""
                # call_system_prompt + environment + allocate_prompt
                call_system_prompt = eval(Call_system_prompt)

                query = chat_history[-1].get_query()
                last_name = chat_history[-1].send_name
                # last_prompt: note + last_prompt + query
                call_last_prompt = eval(Call_last_prompt)

                chat_history_message = Memory.get_chat_history(chat_history)
                # Intermediate historical conversation records
                chat_messages = [{
                    "role": "user",
                    "content": eval(Call_message),
                }]

                extract_words = controller_info[
                    "call_extract_words"] if "call_extract_words" in controller_info else "end"

                response = self.llm.get_response(chat_messages,
                                                 call_system_prompt,
                                                 call_last_prompt)

                # get next role
                next_role = extract(response, extract_words)
            elif controller_type == "order":
                if not self.curr_state.current_role:
                    next_role = self.curr_state.roles[0]
                else:
                    self.curr_state.index += 1
                    self.curr_state.index = (self.curr_state.index) % len(
                        self.curr_state.roles)
                    next_role = self.curr_state.roles[self.curr_state.index]

        if next_role not in self.curr_state.roles:
            next_role = random.choice(self.curr_state.roles)

        self.curr_state.current_role = next_role

        next_agent = agents[self.roles_to_names[self.curr_state.name]
                            [next_role]]

        return next_agent

    def next(self, environment, agents):

        # Check if entering this state for the first time. If so, update the chat history index and return the agent assigned the 'begin role' for this state.
        if self.curr_state.is_begin:
            environment.curr_chat_history_idx = len(
                environment.shared_memory["long_term_memory"])
            begin_role, _ = self.config.get_state_begin_role_and_query(
                self.curr_state.name)
            agent_name = self.config.get_agent_for_state_role(
                self.curr_state.name, begin_role)
            agent = agents[agent_name]
            return self.curr_state, agent

        # Generate relevant history
        long_term_memory = environment.shared_memory['long_term_memory']
        chat_embeddings = environment.shared_memory['chat_embeddings']
        query = long_term_memory[-1].content

        relevant_history = get_relevant_history(
            query,
            long_term_memory[:-1],
            chat_embeddings[:-1],
        )
        relevant_history = Memory.get_chat_history(relevant_history)

        # Determine the next state
        next_state = self.transit(
            chat_history=long_term_memory[environment.curr_chat_history_idx:],
            relevant_history=relevant_history,
            environment=environment,
        )

        # If the next state is the designated finish state, mark the simulation as complete and return no agent.
        if next_state.name == self.finish_state_name:
            self.finished = True
            return None, None

        # Update to the next state and adjust its index
        self.curr_state = next_state
        if next_state.name != self.curr_state.name:
            self.curr_state.index = (self.curr_state.index + 1) % len(
                self.curr_state.roles)

        # In the first entry to a state with a defined 'begin role' and 'begin query', update the chat history index and assign the 'begin' role to the agent.
        if self.curr_state.is_begin and self.curr_state.begin_role:
            environment.current_chat_history_idx = len(long_term_memory)
            agent_name = self.roles_to_names[self.curr_state.name][
                self.curr_state.begin_role]
            agent = agents[agent_name]
            return self.curr_state, agent

        # Determine the next agent
        next_agent = self.route(
            chat_history=long_term_memory[environment.curr_chat_history_idx:],
            agents=agents,
            relevant_history=relevant_history,
        )

        return self.curr_state, next_agent
