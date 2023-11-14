import json
import logging

from llm import OpenAILLM
from state import State

logger = logging.getLogger("chat_logger")

STATE_TRANSITION_TEMPLATE = """{state_desc}

{state_transition_system_prompt}

Current conversation:

{chat_summary}

{chat_messages}
"""


class StateController:

    def __init__(self, config):
        self.config = config
        self.llm = OpenAILLM()

        self.init_states(config.get("states"))
        self.init_relation(config.get("relations"))

        self.controller_info = {
            state_name: states_dict["controller"]
            for state_name, states_dict in config.get("states", {}).items()
            if state_name != "end_state" and "controller" in states_dict
        }

        self.curr_state = self.states[config.get_root_state()]

    def init_states(self, states_info):
        self.states = {}
        for state_name, state_info in states_info.items():
            self.states[state_name] = State(
                name=state_name,
                state_desc=state_info.get('state_desc'),
                roles_info=state_info.get('roles'),
                begin_role=state_info.get('begin_role'),
                begin_query=state_info.get('begin_query'))

    def init_relation(self, relations):
        for state_name, state_relation in relations.items():
            self.states[state_name].next_states = {}
            for idx, next_state_name in state_relation.items():
                self.states[state_name].next_states[idx] = self.states.get(
                    next_state_name)

    def transit(self, environment):

        if len(self.curr_state.next_states) == 1:
            next_state_key = "0"
        else:
            controller_info = self.controller_info[self.curr_state.name]

            chat_summary, chat_messages = environment.get_memory()
            system_prompt = STATE_TRANSITION_TEMPLATE.format(
                state_desc=self.curr_state.state_desc,
                state_transition_system_prompt=controller_info.get(
                    "state_transition_system_prompt"),
                chat_summary=chat_summary,
                chat_messages=chat_messages)

            response = self.llm.get_response(system_prompt,
                                             log_messages=True,
                                             json_mode=True)

            try:
                response = json.loads(response)
                next_state_key = response['state']
            except json.JSONDecodeError:
                next_state_key = "0"

        return self.curr_state.next_states.get(next_state_key)

    def next(self, environment, agents):

        # Check if entering this state for the first time. If so, update the chat history index and return the agent assigned the 'begin role' for this state.
        if self.curr_state.is_begin:
            begin_role, _ = self.config.get_state_begin_role_and_query(
                self.curr_state.name)
            agent_name = self.config.get_agent_for_state_role(
                self.curr_state.name, begin_role)
            return self.curr_state, agents[agent_name]

        # Determine the next state
        next_state = self.transit(environment=environment)

        # In the first entry to a state with a defined 'begin role' and 'begin query', update the chat history index and assign the 'begin' role to the agent.
        if next_state.is_begin:
            next_agent = self.config.get_agent_for_state_role(
                next_state.name, next_state.begin_role)
        else:
            next_role = self.curr_state.update_to_next_role()

            # ensure conversation flips
            # If there's a new state change, find the person who spoke last in the state before last and ensure they are a different agent
            # if next_state.name != self.curr_state.name:
            #     if agent.name == next_agent.name :
            #         next_role = self.curr
            # next_role = self.curr_state.update_to_next_role()
            next_agent = self.config.get_agent_for_state_role(
                self.curr_state.name, next_role)

        self.curr_state = next_state
        return next_state, agents[next_agent]
