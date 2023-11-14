import argparse
import time

import gradio as gr
from agents.agent import Agent
from agents.environment import Environment
from agents.state_controller import StateController
from config_manager import ConfigManager


def init(config_path: str) -> (list, StateController, Environment):
    config = ConfigManager(config_path)
    sc = StateController(config)
    agents = Agent.from_config(config)
    environment = Environment.from_config(config, agents)
    return agents, sc, environment


def run(agents: list, sc: StateController, environment: Environment):
    while not sc.finished:
        curr_state, curr_agent = sc.next(environment, agents)

        user_input = input(f"{curr_agent.name}: ") if curr_agent.is_user else ""

        action = curr_agent.step(curr_state, user_input)
        memory = action.process()
        environment.update_memory(memory, curr_state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the simulation.')
    parser.add_argument('--config',
                        type=str,
                        help='Path to the configuration JSON file.')
    parser.add_argument('--debug',
                        type=bool,
                        required=False,
                        default=False,
                        help='Path to the configuration JSON file.')
    args = parser.parse_args()

    agents, sc, environment = init(args.config)

    def cycle(user_input):
        curr_state, curr_agent = sc.next(environment, agents)
        action = curr_agent.step(curr_state, user_input)
        memory = action.process()
        environment.update_memory(memory, curr_state)

        curr_state, curr_agent = sc.next(environment, agents)
        action = curr_agent.step(curr_state)
        memory = action.process()
        environment.update_memory(memory, curr_state)
        return action.response

    def get_response(user_input, history):
        return cycle(user_input)

    chatbot = gr.ChatInterface(fn=get_response,
                               examples=[],
                               title="ClosedLoop Health Coach")
    chatbot.launch()
