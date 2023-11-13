import argparse

from agents.agent import Agent
from agents.environment import Environment
from agents.sop import SOP
from config_manager import ConfigManager
from utils import setup_logging


def init(config_path: str) -> (list, SOP, Environment):
    config = ConfigManager(config_path)
    sop = SOP(config)
    agents = Agent.from_config(config)
    environment = Environment.from_config(config, agents)
    return agents, sop, environment


def run(agents: list, sop: SOP, environment: Environment):
    while not sop.finished:
        curr_state, curr_agent = sop.next(environment, agents)

        user_input = input(f"{curr_agent.name}: ") if curr_agent.is_user else ""

        action = curr_agent.step(curr_state, user_input)
        memory = action.process()
        environment.update_memory(memory, curr_state)


parser = argparse.ArgumentParser(description='Run the SOP simulation.')
parser.add_argument('--config',
                    type=str,
                    help='Path to the configuration JSON file.')
args = parser.parse_args()

setup_logging()
agents, sop, environment = init(args.config)
run(agents, sop, environment)
