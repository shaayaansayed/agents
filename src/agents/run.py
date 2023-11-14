import argparse
import logging
from typing import List

from agents.agent import Agent
from agents.environment import Environment
from agents.state_controller import StateController
from config_manager import ConfigManager
from utils import setup_logging

logger = logging.getLogger("chat_logger")


def init(config_path: str) -> (list, StateController, Environment):
    config = ConfigManager(config_path)
    state_controller = StateController(config)
    agents = Agent.from_config(config)
    environment = Environment.from_config(config, agents)
    return state_controller, environment, agents


def run(state_controller: StateController, environment: Environment,
        agents: List[Agent]):
    while True:
        curr_state, curr_agent = state_controller.next(environment, agents)
        logger.info(f"Current state: {curr_state.name}")

        user_input = input(f"{curr_agent.name}: ") if curr_agent.is_user else ""

        action = curr_agent.step(curr_state, user_input)
        action.process()
        environment.record_action(action, curr_state)


parser = argparse.ArgumentParser(description='Run the simulation.')
parser.add_argument('--config',
                    type=str,
                    help='Path to the configuration JSON file.')
parser.add_argument('--debug',
                    action='store_true',
                    required=False,
                    default=False,
                    help='Path to the configuration JSON file.')
args = parser.parse_args()

setup_logging(args.debug)
state_controller, environment, agents = init(args.config)
run(state_controller, environment, agents)
